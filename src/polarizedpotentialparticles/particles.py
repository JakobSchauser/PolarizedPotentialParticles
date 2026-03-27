from __future__ import annotations

from typing import TYPE_CHECKING
from polarizedpotentialparticles.custom_conv import CustomNNConv, HNNConv, EHNNConv, PolarizedHNNConv
from polarizedpotentialparticles.particle_types import ParticleType
import torch
import torch.nn.functional as F
from torch_geometric.nn import radius_graph

if TYPE_CHECKING:
    from polarizedpotentialparticles.configs import Config



def uniform_circular_distribution(num_particles, device=None):
    radius = 0.3
    # returns [num_particles, 2] uniformly sampled in a disk of given radius
    theta = 2 * torch.pi * torch.rand(num_particles, device=device)
    r = radius * torch.sqrt(torch.rand(num_particles, device=device))
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.stack([x, y], dim=1)


def uniform_circular_distribution_deterministic(num_particles, noise, config, device=None):
    # radius=0.6
    # radius=0.9
    # radius=0.8
    radius=config.starting_radius

    i = torch.arange(num_particles, device=device, dtype=torch.float32)

    # Golden angle
    golden_angle = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))

    theta = i * golden_angle

    # Uniform area density
    r = radius * torch.sqrt((i + 0.5) / num_particles)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)    
    
    # add a small amount of noise to break perfect symmetry
    x += noise * torch.randn_like(x)
    y += noise * torch.randn_like(y)

    return torch.stack((x, y), dim=1)

def uniform_circular_distribution_batch(num_particles, batch_size, noise, config, device=None):
    noise = 0.00
    base_pos = uniform_circular_distribution_deterministic(num_particles, noise=0.00, config= config, device=device)
    pos = base_pos.repeat(batch_size, 1)  # shape [batch_size * num_particles, 2]
    pos += noise * torch.randn_like(pos)  # add a small amount of noise to break perfect symmetry

    return pos



class ParticleOld(torch.nn.Module):
    particle_type_name = ParticleType.PARTICLE_OLD

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        
        self.message_conv : torch.nn.Module | None = None
        self.own_state_nn : torch.nn.Module | None = None
        self.message_to_output_layer : torch.nn.Module | None = None

        self.setup()



    def setup(self):
        self.initialize_architecture()

    def initialize_architecture(self):
        # Message NN
        self.message_conv = CustomNNConv(self.config)
        self.message_to_output_layer = torch.nn.Linear(
            self.config.particle_config.message_latent_dim,
            self.config.out_dim,
        )

        if self.config.particle_config.zero_initialization:
            with torch.no_grad():
                self.message_to_output_layer.weight.zero_()
                self.message_to_output_layer.bias.zero_()

    def update(self, output, x):
        # assert self.x is not None

        # output: [num_nodes, out_dim]
        # # [dx, dy, dpol_x, dpol_y, d_hidden1, d_hidden2, ...]
        
        end_of_spatial_dims = self.config.N_spatial_dim

        # move spatially along the direction of the first polarization vector
        move_update = output[:, :end_of_spatial_dims] * self.config.simulation_config.dt



        polarization = x[:, end_of_spatial_dims:end_of_spatial_dims + self.config.N_spatial_dim]  # [num_nodes, N_spatial_dim]

        # rotate the move update in the basis of the polarization vector
        # this way the particle moves in its local basis
        orthogonal = torch.stack((-polarization[:, 1], polarization[:, 0]), dim=1)  # [num_nodes, N_spatial_dim]
        spatial_update = move_update * polarization + move_update * orthogonal  # [num_nodes, N_spatial_dim]
        
        spatial = x[:, :end_of_spatial_dims] + spatial_update  # [num_nodes, N_spatial_dim]
         # update the rest
        rest = x[:,end_of_spatial_dims:] + output[:, end_of_spatial_dims:] * self.config.simulation_config.dt

        # normalize the polarization block to unit length without in-place slicing (keeps autograd happy)
        end = 2 * self.config.N_polarizations
        pol = rest[:, :end]
        pol = F.normalize(pol, p=2, dim=1, eps=1e-8)

        # rebuild x to avoid in-place grad issues on a view
        x = torch.cat((spatial, pol, rest[:, end:]), dim=1)

        return x
    
    def message_to_output(self, message):
        assert self.message_to_output_layer is not None
        out =  self.message_to_output_layer(message)

        # clip the output to prevent exploding updates
        out = torch.clamp(out, -1., 1.)
        return out
    

    
    def get_initial_state(self):
        # make a regular grid of particles as initial state
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles

        x = (2.*torch.zeros((num_nodes, self.config.particle_dim), device=self.device)- 1.)*0.001  # [B*N, state_channels]

        x[:, :self.config.N_spatial_dim] = uniform_circular_distribution(num_nodes, device=self.device)

        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)

        # normalize the polarization block to unit length without in-place slicing (keeps autograd happy)
        start = self.config.N_spatial_dim
        end = start + 2 * self.config.N_polarizations
        # pol = x[:, start:end]
        # pol = F.normalize(pol, p=2, dim=1, eps=1e-8)
        
        # for now make pol [0.,1] 
        pol = torch.zeros_like(x[:, start:end])
        pol[:, 0] = 1.

        # rebuild x to avoid in-place grad issues on a view
        x = torch.cat((x[:, :start], pol, x[:, end:]), dim=1)


        return x, batch
    

    def forward(self, x, batch, steps):
        assert self.message_conv is not None 
        assert self.message_to_output_layer is not None
        # x: [B*N, state_channels]
        # batch: [B*N]

        for _ in range(steps):
            edge_index = radius_graph(
                x[:, : self.config.N_spatial_dim],
                r=self.config.neighbor_radius,
                loop=False,
                batch=batch,
            )

            messages = self.message_conv(x, edge_index, batch=batch)  # [B*N, out_channels]

            output = self.message_to_output(messages)  # [B*N, out_dim]
            x = self.update(output, x)

        return x




class HamiltonianParticle(torch.nn.Module):
    particle_type_name = ParticleType.HAMILTONIAN

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        
        self.message_conv : torch.nn.Module | None = None
        self.own_state_nn : torch.nn.Module | None = None

        self.setup()

    def setup(self):
        self.initialize_architecture()

    def initialize_architecture(self):
        # Message NN
        self.message_conv = HNNConv(out_channels = 1, config = self.config)
        self.message_to_output_layer = torch.nn.Linear(
            self.config.particle_config.message_latent_dim,
            self.config.out_dim,
        )


    def update(self, output, x):

        need_graph = self.training and torch.is_grad_enabled()
        dHdx = torch.autograd.grad(
            output.sum(),
            x,
            create_graph=need_graph,
            retain_graph=need_graph
        )[0]

        # clip the updates to prevent exploding gradients
        dHdx = torch.clamp(dHdx, -100., 100.)
        
        random_noise = torch.randn_like(dHdx) * self.config.noise_level

        newstate = x - (dHdx * 0.01 + random_noise) 

        x = newstate

        x.requires_grad_()  # we need to retain gradients for the updated state to compute the Hamiltonian updates in the next step

        return x
    
    def get_initial_state(self):
        # make a regular grid of particles as initial state
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles

        base_pos = uniform_circular_distribution_batch(self.config.N_particles, batch_size, noise=0.01, config=self.config, device=self.device)

        x = (2. * torch.rand((num_nodes, self.config.N_spatial_dim), device=self.device) - 1.) * 0.001
        x[:, :self.config.N_spatial_dim] = base_pos

        x.requires_grad_()  # we need gradients for the initial positions to compute the Hamiltonian updates
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)
        return x, batch
    

    def forward(self, x, batch, steps, return_history: bool = False):
        assert self.message_conv is not None 
        assert self.message_to_output_layer is not None
        # x: [B*N, state_channels]
        # batch: [B*N]

        history = [] if return_history else None

        for _ in range(steps):
            edge_index = radius_graph(
                x[:, : self.config.N_spatial_dim],
                r=self.config.neighbor_radius,
                loop=False,
                batch=batch,
            )

            output = self.message_conv(x, edge_index, batch=batch)  # [B*N, out_channels]

            x = self.update(output, x)

            if return_history and history is not None:
                history.append(x)

        if return_history:
            return x, history

        return x

class Particle(torch.nn.Module):
    particle_type_name = ParticleType.PARTICLE

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        
        self.message_conv : torch.nn.Module | None = None
        self.own_state_nn : torch.nn.Module | None = None
        self.message_to_output_layer : torch.nn.Module | None = None

        self.setup()

    def setup(self):
        self.initialize_architecture()

    def initialize_architecture(self):
        # Message NN
        self.message_conv = HNNConv(
            out_channels=self.config.N_spatial_dim + self.config.particle_config.hidden_dim,
            config=self.config,
        )
        self.message_to_output_layer = torch.nn.Linear(
            self.config.particle_config.message_latent_dim,
            self.config.N_spatial_dim + self.config.particle_config.hidden_dim,
        )
        if self.config.particle_config.zero_initialization:
            with torch.no_grad():
                self.message_to_output_layer.weight.zero_()
                self.message_to_output_layer.bias.zero_()

    def update(self, output, x):
        
        newstate = x - output * 0.01

        x = newstate

        x.requires_grad_()  # we need to retain gradients for the updated state to compute the Hamiltonian updates in the next step

        return x
    
    def get_initial_state(self):
        # make a regular grid of particles as initial state
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles

        base_pos = uniform_circular_distribution_batch(self.config.N_particles, batch_size, noise=0.01, config=self.config, device=self.device)

        x = (2. * torch.rand((num_nodes, self.config.N_spatial_dim + self.config.particle_config.hidden_dim), device=self.device) - 1.) * 0.001
        x[:, :self.config.N_spatial_dim] = base_pos

        x.requires_grad_()  # we need gradients for the initial positions to compute the Hamiltonian updates
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)
        return x, batch
    

    def forward(self, x, batch, steps, return_history: bool = False):
        assert self.message_conv is not None 
        assert self.message_to_output_layer is not None
        # x: [B*N, state_channels]
        # batch: [B*N]

        history = [] if return_history else None

        for _ in range(steps):
            edge_index = radius_graph(
                x[:, : self.config.N_spatial_dim],
                r=self.config.neighbor_radius,
                loop=False,
                batch=batch,
            )

            output = self.message_conv(x, edge_index, batch=batch)  # [B*N, out_channels]
            # output = self.message_to_output_layer(output)
            x = self.update(output, x)

            if return_history and history is not None:
                history.append(x)

        if return_history:
            return x, history

        return x



class PolarizedHamiltonianParticle(torch.nn.Module):
    particle_type_name = ParticleType.POLARIZED_HAMILTONIAN

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        if self.config.particle_config.hidden_dim != 2:
            raise ValueError(
                "PolarizedHamiltonianParticle currently requires particle_config.hidden_dim == 2 "
                f"(got {self.config.particle_config.hidden_dim})."
            )

        # Keep hidden dynamics bounded and weakly contractive to avoid drift.
        self.hidden_step_size = 0.01
        self.hidden_decay = 0.1
        self.hidden_clip = 5.0

        
        self.message_conv : torch.nn.Module | None = None
        self.own_state_nn : torch.nn.Module | None = None
        self.message_to_output_layer : torch.nn.Module | None = None

        self.setup()

    def setup(self):
        self.initialize_architecture()

    def initialize_architecture(self):
        # Message NN
        self.message_conv = HNNConv(out_channels = 1 + self.config.particle_config.hidden_dim, config = self.config)
        self.message_to_output_layer = torch.nn.Linear(
            self.config.particle_config.message_latent_dim,
            self.config.out_dim,
        )

        if self.config.particle_config.zero_initialization:
            with torch.no_grad():
                self.message_to_output_layer.weight.zero_()
                self.message_to_output_layer.bias.zero_()

    def update(self, output, x):

        need_graph = self.training and torch.is_grad_enabled()


        potentialoutput = output[:, 0]  # [num_nodes]
        hidden_output = output[:, 1:]  # [num_nodes, hidden_dim]

        dHdx = torch.autograd.grad(
            potentialoutput.sum(),
            x,
            create_graph=need_graph,
            retain_graph=need_graph
        )[0]

        # clip the updates to prevent exploding gradients
        dHdx = torch.clamp(dHdx, -100., 100.)
        hidden_output = torch.tanh(hidden_output)

        x_old = x.clone()  # [num_nodes, state_dim]
        x_new = x.clone()  # avoid in-place operations on x which can cause autograd issues

        x_new_potential = x_new[:, :self.config.N_spatial_dim] - dHdx[:, :self.config.N_spatial_dim] * 0.01 + torch.randn_like(dHdx[:, :self.config.N_spatial_dim]) * self.config.noise_level


        

        hidden_start = self.config.N_spatial_dim
        hidden_prev = x_new[:, hidden_start:]
        hidden_delta = hidden_output - self.hidden_decay * hidden_prev
        x_new_hidden = hidden_prev + self.hidden_step_size * hidden_delta
        x_new_hidden = F.normalize(x_new_hidden, p=2, dim=1, eps=1e-8)

        if not torch.isfinite(x_new_potential).all() or not torch.isfinite(x_new_hidden).all():
            raise RuntimeError("NaN before cat in PolarizedHamiltonianParticle.update")

        x_new = torch.cat([x_new_potential, x_new_hidden], dim=1)

        # update 1-p of the nodes randomly
        x = torch.where(torch.rand_like(x) > 0.5, x_new, x_old)

        # x.requires_grad_()


        return x
    
    def get_initial_state(self):
        # make a regular grid of particles as initial state
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles

        base_pos = uniform_circular_distribution_batch(self.config.N_particles, batch_size, noise=0.01, config=self.config, device=self.device)

        # x = (2. * torch.rand((num_nodes, self.config.N_spatial_dim + self.config.particle_config.hidden_dim), device=self.device) - 1.) * 0.001
        x = torch.zeros((num_nodes, self.config.N_spatial_dim + self.config.particle_config.hidden_dim), device=self.device)

        x[:, :self.config.N_spatial_dim] = base_pos
        # radial initialization of the hidden part
        x[:, self.config.N_spatial_dim:] = base_pos.clone()  # start with the hidden part the same as the spatial part
        x[:, self.config.N_spatial_dim:] = F.normalize(
            x[:, self.config.N_spatial_dim:],
            p=2,
            dim=1,
            eps=1e-8,
        )

        # fill the hidden part with unit vectors pointing to the right
        # x = torch.zeros((num_nodes, self.config.N_spatial_dim + self.config.particle_config.hidden_dim), device=self.device)
        # x[:, :self.config.N_spatial_dim] = base_pos
        # x[:, self.config.N_spatial_dim:self.config.N_spatial_dim + self.config.particle_config.hidden_dim] = torch.tensor([1., 0.], device=self.device)

        x.requires_grad_()  # we need gradients for the initial positions to compute the Hamiltonian updates
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)
        return x, batch
    

    def forward(self, x, batch, steps, return_history: bool = False):
        assert self.message_conv is not None 
        assert self.message_to_output_layer is not None
        # x: [B*N, state_channels]
        # batch: [B*N]

        history = [] if return_history else None

        for _ in range(steps):
            edge_index = radius_graph(
                x[:, : self.config.N_spatial_dim],
                r=self.config.neighbor_radius,
                loop=False,
                batch=batch,
            )

            output = self.message_conv(x, edge_index, batch=batch)  # [B*N, out_channels]

            x = self.update(output, x)

            if return_history and history is not None:
                history.append(x)

        if return_history:
            return x, history

        return x


class HEdgeParticle(torch.nn.Module):
    particle_type_name = ParticleType.EDGE_HAMILTONIAN

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        self.message_conv : torch.nn.Module | None = None

        self.setup()

    def setup(self):
        self.initialize_architecture()

    def initialize_architecture(self):
        # Predict scalar Hamiltonian per edge; convert to force per edge, then sum.
        self.message_conv = EHNNConv(out_channels=1, config=self.config)

    def update(self, node_force, x):

        dHdx = node_force

        # clip the updates to prevent exploding gradients
        dHdx = torch.clamp(dHdx, -100., 100.)

        random_noise = torch.randn_like(dHdx) * self.config.noise_level * 0.

        newstate = x - (dHdx * 0.001 + random_noise)

        x = torch.where(torch.rand_like(x) > 0.5, newstate, x)


        # x.requires_grad_()  # we need to retain gradients for the updated state to compute the Hamiltonian updates in the next step

        return x

    def get_initial_state(self):
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles

        base_pos = uniform_circular_distribution_batch(self.config.N_particles, batch_size, noise=0.01, config=self.config, device=self.device)

        x = (2. * torch.rand((num_nodes, self.config.N_spatial_dim), device=self.device) - 1.) * 0.001
        x[:, :self.config.N_spatial_dim] = base_pos

        x.requires_grad_()
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)
        return x, batch

    def forward(self, x, batch, steps, return_history: bool = False):
        assert self.message_conv is not None
        assert isinstance(self.message_conv, EHNNConv)
        # x: [B*N, state_channels]
        # batch: [B*N]

        history = [] if return_history else None

        for _ in range(steps):
            edge_index = radius_graph(
                x[:, : self.config.N_spatial_dim],
                r=self.config.neighbor_radius,
                loop=False,
                batch=batch,
            )

            need_graph = self.training and torch.is_grad_enabled()
            node_force = self.message_conv.edge_forces(
                x,
                edge_index,
                create_graph=need_graph,
            )

            x = self.update(node_force, x)

            if return_history and history is not None:
                history.append(x)

        if return_history:
            return x, history

        return x


class PolarizedHEdgeParticle(torch.nn.Module):
    particle_type_name = ParticleType.POLARIZED_EDGE_HAMILTONIAN

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        if self.config.particle_config.hidden_dim != 2:
            raise ValueError(
                "PolarizedHEdgeParticle currently requires particle_config.hidden_dim == 2 "
                f"(got {self.config.particle_config.hidden_dim})."
            )

        # Keep hidden dynamics bounded and weakly contractive to avoid drift.
        self.hidden_step_size = 0.01
        self.hidden_decay = 0.1
        self.hidden_clip = 5.0

        self.message_conv : torch.nn.Module | None = None
        self.hidden_message_conv : torch.nn.Module | None = None

        self.setup()

    def setup(self):
        self.initialize_architecture()

    def initialize_architecture(self):
        self.message_conv = EHNNConv(out_channels=1, config=self.config)
        self.hidden_message_conv = EHNNConv(
            out_channels=self.config.particle_config.hidden_dim,
            config=self.config,
        )

    def update(self, node_force, hidden_output, x):

        dHdx = node_force

        # clip the updates to prevent exploding gradients
        dHdx = torch.clamp(dHdx, -100., 100.)
        hidden_output = torch.tanh(hidden_output)

        x_old = x.clone()  # [num_nodes, state_dim]
        x_new = x.clone()  # avoid in-place operations on x which can cause autograd issues

        x_new_potential = x_new[:, :self.config.N_spatial_dim] - dHdx[:, :self.config.N_spatial_dim] * 0.01 + torch.randn_like(dHdx[:, :self.config.N_spatial_dim]) * self.config.noise_level

        hidden_start = self.config.N_spatial_dim
        hidden_prev = x_new[:, hidden_start:]
        hidden_delta = hidden_output - self.hidden_decay * hidden_prev
        x_new_hidden = hidden_prev + self.hidden_step_size * hidden_delta
        x_new_hidden = F.normalize(x_new_hidden, p=2, dim=1, eps=1e-8)

        if not torch.isfinite(x_new_potential).all() or not torch.isfinite(x_new_hidden).all():
            raise RuntimeError("NaN before cat in PolarizedHEdgeParticle.update")

        x_new = torch.cat([x_new_potential, x_new_hidden], dim=1)

        # update 1-p of the nodes randomly
        x = torch.where(torch.rand_like(x) > 0.5, x_new, x_old)

        return x

    def get_initial_state(self):
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles

        base_pos = uniform_circular_distribution_batch(self.config.N_particles, batch_size, noise=0.01, config=self.config, device=self.device)

        x = torch.zeros((num_nodes, self.config.N_spatial_dim + self.config.particle_config.hidden_dim), device=self.device)

        x[:, :self.config.N_spatial_dim] = base_pos
        # radial initialization of the hidden part
        x[:, self.config.N_spatial_dim:] = base_pos.clone()
        x[:, self.config.N_spatial_dim:] = F.normalize(
            x[:, self.config.N_spatial_dim:],
            p=2,
            dim=1,
            eps=1e-8,
        )

        x.requires_grad_()
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)
        return x, batch

    def forward(self, x, batch, steps, return_history: bool = False):
        assert self.message_conv is not None
        assert self.hidden_message_conv is not None
        assert isinstance(self.message_conv, EHNNConv)
        # x: [B*N, state_channels]
        # batch: [B*N]

        history = [] if return_history else None

        for _ in range(steps):
            edge_index = radius_graph(
                x[:, : self.config.N_spatial_dim],
                r=self.config.neighbor_radius,
                loop=False,
                batch=batch,
            )

            need_graph = self.training and torch.is_grad_enabled()
            node_force = self.message_conv.edge_forces(
                x,
                edge_index,
                create_graph=need_graph,
            )
            hidden_output = self.hidden_message_conv(x, edge_index, batch=batch)

            x = self.update(node_force, hidden_output, x)

            if return_history and history is not None:
                history.append(x)

        if return_history:
            return x, history

        return x


# class PolarizedHamiltonianParticle(torch.nn.Module):
#     def __init__(self, config : Config):
#         super().__init__()
#         self.config = config
#         self.device = torch.device(config.device)

        
#         self.message_conv : torch.nn.Module | None = None
#         self.own_state_nn : torch.nn.Module | None = None
#         self.message_to_output_layer : torch.nn.Module | None = None

#         self.setup()

#     def setup(self):
#         self.initialize_architecture()

#     def initialize_architecture(self):
#         # Message NN
#         self.message_conv = PolarizedHNNConv(self.config)
#         self.message_to_output_layer = torch.nn.Linear(
#             self.config.particle_config.message_latent_dim,
#             self.config.out_dim,
#         )

#         if self.config.particle_config.zero_initialization:
#             with torch.no_grad():
#                 self.message_to_output_layer.weight.zero_()
#                 self.message_to_output_layer.bias.zero_()

#     def update(self, output, x):
#         update = torch.autograd.grad(
#             outputs=output.sum(),  # sum over all particles to get a scalar Hamiltonian
#             inputs=x,  # only take the spatial part of the state for the gradient
#             create_graph=False,  # we need to create a graph for the gradients to compute second derivatives
#         )[0]  # [num_nodes, state_dim]

#         newx = x - update * 0.01

#         # newpol = x[:, self.config.N_spatial_dim:] 

#         # newstate = torch.cat((newx, newpol), dim=1)

#         x = newx

#         x.requires_grad_()  # we need to retain gradients for the updated state to compute the Hamiltonian updates in the next step

#         return x
    
#     def get_initial_state(self):
#         # make a regular grid of particles as initial state
#         batch_size = self.config.simulation_config.batch_size
#         num_nodes = batch_size * self.config.N_particles
#         x = (2.*torch.rand((num_nodes, self.config.N_spatial_dim*2), device=self.device)- 1.)*0.001  # [B*N, state_channels]


#         x[:, :self.config.N_spatial_dim] = uniform_circular_distribution(num_nodes, device=self.device)
        

#         # initialize the polarization block to be unit vectors pointing to the right
#         x[:, self.config.N_spatial_dim:self.config.N_spatial_dim + 1] = 1. 
#         x[:, self.config.N_spatial_dim+1:self.config.N_spatial_dim+2] = 0.

#         x.requires_grad_()  # we need gradients for the initial positions to compute the Hamiltonian updates
#         batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)
#         return x, batch
    

#     def forward(self, x, batch, steps):
#         assert self.message_conv is not None 
#         assert self.message_to_output_layer is not None
#         # x: [B*N, state_channels]
#         # batch: [B*N]

#         for _ in range(steps):
#             edge_index = radius_graph(
#                 x[:, : self.config.N_spatial_dim],
#                 r=self.config.neighbor_radius,
#                 loop=False,
#                 batch=batch,
#             )

#             output = self.message_conv(x, edge_index, batch=batch)  # [B*N, out_channels]

#             x = self.update(output, x)

#         return x
