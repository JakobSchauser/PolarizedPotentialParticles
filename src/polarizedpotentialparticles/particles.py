from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.custom_conv import CustomNNConv, HNNConv, PolarizedHNNConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import radius_graph



def uniform_circular_distribution(num_particles, device=None):
    radius = 0.3
    # returns [num_particles, 2] uniformly sampled in a disk of given radius
    theta = 2 * torch.pi * torch.rand(num_particles, device=device)
    r = radius * torch.sqrt(torch.rand(num_particles, device=device))
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.stack([x, y], dim=1)


def uniform_circular_distribution_deterministic(num_particles, device=None):
    radius=0.6

    i = torch.arange(num_particles, device=device, dtype=torch.float32)

    # Golden angle
    golden_angle = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))

    theta = i * golden_angle

    # Uniform area density
    r = radius * torch.sqrt((i + 0.5) / num_particles)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    # add a small amount of noise to break perfect symmetry
    x += 0.01 * torch.randn_like(x)
    y += 0.01 * torch.randn_like(y)

    return torch.stack((x, y), dim=1)



class Particle(torch.nn.Module):
    def __init__(self, config : Config):
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
            self.config.particle_config.message_out_channels,
            self.config.particle_config.out_dim,
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
    def __init__(self, config : Config):
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
        self.message_conv = HNNConv(self.config)
        self.message_to_output_layer = torch.nn.Linear(
            self.config.particle_config.message_out_channels,
            self.config.particle_config.out_dim,
        )

        if self.config.particle_config.zero_initialization:
            with torch.no_grad():
                self.message_to_output_layer.weight.zero_()
                self.message_to_output_layer.bias.zero_()

    def update(self, output, x):
        # assert self.x is not None
        # output: [num_nodes, out_dim]
        # [H]


        need_graph = self.training and torch.is_grad_enabled()
        dHdx = torch.autograd.grad(
            output.sum(),
            x,
            create_graph=need_graph,
            retain_graph=need_graph
        )[0]
        # clip the update to prevent exploding updates
        # move_update = torch.clamp(move_update, -0.2, 0.2)

        # update the state by moving in the direction of the negative gradient (gradient descent)
        newstate = x - dHdx * 0.01

        x = newstate

        x.requires_grad_()  # we need to retain gradients for the updated state to compute the Hamiltonian updates in the next step

        return x
    
    def get_initial_state(self):
        # make a regular grid of particles as initial state
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles

        base_pos = uniform_circular_distribution_deterministic(self.config.N_particles, device=self.device)
        pos = base_pos.repeat(batch_size, 1)  # shape [batch_size * N_particles, 2]

        x = (2. * torch.rand((num_nodes, self.config.N_spatial_dim + self.config.particle_config.hidden_dim), device=self.device) - 1.) * 0.001
        x[:, :self.config.N_spatial_dim] = pos

        x.requires_grad_()  # we need gradients for the initial positions to compute the Hamiltonian updates
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)
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

            output = self.message_conv(x, edge_index, batch=batch)  # [B*N, out_channels]

            x = self.update(output, x)

        return x




class PolarizedHamiltonianParticle(torch.nn.Module):
    def __init__(self, config : Config):
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
        self.message_conv = PolarizedHNNConv(self.config)
        self.message_to_output_layer = torch.nn.Linear(
            self.config.particle_config.message_out_channels,
            self.config.particle_config.out_dim,
        )

        if self.config.particle_config.zero_initialization:
            with torch.no_grad():
                self.message_to_output_layer.weight.zero_()
                self.message_to_output_layer.bias.zero_()

    def update(self, output, x):
        # assert self.x is not None
        # output: [num_nodes, out_dim]
        # [H]

        # Hamiltonian update: compute the gradient of the Hamiltonian with respect to the state
        update = torch.autograd.grad(
            outputs=output.sum(),  # sum over all particles to get a scalar Hamiltonian
            inputs=x,  # only take the spatial part of the state for the gradient
            create_graph=False,  # we need to create a graph for the gradients to compute second derivatives
        )[0]  # [num_nodes, state_dim]

        # clip the update to prevent exploding updates
        # move_update = torch.clamp(move_update, -0.2, 0.2)

        # update the state by moving in the direction of the negative gradient (gradient descent)
        newx = x[:, :self.config.N_spatial_dim] - update[:, :self.config.N_spatial_dim] * 0.01


        # newpol = x[:, self.config.N_spatial_dim:] - update[:, self.config.N_spatial_dim:] * 0.01

        # newpol = F.normalize(newpol, p=2, dim=1, eps=1e-8)

        newpol = x[:, self.config.N_spatial_dim:] 

        newstate = torch.cat((newx, newpol), dim=1)

        x = newstate

        x.requires_grad_()  # we need to retain gradients for the updated state to compute the Hamiltonian updates in the next step

        return x
    
    def get_initial_state(self):
        # make a regular grid of particles as initial state
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles
        x = (2.*torch.rand((num_nodes, self.config.N_spatial_dim*2), device=self.device)- 1.)*0.001  # [B*N, state_channels]


        x[:, :self.config.N_spatial_dim] = uniform_circular_distribution(num_nodes, device=self.device)
        

        # initialize the polarization block to be unit vectors pointing to the right
        x[:, self.config.N_spatial_dim:self.config.N_spatial_dim + 1] = 1. 
        x[:, self.config.N_spatial_dim+1:self.config.N_spatial_dim+2] = 0.

        x.requires_grad_()  # we need gradients for the initial positions to compute the Hamiltonian updates
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)
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

            output = self.message_conv(x, edge_index, batch=batch)  # [B*N, out_channels]

            x = self.update(output, x)

        return x
