from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.custom_conv import CustomNNConv, HNNConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import radius_graph


class Particle(torch.nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        
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

        # Hamiltonian update: compute the gradient of the Hamiltonian with respect to the state
        move_update = torch.autograd.grad(
            outputs=output.sum(),  # sum over all particles to get a scalar Hamiltonian
            inputs=x,
            create_graph=True,  # we need to create a graph for the gradients to compute second derivatives
        )[0]  # [num_nodes, state_dim]

        # clip the update to prevent exploding updates
        # move_update = torch.clamp(move_update, -0.2, 0.2)

        # update the state by moving in the direction of the negative gradient (gradient descent)
        x = x - move_update * 0.1

        x.requires_grad_()  # we need to retain gradients for the updated state to compute the Hamiltonian updates in the next step

        return x
    

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
