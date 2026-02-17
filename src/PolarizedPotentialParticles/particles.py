from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.custom_conv import CustomNNConv
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
        # We need to parse the output into position updates, polarization updates, and hidden state updates
        # # [dx, dy, dpol_x, dpol_y, d_hidden1, d_hidden2, ...]
        # dx = output[:, 0]  # [num_nodes]
        # dy = output[:, 1]  # [num_nodes]
        # dpol_x = output[:, 2]  # [num_nodes]
        # dpol_y = output[:, 3]  # [num_nodes]
        # d_hidden = output[:, 4:]  # [num_nodes, hidden_dim]

        # dt = self.config.simulation_config.dt
        # # Update positions and polarizations
        # x[:, :self.config.N_spatial_dim] += torch.stack((dx, dy), dim=1) * dt
        # x[:, self.config.N_spatial_dim:self.config.N_spatial_dim + 2 * self.config.N_polarizations] += torch.stack((dpol_x, dpol_y), dim=1) * dt
        # x[:, self.config.N_spatial_dim + 2 * self.config.N_polarizations:] += d_hidden * dt
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



# class ParticleSystem(torch.nn.Module):
# 	def __init__(self, config : ParticleConfig):
# 		super().__init__()
# 		self.config = config

# 		self.particles = Particle(config)

# 	def forward(self, x, edge_index):
# 		return self.particles(x, edge_index)