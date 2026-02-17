from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.custom_conv import CustomNNConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import conv
from torch_geometric.nn.models import MLP


class Particle(torch.nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        
        self.message_conv : torch.nn.Module | None = None
        self.own_state_nn : torch.nn.Module | None = None

        self.setup()



    def setup(self):
        self.initialize_architecture()

    def initialize_architecture(self):
        # Message NN
        self.message_conv = CustomNNConv(self.config)

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

        x = x + output * self.config.simulation_config.dt  # Euler update

        # normalize the polarization block to unit length without in-place slicing (keeps autograd happy)
        start = self.config.N_spatial_dim
        end = start + 2 * self.config.N_polarizations
        pol = x[:, start:end]
        pol = F.normalize(pol, p=2, dim=1, eps=1e-8)

        # rebuild x to avoid in-place grad issues on a view
        x = torch.cat((x[:, :start], pol, x[:, end:]), dim=1)

        return x
    
    def message_to_output(self, message):
        # message: [num_nodes, message_out_channels]
        # output: [num_nodes, out_dim]

        out = torch.nn.Linear(self.config.particle_config.message_out_channels, self.config.particle_config.out_dim)(message)  # [num_nodes, out_dim]

        return out 

    def forward(self, x, edge_index, steps):
        assert self.message_conv is not None 
        # x: [num_nodes, state_channels]
        # edge_index: [2, num_edges]


        for s in range(steps):
            # Compute messages
            messages = self.message_conv(x, edge_index)  # [num_nodes, out_channels]

            # Update own state
            output = self.message_to_output(messages)  # [num_nodes, out_dim]
            x = self.update(output, x)


        return x



# class ParticleSystem(torch.nn.Module):
# 	def __init__(self, config : ParticleConfig):
# 		super().__init__()
# 		self.config = config

# 		self.particles = Particle(config)

# 	def forward(self, x, edge_index):
# 		return self.particles(x, edge_index)