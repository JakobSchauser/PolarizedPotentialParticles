import torch
from configs import ParticleConfig
from torch_geometric.nn import conv
from torch_geometric.nn.models import MLP
from custom_conv import CustomNNConv


class Particle(torch.nn.Module):
	def __init__(self, config : ParticleConfig):
		super().__init__()
		self.config = config

		self.state_dim = config.N_spatial_dim + 2 * config.N_polarizations + config.hidden_dim
		
		
		self.x : torch.Tensor | None = None
		self.message_conv : torch.nn.Module | None = None
		self.own_state_nn : torch.nn.Module | None = None

		self.setup()


	def setup(self):
		self.initialize_state()
		self.initialize_architecture()


	def initialize_state(self):
		self.x = torch.zeros((self.config.N_particles, self.state_dim))  # [num_nodes, state_channels]


	def initialize_architecture(self):
		# Message NN
		self.message_conv = CustomNNConv(self.config)

		# own state NN
		total_channels = self.state_dim + self.config.message_out_channels
		self.own_state_nn = MLP(in_channels=total_channels, hidden_channels=32, out_channels=self.state_dim, num_layers=3)


	def preprocess_state(self, x, edge_index):
		# x.shape: [num_nodes, state_channels]

		# state: N_spatial_dim + 2 * N_polarizations + hidden_dim

		# dist = 


		# return x_transformed

		return x

	def forward(self, x, edge_index):
		assert self.message_conv is not None and self.own_state_nn is not None
		# x: [num_nodes, state_channels]
		# edge_index: [2, num_edges]


		# Compute messages
		messages = self.message_conv(x, edge_index)  # [num_nodes, out_channels]

		# Concatenate own state with messages
		combined = torch.cat([x, messages], dim=-1)  # [num_nodes, state_channels + out_channels]

		# Update own state
		new_state = self.own_state_nn(combined)  # [num_nodes, state_channels]

		return new_state

