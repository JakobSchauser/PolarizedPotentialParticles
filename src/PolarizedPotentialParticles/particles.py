import torch
from configs import ParticleConfig
from torch_geometric.nn import conv
from torch_geometric.nn.models import MLP



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
		mlp = MLP(in_channels=self.state_dim, hidden_channels=32, out_channels=self.state_dim * self.config.message_out_channels, num_layers=3)
		self.message_conv = conv.NNConv(in_channels=self.state_dim, out_channels=self.config.message_out_channels, nn=mlp)

		# own state NN
		total_channels = self.state_dim + self.config.message_out_channels
		self.own_state_nn = MLP(in_channels=total_channels, hidden_channels=32, out_channels=self.state_dim, num_layers=3)


	def preprocess_state(self, x, edge_index):
		# x.shape: [num_nodes, state_channels]

		# state: N_spatial_dim + 2 * N_polarizations + hidden_dim

		# dist = 

		
		# rel_ij =  Dist_ij, dot(pi,pj), dot(qi, qj), dot(r_ij, pi), dot(r_ij, qi), hidden_j - hidden_i, hidden_j, hidden_i    # dim = 1 + 2 + 2 + 3*n_hidden_dim

		# return x_transformed

		return x

	def forward(self, x, edge_index):
		assert self.message_conv is not None and self.own_state_nn is not None
		# x: [num_nodes, state_channels]
		# edge_index: [2, num_edges]

		message_x = self.preprocess_state(x, edge_index)

		# Compute messages
		messages = self.message_conv(message_x, edge_index)  # [num_nodes, out_channels]

		# Concatenate own state with messages
		combined = torch.cat([x, messages], dim=-1)  # [num_nodes, state_channels + out_channels]

		# Update own state
		new_state = self.own_state_nn(combined)  # [num_nodes, state_channels]

		return new_state

