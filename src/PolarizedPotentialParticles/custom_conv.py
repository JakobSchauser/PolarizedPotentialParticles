import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from typing import Callable, Tuple, Union
from torch import Tensor
from torch.nn import Parameter

from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.utils import atomize_state

class CustomNNConv(MessagePassing):
    def __init__(self, config : Config):
        super().__init__()

        self.config = config


        state_channels = config.state_dim


        out_channels = config.particle_config.message_out_channels


        mlp1 = []
        mlp1.append(Linear(config.message_channels, 32))
        mlp1.append(torch.nn.ReLU())
        mlp1.append(Linear(32, state_channels)) # arrnitratry size, but why not 
        
        self.nn = torch.nn.Sequential(*mlp1)

        mlp2 = []
        mlp2.append(Linear(state_channels + state_channels, 32))
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(32, out_channels))

        self.lin = torch.nn.Sequential(*mlp2)

        self.reset_parameters()

        # zero the learnable parameters of the final linear layer
        zeros(self.lin[-1].weight)
        zeros(self.lin[-1].bias)

        self.aggr = 'add'  # or 'mean', 'max', etc. 

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        reset(self.lin)


    def make_msg(self, x_i, x_j):
		# rel_ij =  Dist_ij, 

        #           dot(pi,pj), 
        #           dot(qi, qj), 

        #           dot(r_ij, pi), 
        #           dot(r_ij, qi), 

        #           hidden_j - hidden_i, 
        #           hidden_j, 
        #
        #           # dim = 1 + 2 + 2 + 2*n_hidden_dim

        pos_i, pol_i, hidden_i = atomize_state(x_i, self.config)
        pos_j, pol_j, hidden_j = atomize_state(x_j, self.config)


        r_ij = pos_j - pos_i  # [num_edges, N_spatial_dim]
        dist_ij = torch.norm(r_ij, dim=-1, keepdim=True)  # [num_edges, 1]

        dot_pi_pj = torch.sum(pol_i[0] * pol_j[0], dim=-1, keepdim=True)  # [num_edges, 1]
        dot_qi_qj = torch.sum(pol_i[1] * pol_j[1], dim=-1, keepdim=True)  # [num_edges, 1]


        dot_rij_pi = torch.sum(r_ij * pol_i[0], dim=-1, keepdim=True)  # [num_edges, 1]
        dot_rij_qi = torch.sum(r_ij * pol_i[1], dim=-1, keepdim=True)  # [num_edges, 1]


        hidden_diff = hidden_j - hidden_i  # [num_edges, hidden_dim]

        edge_attr = torch.cat([dist_ij, dot_pi_pj, dot_qi_qj, dot_rij_pi, dot_rij_qi, hidden_diff, hidden_j], dim=-1)  # [num_edges, 1 + 2 + 2  + 2* hidden_dim]

        return edge_attr


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
    ) -> Tensor:

        if not isinstance(x, Tensor):
            raise ValueError("I dont understand Pytorch-error!!!")
        
        # x.shape = [num_nodes, state_channels]

        # propagate calls message, aggr and update in order.
        return self.propagate(edge_index, x=x) # [num_nodes, out_channels]


    def message(self, x_i : Tensor, x_j: Tensor) -> Tensor:
        # x_i, x_j: [num_edges, state_channels]

        edge_attr = self.make_msg(x_i, x_j)
        conv = self.nn(edge_attr)

        return conv
        # weight = weight.view(-1, self.in_channels_l, self.out_channels)
        # return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)


    def update(self, aggr_out: Tensor, x : Tensor) -> Tensor:

        x_no_spatial = x[:, self.config.N_spatial_dim:]  # [num_nodes, state_channels]

        out = torch.cat([x_no_spatial, aggr_out], dim=-1)
        out = self.lin(out)
        return out