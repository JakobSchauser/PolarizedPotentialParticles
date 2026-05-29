from __future__ import annotations

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import degree


from typing import Callable, Tuple, TYPE_CHECKING, Union
from torch import Tensor
from torch.nn import Parameter

from polarizedpotentialparticles.utils import atomize_state

from polarizedpotentialparticles.configs import Config

class CustomNNConv(MessagePassing):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config


        state_channels = config.state_dim


        out_channels = config.particle_config.message_latent_dim


        mlp1 = []
        mlp1.append(Linear(config.message_channels, 32))
        mlp1.append(torch.nn.ReLU())
        mlp1.append(Linear(32, state_channels)) # arrnitratry size, but why not 
        
        self.nn = torch.nn.Sequential(*mlp1)

        mlp2 = []
        mlp2.append(Linear(state_channels + 1 + state_channels + config.message_channels, 32))  # +1 for degree, + message channels for the aggregated skip connection
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(32, out_channels))

        self.lin = torch.nn.Sequential(*mlp2)

        self.reset_parameters()

        # # zero the learnable parameters of the final linear layer
        # if self.config.particle_config.zero_initialization:
        #     zeros(self.lin[-1].weight)
        #     zeros(self.lin[-1].bias)

        self.aggr = 'mean'  # or 'mean', 'max', etc. 

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
        # dot_qi_qj = torch.sum(pol_i[1] * pol_j[1], dim=-1, keepdim=True)  # [num_edges, 1]


        dot_rij_pi = torch.sum(r_ij * pol_i[0], dim=-1, keepdim=True)  # [num_edges, 1]
        # dot_rij_qi = torch.sum(r_ij * pol_i[1], dim=-1, keepdim=True)  # [num_edges, 1]


        hidden_diff = hidden_j - hidden_i  # [num_edges, hidden_dim]

        edge_attr = torch.cat([dist_ij, dot_pi_pj, dot_rij_pi, hidden_diff, hidden_j], dim=-1)  # [num_edges, 1 + 2 + 2  + 2* hidden_dim]

        return edge_attr


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        batch: OptTensor | None = None,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise ValueError("I dont understand Pytorch-error!!!")

        deg = degree(edge_index[0], num_nodes=x.size(0), dtype=x.dtype).unsqueeze(-1)  # Maybe Batch??
        return self.propagate(edge_index, x=x, deg=deg, batch=batch) # [num_nodes, out_channels]


    def message(self, x_i : Tensor, x_j: Tensor, ) -> Tensor:
        # x_i, x_j: [num_edges, state_channels]

        edge_attr = self.make_msg(x_i, x_j)
        conv = self.nn(edge_attr)


        return torch.cat([conv, edge_attr], dim=-1)


    def update(self, aggr_out: Tensor, x : Tensor, deg: Tensor) -> Tensor:

        x_no_spatial = x[:, self.config.N_spatial_dim:]  # [num_nodes, state_channels]

        out = torch.cat([x_no_spatial, deg, aggr_out], dim=-1)
        out = self.lin(out)
        return out
    


class HNNConv(MessagePassing):
    def __init__(self, out_channels: int, config: Config):
        super().__init__()

        self.config = config

        hidden_width = 64
        edge_latent_dim = 32

        # When conditioning_mode == "concat", the particle state passed to the conv is
        # x_in = cat([x, cond], dim=-1) so the effective hidden dim is wider by cond_dim.
        eff_hidden = config.particle_config.hidden_dim + config.cond_dim

        mlp1 = []
        mlp1.append(Linear(config.N_spatial_dim + 2 * eff_hidden + 1, hidden_width))
        mlp1.append(torch.nn.ReLU())
        mlp1.append(Linear(hidden_width, edge_latent_dim))
        
        self.nn = torch.nn.Sequential(*mlp1)

        mlp2 = []
        mlp2.append(Linear(edge_latent_dim + 1 + eff_hidden, hidden_width))  # +1 for degree, + own hidden state
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(hidden_width, hidden_width))
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(hidden_width, hidden_width))
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(hidden_width, out_channels))

        self.lin = torch.nn.Sequential(*mlp2)
        self.reset_parameters()

        if self.config.particle_config.zero_initialization:
            final_layer = self.lin[-1]
            if isinstance(final_layer, Linear):
                zeros(final_layer.weight)
                zeros(final_layer.bias)

        self.aggr = 'add'
        self.dist_eps = 1e-6

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

        r_ij = x_i[:, :self.config.N_spatial_dim] - x_j[:, :self.config.N_spatial_dim]  # [num_edges, N_spatial_dim]

        dist_ij = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1, keepdim=True) + self.dist_eps)  # [num_edges, 1]

        dir_ij = r_ij / dist_ij  # normalize to get direction with smooth norm

        dist_ij = torch.exp(-dist_ij*4)  # [num_edges, 1]

        hidden_i = x_i[:, self.config.N_spatial_dim:]  # [num_edges, hidden_dim]
        hidden_j = x_j[:, self.config.N_spatial_dim:]  # [num_edges, hidden_dim]


        edge_attr = torch.cat([dir_ij, dist_ij, hidden_j - hidden_i, hidden_i], dim=-1)  # [num_edges, N_spatial_dim + 1 + hidden_dim]

        return edge_attr


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        batch: OptTensor | None = None,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise ValueError("I dont understand Pytorch-error!!!")

        deg = degree(edge_index[0], num_nodes=x.size(0), dtype=x.dtype).unsqueeze(-1)  # Maybe Batch??
        return self.propagate(edge_index, x=x, deg=deg, batch=batch) # [num_nodes, out_channels]


    def message(self, x_i : Tensor, x_j: Tensor, ) -> Tensor:
        # x_i, x_j: [num_edges, state_channels]

        edge_attr = self.make_msg(x_i, x_j)

        conv = self.nn(edge_attr)


        return conv


    def update(self, aggr_out: Tensor, x : Tensor, deg: Tensor) -> Tensor:

        hidden_i = x[:, self.config.N_spatial_dim:]  # [num_nodes, hidden_dim]

        out = torch.cat([deg, aggr_out, hidden_i], dim=-1)
        out = self.lin(out) 
        return out
    

class DistanceHNNConv(MessagePassing):
    def __init__(self, out_channels: int, config: Config):
        super().__init__()

        self.config = config

        hidden_width = 64
        edge_latent_dim = 32

        eff_hidden = config.particle_config.hidden_dim + config.cond_dim

        mlp1 = []
        mlp1.append(Linear(config.N_spatial_dim + eff_hidden + 1, hidden_width))
        mlp1.append(torch.nn.ReLU())
        mlp1.append(Linear(hidden_width, edge_latent_dim))
        
        self.nn = torch.nn.Sequential(*mlp1)

        mlp2 = []
        mlp2.append(Linear(edge_latent_dim + 1 + eff_hidden, hidden_width))  # +1 for degree, + own hidden state
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(hidden_width, hidden_width))
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(hidden_width, hidden_width))
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(hidden_width, out_channels))

        self.lin = torch.nn.Sequential(*mlp2)
        self.reset_parameters()

        if self.config.particle_config.zero_initialization:
            final_layer = self.lin[-1]
            if isinstance(final_layer, Linear):
                zeros(final_layer.weight)
                zeros(final_layer.bias)

        self.aggr = 'add'
        self.dist_eps = 1e-6

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

        r_ij = x_i[:, :self.config.N_spatial_dim] - x_j[:, :self.config.N_spatial_dim]  # [num_edges, N_spatial_dim]

        dist_ij = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1, keepdim=True) + self.dist_eps)  # [num_edges, 1]

        weighted_dist = torch.exp(-dist_ij*4)  # [num_edges, 1]

        dir_ij = r_ij / dist_ij  # normalize to get direction with smooth norm

        hidden_i = x_i[:, self.config.N_spatial_dim:]  # [num_edges, hidden_dim]
        hidden_j = x_j[:, self.config.N_spatial_dim:]  # [num_edges, hidden_dim]


        edge_attr = torch.cat([dir_ij / weighted_dist, (hidden_j - hidden_i) / weighted_dist, weighted_dist], dim=-1)  # [num_edges, N_spatial_dim + 1 + hidden_dim]

        return edge_attr


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        batch: OptTensor | None = None,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise ValueError("I dont understand Pytorch-error!!!")

        deg = degree(edge_index[0], num_nodes=x.size(0), dtype=x.dtype).unsqueeze(-1)  # Maybe Batch??
        return self.propagate(edge_index, x=x, deg=deg, batch=batch) # [num_nodes, out_channels]


    def message(self, x_i : Tensor, x_j: Tensor, ) -> Tensor:
        # x_i, x_j: [num_edges, state_channels]

        edge_attr = self.make_msg(x_i, x_j)

        conv = self.nn(edge_attr)


        return conv


    def update(self, aggr_out: Tensor, x : Tensor, deg: Tensor) -> Tensor:

        hidden_i = x[:, self.config.N_spatial_dim:]  # [num_nodes, hidden_dim]

        out = torch.cat([deg, aggr_out, hidden_i], dim=-1)
        out = self.lin(out) 
        return out
    

    




class EHNNConv(MessagePassing):
    """Edge-sum Hamiltonian conv: H_i = sum_j NN(edge_ij). No post-aggregation NN."""
    def __init__(self, out_channels: int, config: Config):
        super().__init__()

        self.config = config

        in_channels = config.N_spatial_dim + 2 * config.particle_config.hidden_dim + 1
        self.heads = torch.nn.ModuleList()
        for _ in range(out_channels):
            head = torch.nn.Sequential(
                Linear(in_channels, 32),
                torch.nn.ReLU(),
                Linear(32, 1),
            )
            self.heads.append(head)

        # Final safety net: if higher-order grads become non-finite, zero/saturate them
        # before optimizer step so parameters do not get poisoned.
        def _sanitize_grad(grad: Tensor) -> Tensor:
            return torch.nan_to_num(grad, nan=0.0, posinf=1e3, neginf=-1e3)

        for param in self.heads.parameters():
            param.register_hook(_sanitize_grad)

        self.reset_parameters()

        if self.config.particle_config.zero_initialization:
            for head in self.heads:
                final_layer = list(head.children())[-1]
                if isinstance(final_layer, Linear):
                    zeros(final_layer.weight)
                    zeros(final_layer.bias)

        self.aggr = 'add'
        self.dist_eps = 1e-6

    def reset_parameters(self):
        super().reset_parameters()
        for head in self.heads:
            reset(head)

    def make_msg(self, x_i, x_j):
        r_ij = x_i[:, :self.config.N_spatial_dim] - x_j[:, :self.config.N_spatial_dim]  # [num_edges, N_spatial_dim]

        # Use a smooth norm to avoid undefined gradients at r_ij == 0.
        dist_ij = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1, keepdim=True) + self.dist_eps)  # [num_edges, 1]

        dir_ij = r_ij / dist_ij  # normalize to get direction

        dist_ij = torch.exp(-dist_ij)  # [num_edges, 1]

        hidden_i = x_i[:, self.config.N_spatial_dim:]  # [num_edges, hidden_dim]
        hidden_j = x_j[:, self.config.N_spatial_dim:]  # [num_edges, hidden_dim]

        edge_attr = torch.cat([dir_ij, dist_ij, hidden_j - hidden_i, hidden_i], dim=-1)  # [num_edges, N_spatial_dim + 2*hidden_dim + 1]

        return edge_attr

    def edge_forces(self, x: Tensor, edge_index: Tensor, create_graph: bool) -> Tensor:
        """Compute per-node forces by differentiating per-edge energy and summing.

        For each directed edge (j -> i), we predict scalar edge energy E_ij,
        compute grad wrt r_ij = x_i - x_j, then accumulate action-reaction:
        F_i += -dE_ij/dr_ij and F_j += +dE_ij/dr_ij.
        """
        if len(self.heads) != 1:
            raise ValueError("edge_forces requires EHNNConv configured with out_channels=1")

        if edge_index.numel() == 0:
            return torch.zeros(
                (x.size(0), self.config.N_spatial_dim),
                dtype=x.dtype,
                device=x.device,
            )

        src, dst = edge_index[0], edge_index[1]
        x_i = x[dst]
        x_j = x[src]

        # Relative position per directed edge; this is what we differentiate through.
        r_ij = x_i[:, :self.config.N_spatial_dim] - x_j[:, :self.config.N_spatial_dim]
        dist_ij = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1, keepdim=True) + self.dist_eps)
        dir_ij = r_ij / dist_ij
        dist_ij = torch.exp(-dist_ij)

        hidden_i = x_i[:, self.config.N_spatial_dim:]
        hidden_j = x_j[:, self.config.N_spatial_dim:]
        edge_attr = torch.cat([dir_ij, dist_ij, hidden_j - hidden_i, hidden_i], dim=-1)

        edge_energy = self.heads[0](edge_attr).squeeze(-1)
        dEdr = torch.autograd.grad(
            edge_energy.sum(),
            r_ij,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]

        force_i = -dEdr
        force_j = dEdr

        num_nodes = x.size(0)
        forces = torch.zeros(
            (num_nodes, self.config.N_spatial_dim),
            dtype=x.dtype,
            device=x.device,
        )
        forces.index_add_(0, dst, force_i)
        forces.index_add_(0, src, force_j)

        return forces

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        batch: OptTensor | None = None,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise ValueError("I dont understand Pytorch-error!!!")

        return self.propagate(edge_index, x=x)  # [num_nodes, out_channels]

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        # x_i, x_j: [num_edges, state_channels]
        edge_attr = self.make_msg(x_i, x_j)
        per_head = [head(edge_attr) for head in self.heads]
        return torch.cat(per_head, dim=-1)

    def update(self, aggr_out: Tensor) -> Tensor:
        # H_i is simply the sum of edge messages — no post-aggregation NN
        return aggr_out




class PolarizedHNNConv(MessagePassing):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        out_channels = 2


        arbitrary_size = 8

        mlp1 = []
        mlp1.append(Linear(config.N_spatial_dim + 1, 32))
        mlp1.append(torch.nn.ReLU())
        mlp1.append(Linear(32, arbitrary_size)) # arrnitratry size, but why not 
        
        self.nn = torch.nn.Sequential(*mlp1)

        mlp2 = []
        mlp2.append(Linear(arbitrary_size + 1, 32))  # +1 for degree, #RuntimeError: mat1 and mat2 shapes cannot be multiplied (80x12 and 9x32)
        mlp2.append(torch.nn.ReLU())
        mlp2.append(Linear(32, out_channels))

        self.lin = torch.nn.Sequential(*mlp2)

        self.reset_parameters()


        self.aggr = 'mean'  # or 'mean', 'max', etc. 

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

        x_i_pos = x_i[:, :self.config.N_spatial_dim]
        x_j_pos = x_j[:, :self.config.N_spatial_dim]

        x_i_pol = x_i[:, self.config.N_spatial_dim:self.config.N_spatial_dim*2]
        x_j_pol = x_j[:, self.config.N_spatial_dim:self.config.N_spatial_dim*2]

        r_ij = x_i_pos - x_j_pos  # [num_edges, N_spatial_dim]

        dist_ij = torch.norm(r_ij, dim=-1, keepdim=True)  # [num_edges, 1]

        dir_ij = r_ij / (dist_ij + 1e-4)  # normalize to get direction, add small epsilon to prevent division by zero

        dist_ij = torch.exp(-dist_ij)  # [num_edges, 1]

        dot_rij_pi = torch.sum(r_ij * x_i_pol, dim=-1, keepdim=True)  # [num_edges, 1]
        x_i_pol_perp = torch.cat([-x_i_pol[:, 1:], x_i_pol[:, :1]], dim=-1)  # Rotate polarization by 90 degrees to get perpendicular direction
        dot_rij_pi_perp = torch.sum(r_ij * x_i_pol_perp, dim=-1, keepdim=True)  # [num_edges, 1]

        dot_rij = torch.cat([dot_rij_pi, dot_rij_pi_perp], dim=-1)  # [num_edges, 2]


        edge_attr = torch.cat([dot_rij, dist_ij], dim=-1)  # [num_edges, N_spatial_dim + 1]

        return edge_attr


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        batch: OptTensor | None = None,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise ValueError("I dont understand Pytorch-error!!!")

        deg = degree(edge_index[0], num_nodes=x.size(0), dtype=x.dtype).unsqueeze(-1)  # Maybe Batch??
        return self.propagate(edge_index, x=x, deg=deg, batch=batch) # [num_nodes, out_channels]


    def message(self, x_i : Tensor, x_j: Tensor, ) -> Tensor:
        # x_i, x_j: [num_edges, state_channels]

        edge_attr = self.make_msg(x_i, x_j)
        conv = self.nn(edge_attr)


        return conv


    def update(self, aggr_out: Tensor, x : Tensor, deg: Tensor) -> Tensor:

        out = torch.cat([deg, aggr_out], dim=-1)
        out = self.lin(out) 
        return out