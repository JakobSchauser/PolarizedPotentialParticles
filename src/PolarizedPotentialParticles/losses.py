import torch
from torch_geometric.nn import radius_graph
from polarizedpotentialparticles.configs import Config


def is_everyone_equidistant(pos : torch.Tensor, config : Config) -> torch.Tensor:
    # minimize variance of pairwise distances between particles within a certain radius
    edge_index = radius_graph(pos, r=config.neighbor_radius, loop=False)  # [2, num_edges]
    pos_i = pos[edge_index[0]]  # [num_edges, 2
    pos_j = pos[edge_index[1]]  # [num_edges, 2]
    dist_ij = torch.norm(pos_i - pos_j, dim=-1)  # [num_edges]
    loss = torch.var(dist_ij)
    return loss



def compute_loss(output : torch.Tensor, config : Config) -> torch.Tensor:
    poss = output[:, :2]  # Assuming the first 2 channels are positions
    return is_everyone_equidistant(poss, config) 