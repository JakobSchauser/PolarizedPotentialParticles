import torch
from torch_geometric.nn import radius_graph
from polarizedpotentialparticles.configs import Config


def is_everyone_equidistant(pos : torch.Tensor, config : Config) -> torch.Tensor:
    # minimize variance of pairwise distances between particles within a certain radius
    edge_index = radius_graph(pos, r=config.neighbor_radius, loop=False)  # [2, num_edges]
    if edge_index.numel() == 0:
        return pos.new_zeros(())

    pos_i = pos[edge_index[0]]  # [num_edges, 2
    pos_j = pos[edge_index[1]]  # [num_edges, 2]
    dist_ij = torch.norm(pos_i - pos_j, dim=-1)  # [num_edges]
    loss = torch.var(dist_ij)
    return loss

def relaxation_distance_loss(output : torch.Tensor, config : Config) -> torch.Tensor:
    # try too keep all neighbohrs 0.5 units apart
    pos = output[:, :config.N_spatial_dim]  # [B*N, N_spatial_dim]
    edge_index = radius_graph(pos, r=config.neighbor_radius, loop=False)  #
    if edge_index.numel() == 0:
        return pos.new_zeros(())
    pos_i = pos[edge_index[0]]  # [num_edges, 2]
    pos_j = pos[edge_index[1]]  # [num_edges, 2
    dist_ij = torch.norm(pos_i - pos_j, dim=-1)  # [num_edges]
    loss = torch.mean((dist_ij - 0.5) ** 2) 
    return loss

def compute_loss(output : torch.Tensor, config : Config, batch : torch.Tensor) -> torch.Tensor:
    losses = []
    for b in torch.unique(batch):
        mask = batch == b
        pos = output[mask][:, :config.N_spatial_dim]
        # losses.append(is_everyone_equidistant(pos, config))
        losses.append(relaxation_distance_loss(output[mask], config))

    return torch.stack(losses).mean()