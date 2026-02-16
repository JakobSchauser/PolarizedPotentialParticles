import torch


def is_everyone_equidistant(pos : torch.Tensor) -> torch.Tensor:
    # minimize variance of pairwise distances between particles within a certain radius
    dist_matrix = torch.cdist(pos, pos)  # [num_particles, num_particles]
    num_particles = pos.shape[0]

    nbs = dist_matrix < 0.1  # [num_particles, num_particles]

    pairwise_distances = dist_matrix[nbs]  # [num_pairs]
    variance = torch.var(pairwise_distances)

    return variance



def compute_loss(output : torch.Tensor) -> torch.Tensor:
    poss = output[:, :2]  # Assuming the first 2 channels are positions
    return is_everyone_equidistant(poss) 