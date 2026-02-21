import numpy as np
import torch
from PIL import Image

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
    # try too keep all neighbohrs 0.65 units apart
    pos = output[:, :config.N_spatial_dim]  # [B*N, N_spatial_dim]
    edge_index = radius_graph(pos, r=config.neighbor_radius, loop=False)  #
    if edge_index.numel() == 0:
        return pos.new_zeros(())
    pos_i = pos[edge_index[0]]  # [num_edges, 2]
    pos_j = pos[edge_index[1]]  # [num_edges, 2
    dist_ij = torch.norm(pos_i - pos_j, dim=-1)  # [num_edges]
    loss = torch.sum((dist_ij - 0.12) ** 2) / (edge_index.shape[1])*1.1  # mean squared error from 0.12 distance 
    return loss

def compute_loss(output : torch.Tensor, config : Config, batch : torch.Tensor) -> torch.Tensor:
    losses = []
    for b in torch.unique(batch):
        mask = batch == b
        pos = output[mask][:, :config.N_spatial_dim]
        # losses.append(is_everyone_equidistant(pos, config))
        # losses.append(relaxation_distance_loss(output[mask], config))
        losses.append(image_loss(output[mask], config))

    return torch.stack(losses).mean()

def gaussian_splat(pos, grid_size=64, sigma=0.05, normalize=True):
    # pos: [P, 2] in [-1, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, grid_size, device=pos.device, dtype=pos.dtype),
        torch.linspace(-1, 1, grid_size, device=pos.device, dtype=pos.dtype),
        indexing="ij",
    )  # [H, W]
    px = pos[:, 0].view(-1, 1, 1)  # [P, 1, 1]
    py = pos[:, 1].view(-1, 1, 1)
    d2 = (xx - px) ** 2 + (yy - py) ** 2  # [P, H, W]
    grid = torch.exp(-d2 / (2 * sigma ** 2)).sum(dim=0)  # [H, W]

    if normalize:
        # normalize the grid to [0, 1]
        grid = grid / (grid.max() + 1e-8)
    return grid


def gaussian_splat_data(pos,):
    return gaussian_splat(pos, grid_size=64, sigma=0.05, normalize=False)


def gaussian_splat_from_image(img_path):
    grid_size = 64

    img = Image.open(img_path).convert("RGBA").resize((grid_size, grid_size))
    img = torch.from_numpy(np.array(img)).float() / 255.0  # [grid_size, grid_size, 4]
    # make mask of alpha channel to extract only the shape of the emoji, ignoring the transparent background
    alpha_mask = img[:, :, 3] > 0.5

    # convert into list of (x,y) coordinates of the pixels that are part of the emoji shape
    img_pos = torch.nonzero(alpha_mask, as_tuple=False).float()

    img_pos = (img_pos / grid_size) * 2 - 1  # normalize to [-1, 1]


    # gaussian splatting of the image
    img_grid = gaussian_splat(img_pos, grid_size=grid_size, sigma=0.05, normalize=False) / 16.


    return img_grid

def image_loss(output : torch.Tensor, config : Config) -> torch.Tensor:
    # try to make the particles form an arbitrary shape
    grid_size = 64

    emoji_path = "C:/Users/jakob/Documents/work/PolarizedPotentialParticles/src/polarizedpotentialparticles/morphologies/" + config.loss_config.target + ".png"

    img_grid = gaussian_splat_from_image(emoji_path) 
    
    # gaussian splatting of the particle positions
    pos = output[:, :config.N_spatial_dim]  # [N_particles, N_spatial_dim]

    # make the positions be in the same coordinate system as the image ([-1, 1])
    pos /= 1.0 

    particle_grid = gaussian_splat_data(pos, )

    # print(f"Image grid max value: {img_grid.max().item():.4f}")
    # print(f"Particle grid max value: {particle_grid.max().item():.4f}")
    # compute mean squared error between the two grids
    loss = torch.mean((img_grid - particle_grid) ** 2)

    return loss