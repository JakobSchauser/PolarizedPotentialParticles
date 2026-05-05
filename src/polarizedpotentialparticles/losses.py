import numpy as np
import torch
from dataclasses import replace
from pathlib import Path
from PIL import Image

from torch_geometric.nn import radius_graph
from polarizedpotentialparticles.configs import Config


_IMG_GRID_CACHE: dict[tuple[str, str, str], torch.Tensor] = {}


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
    losses = compute_losses(output, config, batch)
    return torch.stack(losses).mean()

def _image_loss_for_target(output: torch.Tensor, config: Config, target: str) -> torch.Tensor:
    patched = replace(config, loss_config=replace(config.loss_config, target=target))
    return image_loss(output, patched)


def compute_losses(output : torch.Tensor, config : Config, batch : torch.Tensor) -> list[torch.Tensor]:
    multiple = config.loss_config.multiple
    losses = []
    for b in torch.unique(batch):
        mask = batch == b
        if multiple is not None:
            target = multiple[int(b.item()) % len(multiple)]
            losses.append(_image_loss_for_target(output[mask], config, target))
        else:
            losses.append(image_loss(output[mask], config))
    return losses


def gaussian_splat(pos, sigma, grid_size=64, normalize=True):
    
    # pos: [P, 2] in [-1, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, grid_size, device=pos.device, dtype=pos.dtype),
        torch.linspace(-1, 1, grid_size, device=pos.device, dtype=pos.dtype),
        indexing="ij",
    )  # [H, W]
    px = pos[:, 0].view(-1, 1, 1)  # [P, 1, 1]
    py = pos[:, 1].view(-1, 1, 1)
    d2 = (xx - px) ** 2 + (yy - py) ** 2  # [P, H, W]

    def gaussian(d2, sigma):
        return torch.exp(-d2 / (2 * sigma ** 2))

    # def gaussian_normalized(d2, sigma):
    #     return torch.exp(-d2 / (2 * sigma ** 2))/ (2 * np.pi * sigma ** 2)

    grid = gaussian(d2, sigma).sum(dim=0)  # [H, W] 

    if normalize:
        # normalize the grid to [0, 1]
        grid = grid / (grid.max() + 1e-8)
    return grid



def gaussian_splat_3d(pos, sigma, grid_size=32, normalize=True):
    assert pos.shape[1] == 3, "gaussian_splat_3d requires 3D positions (pos.shape[1] == 3)"
    # pos: [P, 3] in [-1, 1]
    lin = torch.linspace(-1, 1, grid_size, device=pos.device, dtype=pos.dtype)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")  # [G, G, G]
    px = pos[:, 0].view(-1, 1, 1, 1)  # [P, 1, 1, 1]
    py = pos[:, 1].view(-1, 1, 1, 1)
    pz = pos[:, 2].view(-1, 1, 1, 1)
    d2 = (xx - px) ** 2 + (yy - py) ** 2 + (zz - pz) ** 2  # [P, G, G, G]
    grid = torch.exp(-d2 / (2 * sigma ** 2)).sum(dim=0)  # [G, G, G]
    if normalize:
        grid = grid / (grid.max() + 1e-8)
    return grid


def sphere_shell_target_3d(grid_size=32, sigma=0.1, target_radius=0.7, device=None):
    # Fibonacci sphere: ~500 uniformly distributed points on a sphere shell
    N = 500
    i = torch.arange(N, dtype=torch.float32, device=device)
    golden_angle = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))
    theta = i * golden_angle
    phi = torch.acos(1 - 2 * (i + 0.5) / N)
    x = target_radius * torch.sin(phi) * torch.cos(theta)
    y = target_radius * torch.sin(phi) * torch.sin(theta)
    z = target_radius * torch.cos(phi)
    pts = torch.stack([x, y, z], dim=1)  # [N, 3]
    return gaussian_splat_3d(pts, sigma=sigma, grid_size=grid_size, normalize=True)


def plane_target_3d(grid_size=32, sigma=0.1, target_radius=0.7, device=None):
    # Uniformly sampled disc in the XY plane (z=0)
    N = 500
    r = torch.sqrt(torch.rand(N, device=device)) * target_radius
    theta = torch.rand(N, device=device) * 2 * torch.pi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.zeros(N, device=device)
    pts = torch.stack([x, y, z], dim=1)  # [N, 3]
    return gaussian_splat_3d(pts, sigma=sigma, grid_size=grid_size, normalize=True)


def tube_target_3d(grid_size=32, sigma=0.1, target_radius=0.7, device=None):
    # Open cylinder shell: radius = target_radius, z in [-0.8, 0.8]
    N = 500
    theta = torch.rand(N, device=device) * 2 * torch.pi
    x = target_radius * torch.cos(theta)
    y = target_radius * torch.sin(theta)
    z = torch.rand(N, device=device) * 1.6 - 0.8
    pts = torch.stack([x, y, z], dim=1)  # [N, 3]
    return gaussian_splat_3d(pts, sigma=sigma, grid_size=grid_size, normalize=True)


def gaussian_splat_data(pos, config : Config):
    gs =  gaussian_splat(pos, sigma=config.sigma, grid_size=64, normalize=True) ##################
    return gs

def gaussian_splat_from_image(img_path, device=None):
    grid_size = 64

    img = Image.open(img_path).convert("RGBA").resize((grid_size, grid_size))
    img = torch.from_numpy(np.array(img)).float() / 255.0  # [grid_size, grid_size, 4]
    # make mask of alpha channel to extract only the shape of the emoji, ignoring the transparent background
    alpha_mask = img[:, :, 3] > 0.5

    # convert into list of (x,y) coordinates of the pixels that are part of the emoji shape
    img_pos = torch.nonzero(alpha_mask, as_tuple=False).float()

    img_pos = (img_pos / grid_size) * 2 - 1  # normalize to [-1, 1]

    if device is None:
        device = img_pos.device
    img_pos = img_pos.to(device)


    img_grid = gaussian_splat(img_pos, sigma = 0.1, grid_size=grid_size, normalize=True)

    return img_grid


def get_cached_target_grid(target: str, device: torch.device, dtype: torch.dtype, sphere_target_radius: float = 0.7, sphere_target_sigma: float = 0.1) -> torch.Tensor:
    key = (target, str(device), str(dtype), sphere_target_radius, sphere_target_sigma)
    cached = _IMG_GRID_CACHE.get(key)
    if cached is None:
        if target == "sphere":
            cached = sphere_shell_target_3d(device=device, target_radius=sphere_target_radius, sigma=sphere_target_sigma).to(dtype=dtype)
        elif target == "plane":
            cached = plane_target_3d(device=device, target_radius=sphere_target_radius, sigma=sphere_target_sigma).to(dtype=dtype)
        elif target == "tube":
            cached = tube_target_3d(device=device, target_radius=sphere_target_radius, sigma=sphere_target_sigma).to(dtype=dtype)
        else:
            emoji_path = Path(__file__).resolve().parent / "morphologies" / f"{target}.png"
            cached = gaussian_splat_from_image(emoji_path, device=device).to(dtype=dtype)
        _IMG_GRID_CACHE[key] = cached
    return cached

def image_loss(output : torch.Tensor, config : Config) -> torch.Tensor:
    # try to make the particles form an arbitrary shape
    img_grid = get_cached_target_grid(config.loss_config.target, output.device, output.dtype, config.loss_config.sphere_target_radius, config.loss_config.sphere_target_sigma)
    
    # gaussian splatting of the particle positions
    pos = output[:, :config.N_spatial_dim]  # [N_particles, N_spatial_dim]

    # make the positions be in the same coordinate system as the image ([-1, 1])
    pos /= 1.0

    if config.N_spatial_dim == 3:
        particle_grid = gaussian_splat_3d(pos, sigma=config.sigma, grid_size=32, normalize=True)
    else:
        particle_grid = gaussian_splat_data(pos, config=config)

    loss = torch.mean((img_grid - particle_grid) ** 2)

    return loss