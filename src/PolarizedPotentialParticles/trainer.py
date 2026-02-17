from polarizedpotentialparticles.particles import Particle
from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.losses import compute_loss
import torch.nn.functional as F
from torch_geometric.nn import radius_graph

import torch

class Trainer:
    def __init__(self, config : Config):
        self.config = config
        self.particle_system = Particle(config)

        self.optim = torch.optim.Adam(self.particle_system.parameters(), lr=0.001)
        self.learning_steps = 0

        self.history = []  # to store training history (e.g., losses)

    def get_nbs(self, x, batch):
        # get all particles within a certain radius as neighbors per graph in the batch
        edge_index = radius_graph(
            x[:, :self.config.N_spatial_dim],
            r=self.config.neighbor_radius,
            loop=False,
            batch=batch,
        )  # [2, num_edges]
        return edge_index

    def loss_fn(self, output, target):
        # Compute your loss here based on the output and the target
        loss = torch.nn.MSELoss()(output, target)
        return loss
    
    def get_initial_state(self):
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles

        x = 2. * torch.rand((num_nodes, self.config.particle_dim)) - 1.  # [B*N, state_channels]
        x[:, :self.config.N_spatial_dim] *= 2.

        batch = torch.arange(batch_size).repeat_interleave(self.config.N_particles)

        # normalize the polarization block to unit length without in-place slicing (keeps autograd happy)
        start = self.config.N_spatial_dim
        end = start + 2 * self.config.N_polarizations
        pol = x[:, start:end]
        pol = F.normalize(pol, p=2, dim=1, eps=1e-8)

        # rebuild x to avoid in-place grad issues on a view
        x = torch.cat((x[:, :start], pol, x[:, end:]), dim=1)

        return x, batch

    def train(self, optim_steps):
        x, batch = self.get_initial_state()  # Initialize the state of the system

        # Forward pass
        output = self.particle_system(x, batch, steps = optim_steps)  # [B*N, out_dim]


        # Here you would compute your loss and perform backpropagation
        loss = compute_loss(output, self.config, batch)
        # Update your model parameters here
        
        if torch.is_grad_enabled():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.learning_steps += 1
            self.history.append({"loss": loss.item()})


    def rollout(self, steps) -> list:
        with torch.no_grad():
            x, batch = self.get_initial_state()  # Initialize the state of the system

            first_mask = batch == 0
            states = [x[first_mask].detach().cpu().numpy()]

            for _ in range(steps):
                output = self.particle_system(x, batch, steps=1)  # [B*N, out_dim]
                states.append(output[first_mask].detach().cpu().numpy())
                x = output  # Update the state for the next step

            return states
        
