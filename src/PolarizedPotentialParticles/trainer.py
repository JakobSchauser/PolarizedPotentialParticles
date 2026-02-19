from polarizedpotentialparticles.particles import Particle, HamiltonianParticle
from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.losses import compute_loss
import torch.nn.functional as F
from torch_geometric.nn import radius_graph

import torch

class Trainer:
    def __init__(self, config : Config):
        self.config = config
        # self.particle_system = Particle(config)
        self.particle_system = HamiltonianParticle(config)

        self.optim = torch.optim.Adam(self.particle_system.parameters(), lr=0.0001)
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
    
    def get_initial_state_random(self):
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
    




    def get_initial_state_regular(self):
        # make a regular grid of particles as initial state
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles
        side = int(self.config.N_particles ** 0.5)
        x = (2.*torch.rand((num_nodes, self.config.particle_dim))- 1.)*0.001  # [B*N, state_channels]

        for i in range(num_nodes):
            batch_idx = i // self.config.N_particles
            particle_idx = i % self.config.N_particles
            x[i, 0] = (particle_idx % side) * 0.4 + 0.05 * torch.rand(1)  # x position with some noise
            x[i, 1] = (particle_idx // side) * 0.4 + 0.05 * torch.rand(1)  # y position with some noise

        batch = torch.arange(batch_size).repeat_interleave(self.config.N_particles)

        # normalize the polarization block to unit length without in-place slicing (keeps autograd happy)
        start = self.config.N_spatial_dim
        end = start + 2 * self.config.N_polarizations
        # pol = x[:, start:end]
        # pol = F.normalize(pol, p=2, dim=1, eps=1e-8)
        
        # for now make pol [0.,1] 
        pol = torch.zeros_like(x[:, start:end])
        pol[:, 0] = 1.

        # rebuild x to avoid in-place grad issues on a view
        x = torch.cat((x[:, :start], pol, x[:, end:]), dim=1)


        return x, batch
    
    def get_initial_state_hamiltonian(self):
        # make a regular grid of particles as initial state
        batch_size = self.config.simulation_config.batch_size
        num_nodes = batch_size * self.config.N_particles
        side = int(self.config.N_particles ** 0.5)
        x = (2.*torch.rand((num_nodes, self.config.N_spatial_dim))- 1.)*0.001  # [B*N, state_channels]

        dist = 0.1

        for i in range(num_nodes):
            batch_idx = i // self.config.N_particles
            particle_idx = i % self.config.N_particles
            x[i, 0] = (particle_idx % side) * dist + 0.0001 * torch.rand(1)  # x position with some noise
            x[i, 1] = (particle_idx // side) * dist + 0.0001 * torch.rand(1)  # y position with some noise
            if self.config.N_spatial_dim > 2:
                x[i, 2] = 2. * dist * torch.rand(1)  # z position with some noise

        # center the grid around the origin
        x[:, :2] -= dist * side / 2

        x.requires_grad_()  # we need gradients for the initial positions to compute the Hamiltonian updates
        batch = torch.arange(batch_size).repeat_interleave(self.config.N_particles)
        return x, batch

    def get_initial_state(self):
        # You can choose between random or regular initial states
        # return self.get_initial_state_regular()
        # return self.get_initial_state_random()
        return self.get_initial_state_hamiltonian()

    def train(self, optim_steps, accumulate_loss : bool):
        x, batch = self.get_initial_state()  # Initialize the state of the system


        if accumulate_loss:
            total_loss = 0.
            for _ in range(optim_steps):
                output = self.particle_system(x, batch, steps=1)  # [B*N, out_dim]
                loss = compute_loss(output, self.config, batch)
                total_loss += loss

                # add chance to loss
                diff = x - output
                total_loss += 0.1 * torch.mean(diff**2)

                x = output  # Update the state for the next step
            loss = total_loss / optim_steps
        else:
            # Forward pass
            output = self.particle_system(x, batch, steps = optim_steps)  # [B*N, out_dim]
            loss = compute_loss(output, self.config, batch)


        if torch.is_grad_enabled():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.learning_steps += 1
            self.history.append({"loss": loss.item()})


    def rollout(self, steps) -> list:
        # with torch.no_grad():
        x, batch = self.get_initial_state()  # Initialize the state of the system

        first_mask = batch == 0
        states = [x[first_mask].detach().cpu().numpy()]

        for _ in range(steps):
            output = self.particle_system(x, batch, steps=1)  # [B*N, out_dim]
            states.append(output[first_mask].detach().cpu().numpy())
            x = output  # Update the state for the next step

        return states
        
