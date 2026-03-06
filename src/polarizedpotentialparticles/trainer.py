from polarizedpotentialparticles.particles import Particle, HamiltonianParticle, PolarizedHamiltonianParticle
from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.losses import compute_loss
import torch.nn.functional as F
from torch_geometric.nn import radius_graph

import torch

class Trainer:
    def __init__(self, config : Config):
        self.config = config
        self.device = torch.device(config.device)
        # self.particle_system = Particle(config)
        self.particle_system = HamiltonianParticle(config).to(self.device)
        # self.particle_system = PolarizedHamiltonianParticle(config)

        self.optim = torch.optim.Adam(self.particle_system.parameters(), lr=0.00003)
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

        x = 2. * torch.rand((num_nodes, self.config.particle_dim), device=self.device) - 1.  # [B*N, state_channels]
        x[:, :self.config.N_spatial_dim] *= 2.

        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.config.N_particles)

        # normalize the polarization block to unit length without in-place slicing (keeps autograd happy)
        start = self.config.N_spatial_dim
        end = start + 2 * self.config.N_polarizations
        pol = x[:, start:end]
        pol = F.normalize(pol, p=2, dim=1, eps=1e-8)

        # rebuild x to avoid in-place grad issues on a view
        x = torch.cat((x[:, :start], pol, x[:, end:]), dim=1)

        return x, batch
    


    def get_initial_state(self):
        # You can choose between random or regular initial states
        # return self.get_initial_state_regular()
        # return self.get_initial_state_random()
        x, batch = self.particle_system.get_initial_state()
        return x.to(self.device), batch.to(self.device)
    

    def train_accumulated(self, optim_steps, accumulate_loss: bool, step_loss: bool):
        x, batch = self.get_initial_state()

        if torch.is_grad_enabled():
            self.optim.zero_grad()

        total_loss = x.new_zeros(())
        for i in range(optim_steps):
            output = self.particle_system(x, batch, steps=1)
            step_total_loss = output.new_zeros(())

            if accumulate_loss:
                step_total_loss = step_total_loss + compute_loss(output, self.config, batch)

            if step_loss:
                diff_move = x[:, :self.config.N_spatial_dim] - output[:, :self.config.N_spatial_dim]
                step_total_loss = step_total_loss + 0.05 * torch.mean(diff_move**2)

            if (not accumulate_loss) and (i == optim_steps - 1):
                step_total_loss = step_total_loss + compute_loss(output, self.config, batch)

            if torch.is_grad_enabled():
                (step_total_loss / optim_steps).backward()
                x = output.detach()
                x.requires_grad_(True)
            else:
                x = output

            total_loss = total_loss + step_total_loss.detach()

        loss = total_loss / optim_steps

        if torch.is_grad_enabled():
            self.optim.step()
            self.learning_steps += 1
            self.history.append({"loss": loss.item()})

    def train_unaccumulated(self, optim_steps):
        x, batch = self.get_initial_state()

        if torch.is_grad_enabled():
            self.optim.zero_grad()

        output = self.particle_system(x, batch, steps=optim_steps)
        loss = compute_loss(output, self.config, batch)
        if torch.is_grad_enabled():
            loss.backward()
            self.optim.step()
            self.learning_steps += 1
            self.history.append({"loss": loss.item()})

    def train(self, optim_steps, accumulate_loss : bool, step_loss : bool):
        if accumulate_loss or step_loss:
            self.train_accumulated(optim_steps, accumulate_loss, step_loss)
            return

        self.train_unaccumulated(optim_steps)



    def rollout(self, steps) -> list:
        was_training = self.particle_system.training
        self.particle_system.eval()

        x, batch = self.get_initial_state()
        mask0 = batch == 0
        states = [x[mask0].detach().cpu().numpy()]

        for _ in range(steps):
            x.requires_grad_(True)        # allow state grads for Hamiltonian step
            out = self.particle_system(x, batch, steps=1)
            states.append(out[mask0].detach().cpu().numpy())
            x = out.detach()              # break the graph so it doesn’t affect training

        if was_training:
            self.particle_system.train()
        return states
        

    def save_model(self, path):
        torch.save(self.particle_system.state_dict(), path)
        # save the config as well
        torch.save(self.config, path + "_config.pt")

    @staticmethod
    def load_model(path):
        # load the config first
        config = torch.load(path + "_config.pt", weights_only=False)
        trainer = Trainer(config)
        trainer.particle_system.load_state_dict(torch.load(path))
        return trainer