from polarizedpotentialparticles.particles import Particle, HamiltonianParticle# PolarizedHamiltonianParticle
from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.losses import compute_loss, compute_losses
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
import json
from dataclasses import asdict

import numpy as np
import torch


class StatePool:
    def __init__(self, capacity: int, batch_size: int, config: Config, device: torch.device, seed_fn, reseed_count: int = 1):
        self.capacity = capacity
        self.batch_size = batch_size
        self.config = config
        self.device = device
        self.seed_fn = seed_fn
        self.reseed_count = reseed_count
        self.states = []
        self._write_idx = 0
        self._warm()

    def __len__(self):
        return len(self.states)

    def add(self, states):
        for state in states:
            s = state.detach().cpu().clone()
            if len(self.states) < self.capacity:
                self.states.append(s)
            else:
                self.states[self._write_idx] = s
                self._write_idx = (self._write_idx + 1) % self.capacity

    def set(self, indices, states):
        for i, state in zip(indices, states):
            self.states[i] = state.detach().cpu().clone()

    def _split_graphs(self, x, batch):
        return [x[batch == b] for b in torch.unique(batch)]

    def _stack_graphs(self, graph_states):
        x = torch.cat(graph_states, dim=0).to(self.device)
        n = graph_states[0].shape[0]
        batch = torch.arange(len(graph_states), device=self.device).repeat_interleave(n)
        x.requires_grad_(True)
        return x, batch

    def _graph_loss(self, state):
        s = state.to(self.device)
        b = torch.zeros(s.shape[0], dtype=torch.long, device=self.device)
        return compute_loss(s, self.config, b).detach().item()

    def _seed_graph(self):
        x_seed, batch_seed = self.seed_fn()
        return self._split_graphs(x_seed, batch_seed)[0].detach().cpu().clone()

    def _warm(self):
        while len(self.states) < self.capacity:
            self.add([self._seed_graph()])

    def sample_batch(self):
        if self.batch_size <= len(self.states):
            idxs = torch.randperm(len(self.states))[:self.batch_size].tolist()
        else:
            idxs = torch.randint(0, len(self.states), (self.batch_size,)).tolist()

        sampled = [self.states[i].clone() for i in idxs]
        losses = torch.tensor([self._graph_loss(g) for g in sampled])
        order = torch.argsort(losses, descending=True).tolist()

        seed_graph = self._seed_graph()
        for j in range(min(self.reseed_count, len(order))):
            sampled[order[j]] = seed_graph.clone()

        x, batch = self._stack_graphs(sampled)
        return idxs, x, batch

    def writeback(self, indices, x_out, batch_out):
        self.set(indices, self._split_graphs(x_out.detach(), batch_out))

class Trainer:
    def __init__(self, config : Config):
        self.config = config
        self.device = torch.device(config.device)
        # self.particle_system = Particle(config).to(self.device)
        self.particle_system = HamiltonianParticle(config).to(self.device)
        # self.particle_system = PolarizedHamiltonianParticle(config).to(self.device)

        self.optim = torch.optim.Adam(self.particle_system.parameters(), lr=0.01)
        self.learning_steps = 0

        self.history = []  # to store training history (e.g., losses)s

        self.state_pool = None
        if self.config.loss_config.use_state_pool:
            self.state_pool = StatePool(
                capacity=128*10,
                batch_size=self.config.simulation_config.batch_size,
                config=self.config,
                device=self.device,
                seed_fn=self.get_initial_state,
                reseed_count=6,
            )

    def get_nbs(self, x, batch):
        # get all particles within a certain radius as neighbors per graph in the batch
        edge_index = radius_graph(
            x[:, :self.config.N_spatial_dim],
            r=self.config.neighbor_radius,
            loop=False,
            batch=batch,
        )  # [2, num_edges]
        return edge_index


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
        if self.state_pool is None:
            x, batch = self.get_initial_state()
            pool_indices = None
        else:
            pool_indices, x, batch = self.state_pool.sample_batch()

        if torch.is_grad_enabled():
            self.optim.zero_grad()

        output, step_history = self.particle_system(
            x,
            batch,
            steps=optim_steps,
            return_history=True,
        )

        if accumulate_loss:
            img_loss_t = [compute_loss(state, self.config, batch) for state in step_history]
            img_loss_t = torch.stack(img_loss_t).mean()
        else:
            img_loss_t = compute_loss(output, self.config, batch)

        step_loss_t = output.new_zeros(())
        if step_loss:
            # step_history is a Python list of tensors; stack before computing temporal diffs.
            states = torch.stack([x, *step_history], dim=0)
            spatial = states[:, :, :self.config.N_spatial_dim]
            diff_move = spatial[1:] - spatial[:-1]
            step_loss_t = 1. * torch.mean(diff_move**2)


        total_loss = img_loss_t + step_loss_t

        if torch.is_grad_enabled():
            total_loss.backward()
            self.optim.step()
            self.learning_steps += 1

        self.history.append(
            {
                "loss": total_loss.item(),
                "total_loss": total_loss.item(),
                "img_loss": img_loss_t.item(),
                "step_loss": step_loss_t.item(),
            }
        )

        if self.state_pool is not None:
            self.state_pool.writeback(pool_indices, output, batch)

    def train_unaccumulated(self, optim_steps):
        if self.state_pool is None:
            x, batch = self.get_initial_state()
            pool_indices = None
        else:
            pool_indices, x, batch = self.state_pool.sample_batch()

        if torch.is_grad_enabled():
            self.optim.zero_grad()

        output = self.particle_system(x, batch, steps=optim_steps)
        loss = compute_loss(output, self.config, batch)
        if torch.is_grad_enabled():
            loss.backward()
            self.optim.step()
            self.learning_steps += 1
            self.history.append({"loss": loss.item()})

        if self.state_pool is not None:
            self.state_pool.writeback(pool_indices, output, batch)

    def train(self, optim_steps, accumulate_loss : bool, step_loss : bool):
        if accumulate_loss or step_loss:
            self.train_accumulated(optim_steps, accumulate_loss, step_loss)
            return

        self.train_unaccumulated(optim_steps)



    def rollout(self, steps) -> tuple[list, list]:
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

        losses = compute_losses(out, self.config, batch)
        losses = [l.item() for l in losses]

        return states, losses
    

    def rollout_batched(self, steps) -> tuple[list, list]:
        was_training = self.particle_system.training
        self.particle_system.eval()

        x, batch = self.get_initial_state()
        states = [x.detach().cpu().numpy()]

        for _ in range(steps):
            x.requires_grad_(True)        # allow state grads for Hamiltonian step
            out = self.particle_system(x, batch, steps=1)

            aout = out.detach().cpu().numpy()
            states.append(aout)
            x = out.detach()              # break the graph so it doesn’t affect training

        if was_training:
            self.particle_system.train()

        losses = compute_losses(out, self.config, batch)
        losses = [l.item() for l in losses]

        return states, losses
        

    def save_model(self, path):
        torch.save(self.particle_system.state_dict(), path)
        # save the config as well
        torch.save(self.config, path + "_config.pt")

        # also save the config as a json for easier loading without PyTorch

        with open(path + "_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=4)

    @staticmethod
    def load_model(path):
        # load the config first
        config = torch.load(path + "_config.pt", weights_only=False)
        trainer = Trainer(config)
        trainer.particle_system.load_state_dict(torch.load(path))
        return trainer