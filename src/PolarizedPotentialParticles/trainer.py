from particles import Particle
from configs import Config
import torch

class Trainer:
    def __init__(self, config : Config):
        self.config = config
        self.particle_system = Particle(config)
        self.x = torch.zeros((self.config.N_particles, self.config.particle_dim))  # [num_nodes, state_channels]

        self.steps = 50

        self.optim = torch.optim.Adam(self.particle_system.parameters(), lr=0.001)
        self.learning_steps = 0

        self.history = []  # to store training history (e.g., losses)

    def get_nbs(self):
        # For simplicity, we can use a fully connected graph (all particles are neighbors)
        num_particles = self.config.N_particles
        edge_index = torch.combinations(torch.arange(num_particles), r=2).t()  # [2, num_edges]
        return edge_index


    def loss_fn(self, output, target):
        # Compute your loss here based on the output and the target
        loss = torch.nn.MSELoss()(output, target)
        return loss


    def train(self):
        edge_index = self.get_nbs()  # [2, num_edges]

        # Forward pass
        output = self.particle_system(self.x, edge_index, steps = self.steps)  # [num_nodes, out_dim]


        # Here you would compute your loss and perform backpropagation
        loss = self.loss_fn(output, target)  # Compute your loss here
        # Update your model parameters here
        
        if torch.is_grad_enabled():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.learning_steps += 1
            self.history.append({"loss": loss.item()})
            # grads = [ This is makes things slow