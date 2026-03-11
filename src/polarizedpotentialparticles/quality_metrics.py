
import torch
from polarizedpotentialparticles.configs import Config
from polarizedpotentialparticles.trainer import Trainer

class QualityMetrics:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def correctness_metric(self, predictions, targets):
        raise NotImplementedError("Correctness metric not implemented yet.")
        # Placeholder for a correctness metric (e.g., MSE)
        return ((predictions - targets) ** 2).mean().item()

    def stability_metric(self, rollout):
        midpoint = len(rollout) // 2

        diffs = rollout[midpoint + 1:] - rollout[midpoint].unsqueeze(0)
        stability = torch.norm(diffs, dim=1).mean().item()

        return stability


    def robustness_metric(self, rollout):
        raise NotImplementedError("Robustness metric not implemented yet.")
        normal_correctness = self.correctness_metric(rollout)
        noise_lvl = 0.1
        noisy_config = self.config.copy()
        noisy_config.noise_level = noise_lvl
        trainer = Trainer(noisy_config)
        trainer.particle_system.load_state_dict(self.model.state_dict())

        noisy_rollout = trainer.rollout(self.config.simulation_config.rollout_steps)

        noisy_correctness = self.correctness_metric(noisy_rollout)
        robustness = 1 - (noisy_correctness / normal_correctness)

        return robustness

        
    def scalability_metric(self):
        raise NotImplementedError("Scalability metric not implemented yet.")
        # Placeholder for a scalability metric (e.g., inference time)



        return None