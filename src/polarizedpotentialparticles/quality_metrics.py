
import copy
import math
import os
from pathlib import Path

import torch

from polarizedpotentialparticles.losses import gaussian_splat_data, get_cached_target_grid
from polarizedpotentialparticles.trainer import Trainer


class QualityMetrics:
    def __init__(
        self,
        trainer: Trainer,
        save_plots: bool = False,
        plot_dir: str = "docs/misc",
        seed: int | None = 0,
    ):
        self.trainer = trainer
        self.model = trainer.particle_system
        self.config = trainer.config

        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        self.seed = seed

        self._baseline_cache: dict | None = None

    def _set_seed(self, seed: int | None):
        if seed is None:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _simulate(self, trainer: Trainer, steps: int, noise_level: float | None = None, seed: int | None = None):
        was_training = trainer.particle_system.training
        trainer.particle_system.eval()

        self._set_seed(seed)

        old_noise_level = trainer.config.noise_level
        if noise_level is not None:
            trainer.config.noise_level = noise_level

        x0, batch = trainer.get_initial_state()
        x = x0.detach()
        mask0 = batch == 0

        states = [x[mask0].detach()]
        for _ in range(steps):
            x.requires_grad_(True)
            out = trainer.particle_system(x, batch, steps=1)
            x = out.detach()
            states.append(x[mask0].detach())

        trainer.config.noise_level = old_noise_level
        if was_training:
            trainer.particle_system.train()

        states = torch.stack(states, dim=0)
        positions = states[:, :, : self.config.N_spatial_dim]

        target_grid = get_cached_target_grid(
            self.config.loss_config.target,
            device=positions.device,
            dtype=positions.dtype,
        )

        correctness_curve = []
        for pos in positions:
            particle_grid = gaussian_splat_data(pos, self.config)
            correctness_curve.append(torch.mean((target_grid - particle_grid) ** 2))
        correctness_curve = torch.stack(correctness_curve, dim=0)

        return {
            "states": states,
            "positions": positions,
            "correctness_curve": correctness_curve,
            "steps": steps,
        }

    def _baseline(self, steps: int, force_recompute: bool = False):
        if (
            self._baseline_cache is not None
            and self._baseline_cache["steps"] == steps
            and not force_recompute
        ):
            return self._baseline_cache

        self._baseline_cache = self._simulate(self.trainer, steps=steps, seed=self.seed)
        return self._baseline_cache

    def _save_plot(self, x, y, title: str, xlabel: str, ylabel: str, filename: str) -> str | None:
        if not self.save_plots:
            return None
        import matplotlib.pyplot as plt

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        path = self.plot_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def _chamfer_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(a, b)
        return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()

    def correctness_metric(self, steps: int, force_recompute: bool = False):
        base = self._baseline(steps=steps, force_recompute=force_recompute)
        curve = base["correctness_curve"]
        plot_path = self._save_plot(
            x=list(range(len(curve))),
            y=curve.detach().cpu().tolist(),
            title="Correctness Over Rollout",
            xlabel="Step",
            ylabel="Image MSE (lower is better)",
            filename="metric_correctness.png",
        )
        return {
            "value": float(curve[-1].item()),
            "curve": curve.detach().cpu().tolist(),
            "plot": plot_path,
        }

    def stability_metric(self, steps: int, force_recompute: bool = False):
        base = self._baseline(steps=steps, force_recompute=force_recompute)
        pos = base["positions"]

        move = torch.norm(pos[1:] - pos[:-1], dim=-1)
        per_step = move.mean(dim=1)
        midpoint = len(per_step) // 2

        plot_path = self._save_plot(
            x=list(range(1, len(per_step) + 1)),
            y=per_step.detach().cpu().tolist(),
            title="Stability (Movement Magnitude)",
            xlabel="Step",
            ylabel="Mean ||x_t - x_{t-1}||",
            filename="metric_stability.png",
        )
        return {
            "value": float(per_step.mean().item()),
            "late_value": float(per_step[midpoint:].mean().item()),
            "curve": per_step.detach().cpu().tolist(),
            "plot": plot_path,
        }

    def robustness_metric(
        self,
        steps: int,
        noise_factor: float = 10.0,
        force_recompute: bool = False,
    ):
        base = self._baseline(steps=steps, force_recompute=force_recompute)

        noisy_rollout = self._simulate(
            self.trainer,
            steps=steps,
            noise_level=self.config.noise_level * noise_factor,
            seed=self.seed,
        )

        base_pos = base["positions"]
        noisy_pos = noisy_rollout["positions"]

        drift = torch.norm(noisy_pos - base_pos, dim=-1).mean(dim=1)
        spread = torch.norm(
            base_pos[-1] - base_pos[-1].mean(dim=0, keepdim=True),
            dim=-1,
        ).mean()

        score = drift[-1] / (spread + 1e-8)
        plot_path = self._save_plot(
            x=list(range(len(drift))),
            y=drift.detach().cpu().tolist(),
            title="Robustness Drift (Noisy vs Baseline)",
            xlabel="Step",
            ylabel="Mean trajectory drift",
            filename="metric_robustness.png",
        )
        return {
            "value": float(score.item()),
            "curve": drift.detach().cpu().tolist(),
            "noise_factor": noise_factor,
            "plot": plot_path,
        }

    def scalability_metric(
        self,
        steps: int,
        scale_factor: int = 2,
        downsample_mode: str = "random",
        downsample_repeats: int = 5,
        force_recompute: bool = False,
        seed: int | None = None,
    ):
        if scale_factor < 2:
            raise ValueError("scale_factor must be >= 2")

        base = self._baseline(steps=steps, force_recompute=force_recompute)
        base_final = base["positions"][-1]
        n_base = base_final.shape[0]

        cfg_big = copy.deepcopy(self.config)
        cfg_big.N_particles = self.config.N_particles * scale_factor
        big_trainer = Trainer(cfg_big)
        big_trainer.particle_system.load_state_dict(self.trainer.particle_system.state_dict(), strict=True)

        big_rollout = self._simulate(big_trainer, steps=steps, seed=seed if seed is not None else self.seed)
        big_final = big_rollout["positions"][-1]

        scores = []
        repeats = max(1, downsample_repeats)
        for i in range(repeats):
            if downsample_mode == "index":
                idx = torch.arange(n_base, device=big_final.device)
            elif downsample_mode == "random":
                gen = None
                if seed is not None or self.seed is not None:
                    seed_base = seed if seed is not None else self.seed
                    gen = torch.Generator(device=big_final.device)
                    if seed_base is not None:
                        gen.manual_seed(seed_base + i)
                idx = torch.randperm(big_final.shape[0], device=big_final.device, generator=gen)[:n_base]
            else:
                raise ValueError("downsample_mode must be 'random' or 'index'")

            sample = big_final[idx] / math.sqrt(scale_factor)
            scores.append(self._chamfer_distance(base_final, sample))

        scores = torch.stack(scores)

        plot_path = None
        if self.save_plots:
            import matplotlib.pyplot as plt

            self.plot_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(list(range(len(scores))), scores.detach().cpu().tolist())
            ax.set_title("Scalability Score Across Repeats")
            ax.set_xlabel("Repeat")
            ax.set_ylabel("Chamfer distance (lower is better)")
            ax.grid(axis="y", alpha=0.25)
            path = self.plot_dir / "metric_scalability.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            plot_path = str(path)

        return {
            "value": float(scores.mean().item()),
            "std": float(scores.std(unbiased=False).item()),
            "values": scores.detach().cpu().tolist(),
            "downsample_mode": downsample_mode,
            "scale_factor": scale_factor,
            "plot": plot_path,
        }

    def evaluate_all(
        self,
        steps: int,
        noise_factor: float = 10.0,
        scale_factor: int = 2,
        downsample_mode: str = "random",
        downsample_repeats: int = 5,
        force_recompute: bool = False,
    ):
        return {
            "correctness": self.correctness_metric(steps=steps, force_recompute=force_recompute),
            "stability": self.stability_metric(steps=steps, force_recompute=force_recompute),
            "robustness": self.robustness_metric(
                steps=steps,
                noise_factor=noise_factor,
                force_recompute=force_recompute,
            ),
            "scalability": self.scalability_metric(
                steps=steps,
                scale_factor=scale_factor,
                downsample_mode=downsample_mode,
                downsample_repeats=downsample_repeats,
                force_recompute=force_recompute,
            ),
        }
    


    def save_metrics(self, save_dir: str, metrics_results: dict):
        print(metrics_results)
        
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(metrics_results, f, indent=2)
            f.write("\n")