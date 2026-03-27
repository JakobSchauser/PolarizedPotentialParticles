import os
import unittest

import numpy as np
import torch
from torch_geometric.nn import radius_graph

from polarizedpotentialparticles.configs import Config, LossConfig, ParticleConfig, SimulationConfig
from polarizedpotentialparticles.custom_conv import EHNNConv
from polarizedpotentialparticles.losses import compute_loss
from polarizedpotentialparticles.particle_types import ParticleType
from polarizedpotentialparticles.particles import HEdgeParticle
from polarizedpotentialparticles.trainer import Trainer


def make_config(*, use_state_pool: bool, lr: float = 1e-4) -> Config:
    p_cfg = ParticleConfig(hidden_dim=0)
    t_cfg = SimulationConfig(dt=0.1, steps=5, batch_size=4)
    l_cfg = LossConfig(target="smalldonut", use_state_pool=use_state_pool, learning_rate=lr)

    cfg = Config(
        particle_config=p_cfg,
        simulation_config=t_cfg,
        loss_config=l_cfg,
        particle_type_name=ParticleType.EDGE_HAMILTONIAN,
        N_particles=40,
        starting_radius=0.6,
        device="cpu",
    )
    return cfg


def make_notebook_like_config() -> Config:
    p_cfg = ParticleConfig()
    t_cfg = SimulationConfig()
    l_cfg = LossConfig()

    l_cfg.target = "smalldonut"
    l_cfg.learning_rate = 1e-4

    return Config(
        particle_config=p_cfg,
        simulation_config=t_cfg,
        loss_config=l_cfg,
        particle_type_name=ParticleType.EDGE_HAMILTONIAN,
        N_particles=70,
        starting_radius=0.6,
        device="cpu",
    )


def assert_finite_tensor(tensor: torch.Tensor, stage: str) -> None:
    if not torch.isfinite(tensor).all():
        bad = (~torch.isfinite(tensor)).sum().item()
        raise AssertionError(f"Non-finite values at stage '{stage}' (count={bad})")


def first_non_finite_rollout_step(trainer: Trainer, steps: int) -> int | None:
    was_training = trainer.particle_system.training
    trainer.particle_system.eval()

    if trainer.state_pool is None:
        x, batch = trainer.get_initial_state()
    else:
        _idxs, x, batch = trainer.state_pool.sample_batch()

    if not torch.isfinite(x).all():
        return 0

    first_bad = None
    for step in range(1, steps + 1):
        x.requires_grad_(True)
        out = trainer.particle_system(x, batch, steps=1)
        if not torch.isfinite(out).all():
            first_bad = step
            break
        x = out.detach()

    if was_training:
        trainer.particle_system.train()

    return first_bad


class TestNanRootCause(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_edge_forces_empty_edges_are_zero_and_finite(self) -> None:
        cfg = make_config(use_state_pool=False)
        conv = EHNNConv(out_channels=1, config=cfg)

        x = torch.randn((12, cfg.N_spatial_dim), dtype=torch.float32, requires_grad=True)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        forces = conv.edge_forces(x, edge_index, create_graph=True)

        self.assertEqual(forces.shape, x.shape)
        assert_finite_tensor(forces, "edge_forces_empty")
        self.assertTrue(torch.allclose(forces, torch.zeros_like(forces)))

    def test_edge_forces_overlapping_particles_stay_finite(self) -> None:
        cfg = make_config(use_state_pool=False)
        conv = EHNNConv(out_channels=1, config=cfg)

        # All particles at the same position stresses the dist->0 path.
        x = torch.zeros((16, cfg.N_spatial_dim), dtype=torch.float32, requires_grad=True)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        edge_index = radius_graph(x[:, : cfg.N_spatial_dim], r=cfg.neighbor_radius, loop=False, batch=batch)

        forces = conv.edge_forces(x, edge_index, create_graph=True)
        assert_finite_tensor(forces, "edge_forces_overlap")

    def test_hedge_particle_forward_multistep_is_finite(self) -> None:
        cfg = make_config(use_state_pool=False)
        particle = HEdgeParticle(cfg)

        x, batch = particle.get_initial_state()
        out = particle(x, batch, steps=30)

        assert_finite_tensor(out, "hedge_forward")

    def test_state_pool_sanitizes_non_finite_states(self) -> None:
        cfg = make_config(use_state_pool=True)
        trainer = Trainer(cfg)
        self.assertIsNotNone(trainer.state_pool)
        assert trainer.state_pool is not None

        trainer.state_pool.states[0] = torch.full_like(trainer.state_pool.states[0], float("nan"))
        idxs, x, batch = trainer.state_pool.sample_batch()
        assert_finite_tensor(x, "state_pool_sample")

        x_out = x.detach().clone()
        x_out[0, 0] = float("nan")
        trainer.state_pool.writeback(idxs, x_out, batch)

        for i in idxs:
            self.assertTrue(torch.isfinite(trainer.state_pool.states[i]).all())

    def test_training_pipeline_reports_first_non_finite_stage(self) -> None:
        cfg = make_config(use_state_pool=True, lr=1e-4)
        trainer = Trainer(cfg)

        self.assertIsNotNone(trainer.state_pool)
        assert trainer.state_pool is not None

        for step in range(12):
            pool_indices, x, batch = trainer.state_pool.sample_batch()
            assert_finite_tensor(x, f"iter={step}:pool_sample")

            trainer.optim.zero_grad(set_to_none=True)
            out = trainer.particle_system(x, batch, steps=6)
            assert_finite_tensor(out, f"iter={step}:particle_forward")

            loss = compute_loss(out, cfg, batch)
            self.assertTrue(torch.isfinite(loss).item(), f"iter={step}:loss_non_finite")

            loss.backward()

            bad_grads = [
                name
                for name, param in trainer.particle_system.named_parameters()
                if param.grad is not None and not torch.isfinite(param.grad).all()
            ]
            self.assertFalse(
                bad_grads,
                f"iter={step}:non_finite_gradients in {bad_grads}",
            )

            torch.nn.utils.clip_grad_norm_(trainer.particle_system.parameters(), trainer.grad_clip_max_norm)
            trainer.optim.step()

            bad_params = [
                name
                for name, param in trainer.particle_system.named_parameters()
                if not torch.isfinite(param).all()
            ]
            self.assertFalse(
                bad_params,
                f"iter={step}:non_finite_parameters in {bad_params}",
            )

            trainer.state_pool.writeback(pool_indices, out, batch)
            for i, state in enumerate(trainer.state_pool.states):
                self.assertTrue(
                    torch.isfinite(state).all(),
                    f"iter={step}:non_finite_pool_state index={i}",
                )

    def test_training_pipeline_stress_diagnostic(self) -> None:
        if os.getenv("PPP_STRESS_NAN", "0") != "1":
            self.skipTest("Set PPP_STRESS_NAN=1 to run the long NaN stress diagnostic")

        cfg = make_config(use_state_pool=True, lr=5e-4)
        cfg.noise_level = 1e-5
        trainer = Trainer(cfg)

        self.assertIsNotNone(trainer.state_pool)
        assert trainer.state_pool is not None

        for step in range(250):
            pool_indices, x, batch = trainer.state_pool.sample_batch()
            assert_finite_tensor(x, f"stress_iter={step}:pool_sample")

            trainer.optim.zero_grad(set_to_none=True)
            out = trainer.particle_system(x, batch, steps=8)
            assert_finite_tensor(out, f"stress_iter={step}:particle_forward")

            loss = compute_loss(out, cfg, batch)
            self.assertTrue(torch.isfinite(loss).item(), f"stress_iter={step}:loss_non_finite")

            loss.backward()

            bad_grads = [
                name
                for name, param in trainer.particle_system.named_parameters()
                if param.grad is not None and not torch.isfinite(param.grad).all()
            ]
            self.assertFalse(
                bad_grads,
                f"stress_iter={step}:non_finite_gradients in {bad_grads}",
            )

            torch.nn.utils.clip_grad_norm_(trainer.particle_system.parameters(), trainer.grad_clip_max_norm)
            trainer.optim.step()

            bad_params = [
                name
                for name, param in trainer.particle_system.named_parameters()
                if not torch.isfinite(param).all()
            ]
            self.assertFalse(
                bad_params,
                f"stress_iter={step}:non_finite_parameters in {bad_params}",
            )

            trainer.state_pool.writeback(pool_indices, out, batch)

    def test_notebook_lr_config_mutation_does_not_update_optimizer(self) -> None:
        cfg = make_notebook_like_config()
        trainer = Trainer(cfg)

        self.assertAlmostEqual(trainer.optim.param_groups[0]["lr"], 1e-4)

        # This mirrors the notebook line: trainer.config.loss_config.learning_rate = 0.00001
        trainer.config.loss_config.learning_rate = 1e-5

        # Optimizer learning rate is unchanged unless param_groups are updated explicitly.
        self.assertAlmostEqual(trainer.optim.param_groups[0]["lr"], 1e-4)

    def test_notebook_parity_loop_diagnostic(self) -> None:
        if os.getenv("PPP_NOTEBOOK_PARITY", "0") != "1":
            self.skipTest("Set PPP_NOTEBOOK_PARITY=1 to run notebook-parity diagnostic loop")

        torch.manual_seed(0)
        cfg = make_notebook_like_config()
        trainer = Trainer(cfg)

        # Match what the notebook intends to do by applying the LR to the optimizer too.
        for group in trainer.optim.param_groups:
            group["lr"] = 1e-5

        steps = 150
        d = 10

        for ep in range(1200):
            rnd = int(torch.randint(-d, d, (1,)).item()) if d > 0 else 0
            trainer.train(steps + rnd, accumulate_loss=False, step_loss=False)

            # Check upstream training state before rollout diagnostics.
            for name, param in trainer.particle_system.named_parameters():
                self.assertTrue(
                    torch.isfinite(param).all(),
                    f"notebook_parity:ep={ep + 1}:non_finite_param:{name}",
                )

            last_loss = trainer.history[-1]["loss"]
            self.assertTrue(
                np.isfinite(last_loss),
                f"notebook_parity:ep={ep + 1}:non_finite_train_loss",
            )

            if trainer.state_pool is not None:
                pool_bad = [i for i, s in enumerate(trainer.state_pool.states) if not torch.isfinite(s).all()]
                self.assertFalse(
                    pool_bad,
                    f"notebook_parity:ep={ep + 1}:non_finite_pool_states:{pool_bad[:8]}",
                )

            if (ep + 1) % 100 == 0:
                rollout, _losses = trainer.rollout_batched(steps=2 * steps)
                last = torch.from_numpy(rollout[-1])
                if not torch.isfinite(last).all():
                    first_bad_step = first_non_finite_rollout_step(trainer, steps=2 * steps)
                    self.fail(
                        f"notebook_parity:ep={ep + 1}:rollout_non_finite; "
                        f"first_bad_rollout_step={first_bad_step}"
                    )

    def test_notebook_first_step_gradient_diagnostic(self) -> None:
        if os.getenv("PPP_NOTEBOOK_GRAD", "0") != "1":
            self.skipTest("Set PPP_NOTEBOOK_GRAD=1 to run first-step gradient diagnostic")

        torch.manual_seed(0)
        cfg = make_notebook_like_config()
        trainer = Trainer(cfg)

        for group in trainer.optim.param_groups:
            group["lr"] = 1e-5

        if trainer.state_pool is None:
            _idxs = None
            x, batch = trainer.get_initial_state()
        else:
            _idxs, x, batch = trainer.state_pool.sample_batch()

        assert_finite_tensor(x, "notebook_grad:first_step:pool_sample")

        trainer.optim.zero_grad(set_to_none=True)
        out = trainer.particle_system(x, batch, steps=150)
        assert_finite_tensor(out, "notebook_grad:first_step:forward")

        loss = compute_loss(out, cfg, batch)
        self.assertTrue(torch.isfinite(loss).item(), "notebook_grad:first_step:loss_non_finite")

        loss.backward()

        bad_grads = [
            name
            for name, param in trainer.particle_system.named_parameters()
            if param.grad is not None and not torch.isfinite(param.grad).all()
        ]
        self.assertFalse(
            bad_grads,
            f"notebook_grad:first_step:non_finite_gradients in {bad_grads}",
        )


if __name__ == "__main__":
    unittest.main()
