from dataclasses import dataclass


@dataclass
class ParticleConfig:
    N_spatial_dim : int = 2
    N_polarizations : int = 1
    N_particles : int = 100
    hidden_dim : int = 16
    message_out_channels : int = 16
    out_dim : int = 3
    zero_initialization : bool = False