from dataclasses import dataclass
from typing import Callable, Literal




@dataclass
class SimulationConfig:
    dt : float = 0.1
    steps : int = 50
    batch_size : int = 32


@dataclass
class ParticleConfig:
    hidden_dim : int = 0
    message_latent_dim : int = 8
    zero_initialization : bool = True


@dataclass
class LossConfig:
    target : Literal["square", "circle", "oval", "donut"] = "square"

    sigma = 0.02

    use_state_pool : bool = True

    

@dataclass
class Config:
    particle_config : ParticleConfig
    simulation_config : SimulationConfig
    loss_config : LossConfig

    N_spatial_dim : int = 2
    N_polarizations : int = 1
    N_particles : int = 55

    neighbor_radius : float = 0.1

    device : Literal["cpu", "cuda"] = "cuda"

    @property
    def message_channels(self) -> int:
		# rel_ij =  Dist_ij, 

        #           dot(pi,pj), 
        #           dot(qi, qj), 

        #           dot(r_ij, pi), 
        #           dot(r_ij, qi), 

        #           hidden_j - hidden_i, 
        #           hidden_j, 
        #
        #           # dim = 1 + 2 + 2 + 2*n_hidden_dim

        return 1 + self.N_polarizations + self.N_polarizations + 2 * self.particle_config.hidden_dim
    
    @property
    def state_dim(self) -> int:
        # state: polarizations + hidden_dims
        return self.N_spatial_dim * self.N_polarizations + self.particle_config.hidden_dim
    
    @property
    def particle_dim(self) -> int:
        return self.N_spatial_dim + self.state_dim
    
    @property
    def out_dim(self) -> int:
        return self.N_spatial_dim * self.N_polarizations + self.particle_config.hidden_dim

