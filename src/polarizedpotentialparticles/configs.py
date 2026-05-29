from dataclasses import dataclass
from typing import Callable, Literal
from polarizedpotentialparticles.particle_types import ParticleType




@dataclass
class SimulationConfig:
    dt : float = 0.1
    steps : int = 50
    batch_size : int = 8


@dataclass
class ParticleConfig:
    hidden_dim : int = 2
    message_latent_dim : int = 8
    zero_initialization : bool = True
    # is_potential : bool = False



@dataclass
class LossConfig:
    target : Literal["square", "circle", "smallcircle", "oval", "donut", "thiccdonut", "smalldonut", "sphere", "plane", "tube"] = "square"
    sphere_target_radius : float = 0.7  # radius of the target sphere shell (only used when target == "sphere")
    sphere_target_sigma : float = 0.1   # gaussian blur thickness of the target shell (larger = thicker)
    multiple : list[str] | None = None  # if not None, train each batch element toward a different target (round-robin)
    use_state_pool : bool = False

    learning_rate : float = 0.001

    

@dataclass
class Config:
    particle_config : ParticleConfig
    simulation_config : SimulationConfig
    loss_config : LossConfig

    particle_type_name: ParticleType = ParticleType.PARTICLE

    # "none"         — no conditioning signal at all
    # "initial_only" — (default, backward-compat) target index encoded once into last hidden channel at init
    # "concat"       — target index re-injected at every GNN step as an extra transient input channel
    conditioning_mode: Literal["none", "initial_only", "concat"] = "initial_only"

    N_spatial_dim : int = 2
    N_polarizations : int = 1
    N_particles : int = 35
    starting_radius : float = 0.4
    # N_particles : int = 200

    neighbor_radius : float = 0.07*4.

    device : Literal["cpu", "cuda"] = "cuda"

    learned_sigma : bool = False
    sigma : float = (2/2.355) * 0.07


    noise_level : float = 1e-7

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

    @property
    def cond_dim(self) -> int:
        """Extra conditioning channels appended transiently to x before each GNN call.
        Only non-zero when conditioning_mode == 'concat' and multiple targets are set."""
        if self.conditioning_mode == "concat" and self.loss_config.multiple is not None:
            return 1
        return 0

