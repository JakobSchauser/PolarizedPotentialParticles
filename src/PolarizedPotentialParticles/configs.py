from dataclasses import dataclass




@dataclass
class SimulationConfig:
    dt : float = 0.1
    steps : int = 50
    batch_size : int = 5


@dataclass
class ParticleConfig:
    hidden_dim : int = 8
    message_out_channels : int = 8

    out_dim : int = 1 + 2 * 1 + 8  # dx, dy, dpol_x, dpol_y, d_hidden1...
    zero_initialization : bool = True




    

@dataclass
class Config:
    particle_config : ParticleConfig
    simulation_config : SimulationConfig

    N_spatial_dim : int = 3
    N_polarizations : int = 1
    N_particles : int = 16*4

    neighbor_radius : float = 0.15

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