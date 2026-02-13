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
        #           # dim = 1 + 2 + 2 + 3*n_hidden_dim

        return 1 + 2 * self.N_polarizations + 2 * self.N_polarizations + 2 * self.hidden_dim
    
    @property
    def state_dim(self) -> int:
        # state: polarizations + hidden_dims
        return self.N_spatial_dim * self.N_polarizations + self.hidden_dim