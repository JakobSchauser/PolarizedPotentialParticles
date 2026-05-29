from enum import Enum


class ParticleType(str, Enum):
    PARTICLE_OLD = "particle_old"
    HAMILTONIAN = "hamiltonian"
    PARTICLE = "particle"
    POLARIZED_HAMILTONIAN = "polarized_hamiltonian"
    POLARIZED_HAMILTONIAN_WITH_HC = "polarized_hamiltonian_with_hc"
    EDGE_HAMILTONIAN = "edge_hamiltonian"
    POLARIZED_EDGE_HAMILTONIAN = "polarized_edge_hamiltonian"
    POLARIZED_EDGE_HAMILTONIAN_WITH_HC = "polarized_edge_hamiltonian_with_hc"
    DISTANCE_POLARIZED_HAMILTONIAN_WITH_HC = "distance_polarized_hamiltonian_with_hc"
