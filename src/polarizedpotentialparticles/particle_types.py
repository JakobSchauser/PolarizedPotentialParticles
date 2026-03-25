from enum import Enum


class ParticleType(str, Enum):
    PARTICLE_OLD = "particle_old"
    HAMILTONIAN = "hamiltonian"
    PARTICLE = "particle"
    POLARIZED_HAMILTONIAN = "polarized_hamiltonian"
    EDGE_HAMILTONIAN = "edge_hamiltonian"
    POLARIZED_EDGE_HAMILTONIAN = "polarized_edge_hamiltonian"
