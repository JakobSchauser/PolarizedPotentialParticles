from polarizedpotentialparticles.configs import Config



def atomize_state(x, config : Config):
    # x: [num_nodes, state_channels]

    pos = x[:, :config.N_spatial_dim]  # [num_nodes, N_spatial_dim]
    
    pols = []
    for i in range(config.N_polarizations):
        start = config.N_spatial_dim + config.N_spatial_dim * i
        end = start + config.N_spatial_dim
        pol_i = x[:, start:end]  # [num_nodes, 2]
        pols.append(pol_i)

    hidden_start = config.N_spatial_dim + config.N_spatial_dim * config.N_polarizations
    hidden = x[:, hidden_start:]  # [num_nodes, hidden_dim]

    return pos, pols, hidden
