from polarizedpotentialparticles.configs import Config



def atomize_state(x, config : Config):
    # x: [num_nodes, state_channels]

    pos = x[:, :config.N_spatial_dim]  # [num_nodes, N_spatial_dim]
    
    pols = []
    for i in range(config.N_polarizations):
        pol_i = x[:, config.N_spatial_dim + config.N_spatial_dim * i : config.N_spatial_dim + config.N_spatial_dim * (i + 1)]  # [num_nodes, 2]
        pols.append(pol_i) 

    hidden = x[:, config.N_spatial_dim + config.N_spatial_dim * (i + 1):]  # [num_nodes, hidden_dim]

    return pos, pols, hidden
