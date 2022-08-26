# This script allows to create simulated data for the parameters found to have
# a similar malthus/variance to the experimental data (must be moved to
# study_data/analysis_data/input/experimental after creation).

# Packages
import numpy as np
from simulator_cells_dynamics import *
import os
import warnings

# Don't show warnings when we launch script (more ergonomic, to be removed if needed).
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    ### Parameters simulation
    n_cells_init = 1
    age_cells_init = np.zeros(1)
    max_cells = 13000
    pas_temps = 1
    n_simulations = 5000

    param_1_k = 0.6
    param_2_theta = 4.83
    theta = (2 ** (1 / all_param_k) - 1) / malthus

    law_gamma = lambda: gamma(param_1_k, param_2_theta)
    all_times = np.zeros((n_simulations, max_cells))
    all_cells = np.zeros((n_simulations, max_cells))

    ### Simulation
    for simul in range(n_simulations):

        print(simul)

        (times, n_cells, age_cells, time_between_division,) = simulate_division(
            n_cells_init=n_cells_init,
            age_cells_init=age_cells_init,
            law_time_div=law_gamma,
            stopping_criteria="cells",
            max_cells=max_cells,
        )

        all_times[simul] = times
        all_cells[simul] = n_cells

    ### Save
    path_name = "output/created_data_simulation"
    if not os.path.isdir(path_name):
        os.mkdir(path_name)

    np.save(
        os.path.join(path_name, "times.npy"), all_times,
    )
    np.save(
        os.path.join(path_name, "cells.npy"), all_cells,
    )

