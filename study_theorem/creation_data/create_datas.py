# This script allows to create simulated data, for parameters saved in the csv
# file in input. After its creation, the files must me moved to
# "study_theorem/study_theorem/input".

# Packages
import numpy as np
import pandas as pd
from simulator_cells_dynamics import *
import os
import warnings

# Don't show warnings when we launch script (more ergonomic, to be removed if needed).
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    ### Number of simulations and parameters for each simulation
    n_simulations = 3000
    n_cells_init = 1
    age_cells_init = np.zeros(n_cells_init)
    max_cells = 8000

    ### Load parameters/distribution we will use
    param_distrib = pd.read_csv("input/parameters_to_create.csv")
    all_distrib = param_distrib["Name_distrib"].values()
    all_first_parameters = param_distrib[
        "First_param"
    ].values()  # Correspond to parameter k for gamma distrib, and mean param for truncated gaussian and lognorm
    all_second_parameters = param_distrib[
        "Second_param"
    ].values()  # Correspond to parameter theta for gamma distrib, and standard deviation param for truncated gaussian and lognorm

    ### Simulation for each param/distrib and saving
    for index_param in range(0, len(param_distrib)):

        ## Distribution and parameters
        name_distrib = all_distrib[index_param]
        first_parameter = all_first_parameters[index_param]
        second_parameter = all_second_parameters[index_param]

        ## Initialization according to the name of the distribution
        if name_distrib == "gamma":
            law_time_div = lambda: gamma(first_parameter, second_parameter)

        elif name_distrib == "trunc_norm":
            law_time_div = lambda: truncated_gaussian(first_parameter, second_parameter)

        elif name_distrib == "lognorm":
            law_time_div = lambda: lognorm(first_parameter, second_parameter)

        ## Simulation
        all_times = np.zeros((n_simulations, max_cells))
        all_cells = np.zeros((n_simulations, max_cells))
        for simul in range(n_simulations):

            times, n_cells, age_cells_, _ = simulate_division_number(
                n_cells_init, age_cells_init, law_gamma, max_cells
            )

            all_times[simul] = times
            all_cells[simul] = cells

        ## Saving
        path_name = "output/"
        file_name = (
            name_distrib
            + "_firstparam_"
            + str(first_parameter)
            + "_secondparam_"
            + str(second_parameter)
            + ".npy"
        )
        np.save(
            os.path.join(path_name, "all_times_" + file_name), all_times,
        )

        np.save(
            os.path.join(path_name, "all_cells_" + file_name), cells,
        )

