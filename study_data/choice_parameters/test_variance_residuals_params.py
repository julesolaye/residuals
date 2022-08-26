# This script allows to test for different parameters variance of residuals, in
# order to fit these with data.

# Packages
import numpy as np
import pandas as pd
from simulator_cells_dynamics import *
import warnings

# Don't show warnings when we launch script (more ergonomic, to be removed if needed).
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    ## Range of parameters we will use for the test
    all_param_k = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 1.5, 3, 10])
    malthus = 0.45
    all_params_theta = (2 ** (1 / all_param_k) - 1) / malthus

    ## Simulation parameters
    n_cells_init = 1
    age_cells_init = np.zeros(1)
    max_cells = 6000
    pas_temps = 1
    n_simulations = 500

    ## Test for each parameters
    variances_residuals = np.zeros(len(all_param_k))
    for index_param in range(0, len(all_param_k)):

        param_1_k = all_param_k[index_param]
        param_2_theta = all_params_theta[index_param]

        law_gamma = lambda: gamma(param_1_k, param_2_theta)
        all_value_process_before = np.zeros(n_simulations)
        n_final_cells = np.zeros(n_simulations)
        residuals = np.zeros(n_simulations)

        malthus = (2 ** (1 / param_1_k) - 1) / param_2_theta

        for simul in range(n_simulations):

            # Simulation
            (
                times_,
                n_cells_before,
                age_cells_before,
                time_between_division,
            ) = simulate_division_number(
                n_cells_init, age_cells_init, law_gamma, max_cells
            )
            value_process_before = n_cells_before[-1]
            all_value_process_before[simul] = value_process_before

            # We compute final cells number after step time
            times, n_cells, age_cells = simulate_division_time(
                value_process_before,
                age_cells_before,
                law_gamma,
                pas_temps,
                time_between_division,
            )
            n_final_cells[simul] = n_cells[-1]

            # Compute of residuals
            indice_max_moins_pas = np.where(times)

            residuals[simul] = (
                n_cells[-1] - value_process_before * np.exp(pas_temps * malthus)
            ) / (np.sqrt(value_process_before))

        ## Save variance
        variances_residuals[index_param] = np.var(residuals)

    ## Save all variances in a dataframe
    variances_dataframe = pd.DataFrame(
        data=np.array([all_param_k, variances_dataframe]),
        columns=["Parameter k", "Variances"],
    )

