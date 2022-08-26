# This script allows to test for different parameters variance of residuals, in
# order to fit these with data.

# Packages
import numpy as np
import pandas as pd
import os
from simulator_cells_dynamics import simulate_division, gamma
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
        print("----------------")
        print("Parameter k: {}".format(all_param_k[index_param]))

        law_gamma = lambda: gamma(
            all_param_k[index_param], all_params_theta[index_param]
        )
        residuals = np.zeros(n_simulations)

        malthus = (2 ** (1 / all_param_k[index_param]) - 1) / all_params_theta[
            index_param
        ]

        for simul in range(n_simulations):

            # Simulation
            (times_, n_cells_before, time_before_div,) = simulate_division(
                n_cells_init=n_cells_init,
                law_time_div=law_gamma,
                stopping_criteria="cells",
                max_cells=max_cells,
            )
            value_process_before = n_cells_before[-1]

            # We compute final cells number after step time
            times, n_cells, _ = simulate_division(
                n_cells_init=value_process_before,
                law_time_div=law_gamma,
                stopping_criteria="time",
                max_time=pas_temps,
                time_before_div=time_before_div,
            )

            # Compute of residuals
            indice_max_moins_pas = np.where(times)
            residuals[simul] = (
                n_cells[-1] - value_process_before * np.exp(pas_temps * malthus)
            ) / (np.sqrt(value_process_before))

        ## Save variance
        variances_residuals[index_param] = np.var(residuals)

    ## Save all variances in a dataframe
    variances_dataframe = pd.DataFrame(
        data=np.array([all_param_k, variances_residuals]).T,
        columns=["Parameter k", "Variances"],
    )
    path_name = "output/variances"
    file_name = "scores.csv"
    if not os.path.isdir(path_name):
        os.mkdir(path_name)
    variances_dataframe.to_csv(os.path.join(path_name, file_name))

