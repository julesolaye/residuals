# This script allows to launch the plotting of all figures which illustrate the
# theorem. For each parameter/distribution we will estimate malthusian coefficient,
# time step and residuals and after that plot curves of residuals, depending on
# the renormalization.

import os
import numpy as np
from info_residuals import Info_residuals
from plot_root_renormalization import Plotter_root
from plot_exponential_renormalization import Plotter_exponential

if __name__ == "__main__":

    ### All parameters we will use to study theorem
    all_means = [12, 15, 19, 32, 7, 23, 15, 17, 21, 50, 17, 13, 70, 34, 15]
    all_vars = [6, 8, 3, 12, 3, 18, 12, 6, 16, 36, 10, 8, 64, 7, 12]

    for index_param in range(0, len(all_means)):

        ### Parameters
        mean = all_means[index_param]
        var = all_vars[index_param]
        param_1_k = (mean ** 2) / var
        param_2_theta = var / mean
        name_distrib = "gamma"

        ### Load data and class
        all_times = np.load(
            "input/all_times/all_times_gamma_"
            + str(mean)
            + "_variance_"
            + str(var)
            + ".npy"
        )
        all_cells = np.load(
            "input/all_cells/all_cells_gamma_"
            + str(mean)
            + "_variance_"
            + str(var)
            + ".npy",
        )

        info_residuals = Info_residuals(all_times, all_cells)

        ### Compute malthusian coefficient, time step and residuals
        info_residuals.estimate_malthus_and_timestep()
        info_residuals.compute_residuals()
        info_residuals.estimate_renormalization()

        ### Plot all curves
        if info_residuals.root_renormalization:
            plotter = Plotter_root(
                info_residuals, name_distrib, param_1_k, param_2_theta
            )
        else:
            plotter = Plotter_exponential(
                info_residuals, name_distrib, param_1_k, param_2_theta
            )
        plotter.plot_all_curves()

