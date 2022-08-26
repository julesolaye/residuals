# This script allows to compare the two terms which compose residuals for two
# specific time steps: ln(2)/malthus and its half.
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.linear_model import LinearRegression
from multitype_simulation import Multitype


if __name__ == "__main__":

    # Parameters
    n_types = 80
    rate_event = 3

    # Important elements
    multitype = Multitype(n_types, rate_event)
    all_time_step = [np.log(2) / (2 * multitype.malthus), np.log(2) / multitype.malthus]
    name_time_step = [  # Name of the two steps, we will use it when we will save our figure
        "half_step_log_2_malthus",
        "step_log_2_malthus",
    ]

    ### We will plot curve for each term, then overlap the two curves, and finally
    ### plot the sum of the two.
    for index_simul in range(0, len(all_time_step)):

        # Name of the directory where we will save results
        path_name = "output/comparaison_term/" + name_time_step[index_simul]
        if not os.path.isdir(path_name):
            os.makedirs(path_name)

        ## Simulation of residuals
        times, residuals, first_term, second_term = multitype.simulate_residuals(
            all_time_step[index_simul]
        )

        ## Plot first term
        fig = plt.figure()
        plt.plot(times, first_term, color="blue", label="First term")
        plt.xlabel("Time")
        plt.ylabel("First term")
        plt.title("First term residuals versus time", fontweight="bold")
        plt.legend()
        plt.savefig(
            os.path.join(path_name, name_time_step[index_simul] + "_first_term.png")
        )

        ## Plot second term
        fig = plt.figure()
        plt.plot(times, second_term, color="red", label="Second term")
        plt.xlabel("Time")
        plt.ylabel("Value second term")
        plt.title("Second term residuals versus time", fontweight="bold")
        plt.legend()
        plt.savefig(
            os.path.join(path_name, name_time_step[index_simul] + "_second_term.png")
        )

        ## Overlap terms
        fig = plt.figure()
        plt.plot(
            times, first_term, color="blue", label="First term",
        )
        plt.plot(
            times, second_term, color="red", label="Second term",
        )
        plt.xlabel("Time")
        plt.ylabel("Terms")
        plt.title("Comparison of the two terms versus time", fontweight="bold")
        plt.legend()
        plt.savefig(
            os.path.join(path_name, name_time_step[index_simul] + "_two_terms.png")
        )

        ## Sum terms
        fig = plt.figure()
        plt.plot(times, residuals, color="green", label="Sum of two terms")
        plt.xlabel("Time")
        plt.ylabel("Residuals")
        plt.title("Sum of two terms (residuals) versus time", fontweight="bold")
        plt.legend()
        plt.savefig(
            os.path.join(path_name, name_time_step[index_simul] + "_sum_term.png")
        )

