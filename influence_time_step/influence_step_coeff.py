# Script used to compute modulus of complex coefficient versus step time + compute
# ratio of the coefficient for two specific time step.

# Packages
import os
import numpy as np
import matplotlib.pyplot as plt
from multitype_simulation import Multitype


if __name__ == "__main__":

    # Name of the directory where we will save results
    path_name = "output/influence_step_coeff"
    if not os.path.isdir(path_name):
        os.makedirs(path_name)

    # Parameters
    n_types = 80
    rate_event = 3
    malthus = (2 ** (1 / n_types) - 1) * rate_event
    second_eigenval = (
        (2 ** (1 / n_types)) * np.exp((2 * np.pi / n_types) * 1j) - 1
    ) * rate_event

    ### Plot modulus complex coefficient versus step time
    step_time_min = 0
    step_time_max = 3 * np.log(2) / malthus + 1
    n_points = 2500
    all_step_time = np.linspace(step_time_min, step_time_max, n_points)

    fig = plt.figure()
    plt.plot(
        all_step_time,
        np.abs(
            np.exp(malthus * all_step_time) - np.exp(second_eigenval * all_step_time)
        ),
    )
    plt.axvline(
        x=[np.log(2) / malthus],
        linestyle="dashed",
        color="red",
        label="x= k.ln(2)/alpha",
    )
    plt.axvline(
        x=[2 * np.log(2) / malthus], linestyle="dashed", color="red",
    )
    plt.axvline(
        x=[3 * np.log(2) / malthus], linestyle="dashed", color="red",
    )
    plt.xlabel("Step time")
    plt.ylabel("Modulus complex coeff.")
    plt.title("Modulus complex coefficient versus time step", fontweight="bold")
    plt.legend()
    plt.savefig(os.path.join(path_name, "modulus_versus_time_step.png"))

    ### Ratio of modulus's complex coefficient for steps log(2)/(2*malthus) and log(2)/malthus.
    rapport_coefficient_theorique = np.abs(
        (
            (
                np.exp(malthus * (np.log(2) / (2 * malthus)))
                - np.exp(second_eigenval * (np.log(2) / (2 * malthus)))
            )
        )
    ) / np.abs(
        (
            np.exp(malthus * np.log(2) / malthus)
            - np.exp(second_eigenval * np.log(2) / malthus)
        )
    )
    multitype = Multitype(n_types, rate_event)
    times, n_cells = multitype.simulate()

    all_step_time = [np.log(2) / (2 * multitype.malthus), np.log(2) / multitype.malthus]
    name_step_time = [
        "ln(2)/(2*malthus)",
        "ln(2)/malthus",
    ]  # Labels associated to step
    colors = ["green", "brown"]  # Colors associated to step
    name_directory = "comparaison_curve"

    fig = plt.figure()
    for index_simul in range(0, len(all_step_time)):

        times, residuals, first_term, second_term = multitype.simulate_residuals(
            all_step_time[index_simul]
        )

        plt.plot(
            times,
            second_term,
            label="time step = {}".format(name_step_time[index_simul]),
            color=colors[index_simul],
        )

    plt.xlabel("Time")
    plt.ylabel("Second term")
    plt.title(
        "Second term vs time for different steps, ratio coefficient : {}".format(
            np.round(rapport_coefficient_theorique, 2)
        ),
        fontweight="bold",
    )
    plt.legend()
    plt.savefig(os.path.join(path_name, "ratio_modulus.png"))

