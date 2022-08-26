# This script allow to plot the value of the spectral gap versus parameter k

# Packages
from os.path import join
import numpy as np
import matplotlib.pyplot as plt


def plot_spectral_gap(
    start_param_k: float,
    end_param_k: float,
    name_fig: str,
    step_param: float = 0.01,
    with_vline: bool = False,
):
    """
    This function allow to plot spectral gap versus parameters k (we suppose 
    thrta = 1) on a given range of parameters. We will represent the integers
    with blue points, and if specified we will plot a vertical line which cut the 
    abscissa.

    Parameters
    ------------
    start_param_k : float 
        Lower bound for the range of parameters k we will use here.

    end_param_k : float 
        Higher bound for the range of parameters k we will use here.

    name_fig : str
        Name of the figure we will use when we will save it.

    step_param : float 
        Step time we will use for the range of parameters.

    with_vline : bool    
        If it is True, will plot on the figure the line which cut the axe of abscissa.
    """
    # Spectral gap
    all_params_k = np.arange(start_param_k, end_param_k, step_param)
    all_spectral_gap = np.zeros(len(all_params_k))
    for index in range(0, len(all_params_k)):

        all_spectral_gap[index] = (
            (2 ** (1 / all_params_k[index]))
            * (2 * np.cos(2 * np.pi / all_params_k[index]) - 1)
            - 1
        ) / 2

    # Spectral gap for integers
    start_param_k_integer = max(int(start_param_k), 1)
    end_param_k_integer = max(int(end_param_k), 1)

    all_params_k_integer = np.arange(start_param_k_integer, end_param_k_integer,)
    all_spectral_gap_integer = np.zeros(len(all_params_k_integer))
    for index in range(0, len(all_params_k_integer)):

        all_spectral_gap_integer[index] = (
            (2 ** (1 / all_params_k_integer[index]))
            * (2 * np.cos(2 * np.pi / all_params_k_integer[index]) - 1)
            - 1
        ) / 2

    ## Plot
    abs_abs = [start_param_k, end_param_k]
    ord_abs = [0, 0]
    abs_dashed_line = 57.24

    size_fig = 10
    fig = plt.figure(figsize=(size_fig, size_fig))
    plt.plot(
        all_params_k, all_spectral_gap, label="Spectral gap", color="blue",
    )
    plt.plot(abs_abs, ord_abs, label="y = 0", color="black")
    plt.plot(
        all_params_k_integer,
        all_spectral_gap_integer,
        linestyle=" ",
        marker="o",
        color="blue",
        label="Integer",
    )

    if with_vline:
        plt.axvline(
            x=abs_dashed_line, linestyle="dashed", color="red", label="x = 57.24"
        )
    plt.xlabel("Parameter k")
    plt.ylabel("Spectral gap")
    plt.title("Spectral gap versus parameter k", fontweight="bold")
    plt.legend()
    plt.savefig(join("output", name_fig))


if __name__ == "__main__":

    ## Low parameters curve
    plot_spectral_gap(
        start_param_k=0.2, end_param_k=4, name_fig="low_parameters_spectral_gap.png"
    )

    ## Mid parameters curve
    plot_spectral_gap(
        start_param_k=4, end_param_k=40, name_fig="mid_parameters_spectral_gap.png"
    )

    ## Parameters with cut the abscisse curve
    plot_spectral_gap(
        start_param_k=30,
        end_param_k=70,
        name_fig="cut_parameters_spectral_gap.png",
        with_vline=True,
    )

    ## High parameters curve
    plot_spectral_gap(
        start_param_k=50,
        end_param_k=500,
        name_fig="high_parameters_spectral_gap.png",
        with_vline=True,
    )
