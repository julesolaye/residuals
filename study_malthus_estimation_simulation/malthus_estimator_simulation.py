# Packages
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


class Malthus_estimator_sim(object):
    """
    This class allows to compare different estimators of malthusian coefficient.
    There are four estimators:

    - ln(N_t)/t,
    - [ln(N_{t_1}) - ln(N_{t_0})]/[t_1 - t_0],
    - linear regression between log(number of cellules) and times, with least
    squares, in a range of time [0 t_{max}], 
    - linear regression between log(number of cellules) and times, with least
    squares, in a range of time [T_{n_0}, T_{n_1}] where T_n is the first
    time where we have n cells.

    In this class, functions allows to estimate with all of these methods, to 
    plot some curves linked to the estimation and to save scores of estimation
    (the score is the relative error). "Norange methods" corresponds to the first 
    and the third estimators in the above liste (the only change is the linear
    regression), and "Range methods" corresponds to the second and the fourth 
    estimators.

    Attributes
    ------------
    all_times : "np.array"
        2D array where there are all of the times used for all simulations of cells
        dynamics.

    all_cells  : "np.array"
        2D array with the evolution of the cells number for each simulation.

    param_1_k : float
        Parameter k of gamma distribution.

    param_2_theta : float 
        Parameter theta of gamma distribution.
    """

    def __init__(
        self,
        all_times: "np.array",
        all_cells: "np.array",
        param_1_k: float,
        param_2_theta: float,
    ):

        # Number of simulations
        self.n_simuls = len(all_times)

        # Load data
        self.all_times = all_times
        self.all_cells = all_cells

        # Save params and true value of malthusian coefficient
        self.param_1_k = param_1_k
        self.param_2_theta = param_2_theta
        self.true_malthus = (2 ** (1 / param_1_k) - 1) / param_2_theta

        # Directory where will be save elements
        self.path_name = (
            "output/param_k_"
            + str(np.round(self.param_1_k, 1))
            + "_param_theta_"
            + str(np.round(self.param_2_theta, 1))
        )
        if not os.path.isdir(self.path_name):
            os.mkdir(self.path_name)

    def estim_malthus_first_method(self, with_regression: bool = False):
        """
        This funtion allows to make an estimation of the malthusian coefficient,
        with:

        - ln(N_t)/t when with_regression = False,
        - linear regression between log(number of cellules) and times, with least
        squares, in a range of time [0 t_{max}] when with_regression = True.

        The final estimator is the mean of all the estimations made for each 
        individual simulation.

        Parameters
        ------------
        with_regresion : bool
            Parameter to know if we will use the linear regression or not.
        """

        self.all_malthus_first = np.zeros(self.n_simuls)
        for simul in range(0, self.n_simuls):

            times = self.all_times[simul]
            n_cells = self.all_cells[simul]

            max_time = times[-1]
            index_max_time = np.where(times <= max_time)[0][-1]

            if with_regression:
                linear_regression = LinearRegression().fit(
                    times.reshape(-1, 1), np.log(n_cells)
                )
                self.all_malthus_first[simul] = linear_regression.coef_[0]
            else:

                self.all_malthus_first[simul] = (
                    np.log(n_cells[index_max_time]) / max_time
                )

        self.malthus_first = np.mean(self.all_malthus_first)

    def estim_malthus_second_method(
        self,
        start_cells: int = 1000,
        end_cells: int = 6000,
        with_regression: bool = False,
    ):
        """
        This funtion allows to make an estimation of the malthusian coefficient,
        with:

        - [ln(N_{t_1}) - ln(N_{t_0})]/[t_1 - t_0] when with_regression = False,
        - linear regression between log(number of cellules) and times, with least
        squares, in a range of time [T_{n_0}, T_{n_1}] where T_n is the first
        time where we have n cells (it is a stopping time) when with_regression = True.

        The final estimator is the mean of all the estimations made for each 
        individual simulation.

        Parameters
        ------------
        start_cells : int
            Value of the number of cells we will use for the first stopping time.
            It is the lower bound of the range where we will make our linear 
            regression.

        end_cells : int
            Value of the number of cells we will use for the second stopping time.
            It is the higher bound of the range where we will make our linear 
            regression.

        with_regresion : bool
            Parameter to know if we will use the linear regression or not.
        """
        self.all_malthus_second = np.zeros(self.n_simuls)
        for simul in range(0, self.n_simuls):

            times = self.all_times[simul]
            n_cells = self.all_cells[simul]

            self.index_min_time = np.where(n_cells == start_cells)[0][
                0
            ]  # Self beacause we need it after
            self.index_max_time = np.where(n_cells == end_cells)[0][0]

            # On garde que les derniers temps
            times_reg = times[self.index_min_time : self.index_max_time]
            n_cells_reg = n_cells[self.index_min_time : self.index_max_time]

            if with_regression:
                linear_regression = LinearRegression().fit(
                    times_reg.reshape(-1, 1), np.log(n_cells_reg)
                )
                self.all_malthus_second[simul] = linear_regression.coef_[0]
            else:
                self.all_malthus_second[simul] = (
                    np.log(n_cells_reg[-1] / n_cells_reg[0])
                ) / (times_reg[-1] - times_reg[0])
        self.malthus_second = np.mean(self.all_malthus_second)

    def plot_norange_method(self, index_to_plot: int, with_regression: bool = False):
        """
        This function allows to compute relative error of our estimation, and plot 
        some curves to illustrate the estimation we have done with methods without 
        range. 

        Parameters
        ------------
        index_to_plot : int 
            Index of the simulation we will use for the figures.
        
        with_regression : bool
            If the estimation has been done with Linear Regression, the curves 
            will be different (the title). The boolean allows to take this into
            account.
        """

        ### Compute score
        final_relative_error = np.abs(
            (self.malthus_first - self.true_malthus) / self.true_malthus
        )
        self.scores = final_relative_error

        ### Plot
        size_fig = 10
        fig = plt.figure(figsize=(size_fig, size_fig))
        plt.plot(
            self.all_times[index_to_plot],
            np.log(self.all_cells[index_to_plot]) / self.all_times[index_to_plot],
            label="Estimator",
        )
        plt.axhline(
            [self.true_malthus],
            linestyle="dashed",
            color="red",
            label="y = malthus_theo",
        )
        plt.xlabel("Time")
        plt.ylabel("Estimator")
        main_title = plt.title(
            "ln(N_t)/t versus time for one random simulation, relative error: {}".format(
                np.round(
                    np.abs(
                        (self.all_malthus_first[index_to_plot] - self.true_malthus)
                        / self.true_malthus
                    ),
                    3,
                )
            ),
            fontweight="bold",
        )

        ### Save
        name_fig = (
            "random_curve_method_1_param_k_"
            + str(np.round(self.param_1_k, 1))
            + "_param_theta_"
            + str(np.round(self.param_2_theta, 1))
            + ".png"
        )
        if with_regression:  # Name are differents for regression
            name_fig = "regression_" + name_fig
        plt.savefig(
            os.path.join(self.path_name, name_fig), bbox_extra_artists=[main_title],
        )

    def plot_range_method(self, index_to_plot: int, with_regression: bool = False):
        """
        This function allows to compute relative error of our estimation, and plot 
        some curves to illustrate the estimation we have done with methods with
        a range for the times. 

        Parameters
        ------------
        index_to_plot : int 
            Index of the simulation we will use for the figures.
        
        with_regression : bool
            If the estimation has been done with Linear Regression, the curves 
            will be different (the title). The boolean allows to take this into
            account.
        """
        ### Compute score
        final_relative_error = np.abs(
            (self.malthus_second - self.true_malthus) / self.true_malthus
        )
        self.scores = final_relative_error

        ### Plot
        size_fig = 10
        fig = plt.figure(figsize=(size_fig, size_fig))
        plt.plot(
            self.all_times[index_to_plot],
            np.log(self.all_cells[index_to_plot]),
            label="Estimator",
        )
        plt.axvline(
            self.all_times[index_to_plot, self.index_min_time],
            linestyle="dashed",
            color="red",
            label="Range linear regression",
        )
        plt.axvline(
            self.all_times[index_to_plot, self.index_max_time],
            linestyle="dashed",
            color="red",
        )
        plt.xlabel("Time")
        plt.ylabel("log(Cell numbers)")

        main_title = plt.title(
            "ln(N_t) versus time for one simulation, relative error: {}".format(
                np.round(
                    np.abs(
                        (self.all_malthus_second[index_to_plot] - self.true_malthus)
                        / self.true_malthus
                    ),
                    4,
                )
            ),
            fontweight="bold",
        )
        plt.legend()

        ### Save
        name_fig = (
            "random_curve_method_2_param_k_"
            + str(np.round(self.param_1_k, 1))
            + "_param_theta_"
            + str(np.round(self.param_2_theta, 1))
            + ".png"
        )
        if with_regression:  # Name are differents for regression
            name_fig = "regression_" + name_fig
        plt.savefig(
            os.path.join(self.path_name, name_fig), bbox_extra_artists=[main_title],
        )

    def plot_influence_n_simulation(self):
        """
        This function allows to see the influence of number of simulations in the
        performance of the estimator. We will compute the relative error of the 
        estimator for several number of simulation, see how it evolves. We will 
        plot this evolution and save it after that. 
        """

        ## Compute quantities
        malthus_by_n_simul = np.cumsum(self.all_malthus_first) / np.arange(
            1, self.n_simuls + 1
        )
        ecart_n_simul = (
            np.abs(malthus_by_n_simul - self.true_malthus) / self.true_malthus
        )

        ## Plot
        fig = plt.figure()
        plt.plot(
            np.arange(1, self.n_simuls + 1),
            ecart_n_simul * 100,
            label="param_1_k = {}, param_2_theta = {}".format(
                np.round(self.param_1_k, 3), np.round(self.param_2_theta, 3)
            ),
            color="blue",
        )
        plt.xlabel("Numver of simulations")
        plt.ylabel("Relative error (%)")
        main_title = plt.title("Relative error estimator versus number of simulations")
        plt.legend()
        plt.savefig(
            os.path.join(
                self.path_name,
                "influence_n_simulation_param_k_"
                + str(np.round(self.param_1_k, 1))
                + "_param_theta_"
                + str(np.round(self.param_2_theta, 1))
                + ".png",
            ),
            bbox_extra_artists=[main_title],
        )

