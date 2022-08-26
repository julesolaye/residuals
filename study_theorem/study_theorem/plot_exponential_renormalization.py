# Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from info_residuals import Info_residuals


class Plotter_exponential(object):
    """ 
    This class contains all the functions which can be used when we will plot 
    figures in order to illustrate the theorem when we have the exponential 
    renormalization. We will plot:

    - results of linear regression in order to estimate the renormalisation,
    - the evolution of histograms with growing time,
    - some individual curves of residuals.

    The figures are a little bit different from those done with square root
    renormalization that's why we use two different classes even if there are 
    similar.

    Parameters
    ------------
    info_residuals: Info_residuals
        Contains all informations linked to residuals (malthusian coeff, n_simulations,
        time step, data etc...). We will load all of the information which is in
        this class.

    name_distribution : str
        Name of the distribution which has been used to create data.

    parameter_1_distrib : float 
        Value of the first parameter of the distribution (name of parameters depend
        on distribution, but first parameter corresponds to parameter k for gamma, 
        and mean parameter for truncated gaussian and lognormal).

    parameter_2_distrib : float
        Value of the second parameter of the distribution (name of parameters depend
        on distribution, but first parameter corresponds to parameter theta for gamma, 
        and standard deviation for truncated gaussian and lognormal).
    """

    def __init__(
        self,
        info_residuals: Info_residuals,
        name_distrib: str,
        parameter_1_distrib: float,
        parameter_2_distrib: float,
    ):

        # Important parameters from info residuals
        self.load_info_residuals(info_residuals)

        # Distrib and parameters
        self.name_distrib = name_distrib
        self.parameter_1_distrib = parameter_1_distrib
        self.parameter_2_distrib = parameter_2_distrib

    def plot_all_curves(self):
        """
        This function will launch the function which create path where figures
        are saved, then launch the plotting of all of the figures (with the help
        of functions of the class). 
        """
        ### Create path
        self.create_path()

        ### Plot
        self.plot_linear_regressions_results()
        self.plot_renormalization_residuals()
        self.plot_histograms()
        self.plot_individual_curves()
        self.plot_histograms_limit_martingal()

    def plot_linear_regressions_results(self):
        """
        This function allows to plot variance of residuals versus times, with 
        the results of the estimation of exponential coefficient on the figure (
        and compare this value with the malthusian coefficient). 
        """
        ### Plot to illustrate regression
        size_fig = 10
        fig = plt.figure(figsize=(size_fig, size_fig))
        plt.plot(
            self.times_residuals,
            np.var(self.all_residuals, axis=0),
            label="Coeff exp = {}, malthus = {}".format(
                np.round(2 * self.coeff_var, 3), np.round(self.malthus, 3),
            ),
            color="green",
        )
        plt.xlabel("Time")
        plt.ylabel("Variance residuals")
        plt.title(
            "Variance residuals versus time, " + self.end_title, fontweight="bold",
        )
        plt.legend()
        plt.savefig(
            os.path.join(
                self.path_name,
                self.name_distrib
                + "_param1_"
                + str(self.parameter_1_distrib)
                + "_param2_"
                + str(self.parameter_2_distrib)
                + "_variance_residuals.png",
            ),
        )

    def plot_renormalization_residuals(self):
        """
        This function allows to plot the variance of renormalized residuals versus
        time. It allows to see if the renormalization was good or not.
        """
        ### Plot with renormalization
        size_fig = 10
        fig = plt.figure(figsize=(size_fig, size_fig))
        plt.plot(
            self.times_residuals,
            np.var(self.all_residuals_renormalized, axis=0),
            color="green",
        )
        plt.xlabel("Time")
        plt.ylabel("Variance renormalized residuals")
        plt.title(
            "Variance renormalized residuals versus time" + self.end_title,
            fontweight="bold",
        )
        plt.savefig(
            os.path.join(
                self.path_name,
                self.name_distrib
                + "_param1_"
                + str(self.parameter_1_distrib)
                + "_param2_"
                + str(self.parameter_2_distrib)
                + "_variance_residuals_renorm.png",
            ),
        )

    def plot_histograms(self, n_histogram: int = 6):
        """
        This function allows to plot histograms of residuals for several times. 
        We will take growing time in order to show that the residuals tends to
        become gaussian.

        Parameters
        ------------
        n_histogram : int 
            Number of histograms we will plot.
        """
        ### Preprocess
        # Choix temps
        round_time = 10
        step_time = int(
            (np.round(self.times_residuals[-1] / n_histogram, 1) // round_time)
            * round_time
        )
        self.times_test_norm = np.arange(
            step_time, (n_histogram + 1) * step_time, step_time
        )  # Save
        self.residuals_plot_hist = np.zeros((self.n_simuls, n_histogram))
        for simul in range(0, self.n_simuls):

            times = self.all_times[simul]
            n_cells = self.all_cells[simul]
            for index_time_residu in range(0, n_histogram):
                index_temps = np.where(
                    self.times_test_norm[index_time_residu] >= times
                )[0][-1]
                index_time_step = np.where(
                    self.times_test_norm[index_time_residu] + self.time_step >= times
                )[0][-1]

                self.residuals_plot_hist[simul, index_time_residu] = (
                    n_cells[index_time_step]
                    - np.exp(self.malthus * self.time_step) * n_cells[index_temps]
                ) / np.exp((self.coeff_var) * times[index_temps])

        ### Plot
        # Parameters
        size_fig = 12
        n_rows = 2
        n_cols = n_histogram // n_rows
        n_bins = 25
        figs, axs = plt.subplots(n_rows, n_cols, figsize=(size_fig, size_fig))
        for index_plot in range(0, n_histogram):

            # Shapiro test
            test_shapiro = scipy.stats.shapiro(self.residuals_plot_hist[:, index_plot])
            pvalue_test = test_shapiro[1]

            # Plot
            axs[index_plot // n_cols, index_plot % n_cols,].hist(
                self.residuals_plot_hist[:, index_plot],
                n_bins,
                color="blue",
                label="variance = {}".format(
                    np.round(np.var(self.residuals_plot_hist[:, index_plot]), 3)
                ),
            )
            axs[index_plot // n_cols, index_plot % n_cols,].set_title(
                "Time {}, pval = {}".format(
                    self.times_test_norm[index_plot], pvalue_test
                )
            )

            if index_plot % n_cols == 0:
                axs[index_plot // n_cols, index_plot % n_cols,].set_xlabel(
                    "Renormalized residuals"
                )
            axs[index_plot // n_cols, index_plot % n_cols,].set_ylabel("Occurrences")
            axs[index_plot // n_cols, index_plot % n_cols,].legend()

        main_title = figs.suptitle(
            "Histograms residus for different times" + self.end_title,
            fontweight="bold",
        )
        plt.savefig(
            os.path.join(
                self.path_name,
                self.name_distrib
                + "_param1_"
                + str(self.parameter_1_distrib)
                + "_param2_"
                + str(self.parameter_2_distrib)
                + "_all_histograms.png",
            ),
            bbox_extra_artists=[main_title],
        )

    def plot_individual_curves(self):
        """
        This functions allows to plot curves of residuals for a few simulations.
        It allows to have an idea of the evolution of residuals at an individual 
        point of view. 
        """
        ## Plot des courbes individuelles
        n_residuals_to_plot = 9
        n_rows = 3
        n_cols = n_residuals_to_plot // n_rows
        size_fig = 12
        figs, axs = plt.subplots(n_rows, n_cols, figsize=(size_fig, size_fig))

        for index_plot in range(0, n_residuals_to_plot):

            axs[index_plot // n_cols, index_plot % n_cols,].plot(
                self.times_residuals,
                self.all_residuals_renormalized[index_plot, :],
                color="green",
            )
            axs[index_plot // n_cols, index_plot % n_cols,].set_xlabel("Temps")
            axs[index_plot // n_cols, index_plot % n_cols,].set_ylabel("Valeur résidus")

        main_title = figs.suptitle(
            "Residuals versus time for different simulations" + self.end_title,
            fontweight="bold",
        )
        plt.savefig(
            os.path.join(
                self.path_name,
                self.name_distrib
                + "_param1_"
                + str(self.parameter_1_distrib)
                + "_param2_"
                + str(self.parameter_2_distrib)
                + "_indiv_curves.png",
            ),
            bbox_extra_artists=[main_title],
        )

    def plot_histograms_limit_martingal(self, index_start_analysis: int = 750):
        """
        This function allows to plot histogram of the limit random variable in 
        the case of an exponential renormalization. It plots the histogram of 
        the modulus of the random variable, and histogram at a certain time 
        of the argument of the random variable. It allows to highlight the 
        random modulus and phasis of the limit random variable.

        Parameters
        ------------
        index_start_analysis : int 
            Index of the array  all_residuals_renormalized where we consider that
            residuals have almost converged.
        """
        # Compute elements
        all_modulus = (
            np.max(self.all_residuals_renormalized[:, index_start_analysis:], axis=1)
            - np.min(self.all_residuals_renormalized[:, index_start_analysis:], axis=1)
            / 2
        )
        all_arguments = np.arccos(self.residuals_plot_hist[:, -1] / all_modulus)

        # Modules
        size_fig = 12
        n_bins = 25
        fig = plt.figure(figsize=(size_fig, size_fig))
        plt.hist(all_modulus, n_bins, color="brown")
        plt.xlabel("Modulus")
        plt.ylabel("Occurences")
        main_title = plt.title(
            "Modulus of limit random variable for several simulations" + self.end_title,
            fontweight="bold",
        )
        plt.savefig(
            os.path.join(
                self.path_name,
                self.name_distrib
                + "_param1_"
                + str(self.parameter_1_distrib)
                + "_param2_"
                + str(self.parameter_2_distrib)
                + "_modulus.png",
            ),
            bbox_extra_artists=[main_title],
        )

        # Phases
        size_fig = 12
        n_bins = 25
        fig = plt.figure(figsize=(size_fig, size_fig))
        plt.hist(all_arguments, n_bins, color="brown")
        plt.xlabel("Argument (radians)")
        plt.ylabel("Occurences")
        main_title = plt.title(
            "Phase à t = {} pour plusieurs simulations".format(self.times_test_norm[-1])
            + self.end_title,
            fontweight="bold",
        )
        plt.savefig(
            os.path.join(
                self.path_name,
                self.name_distrib
                + "_param1_"
                + str(self.parameter_1_distrib)
                + "_param2_"
                + str(self.parameter_2_distrib)
                + "_argument.png",
            ),
            bbox_extra_artists=[main_title],
        )

    def create_path(self):
        """
        This functions allows:

        1) To create (and save path name) the path where all the curves will be 
        plot (if the path does not exist). The name of the path depends on the 
        name of the distribution and the parameters.

        2) To create the end of titles of all figures we will plot. Because all 
        figures will have the name of distribution and parameters on its titles.
        """
        ### Name path and title
        if self.name_distrib == "gamma":
            self.path_name = (
                "output/gamma_paramk_"
                + str(np.round(self.parameter_1_distrib, 2))
                + "_paramtheta_"
                + str(np.round(self.parameter_2_distrib, 2))
            )
            self.end_title = "gamma distribution param k {}, param theta {}".format(
                np.round(self.parameter_1_distrib, 2),
                np.round(self.parameter_2_distrib, 2),
            )

        ### Create path
        if not os.path.isdir(self.path_name):
            os.mkdir(self.path_name)

    def load_info_residuals(self, info_residuals: Info_residuals):
        """
        This function allows to load all the information from info_residuals in
        this class. The informations are:

        - data,
        - number of simulations,
        - malthusian coefficient and exponential coefficient of variance,
        - time step,

        and must have been computed/simulated before.

        Parameters
        ------------
        info_residuals : Info_residuals
            Class which contains all the information we need to study residuals.
        """
        ## Times and cells
        self.all_times = info_residuals.all_times
        self.all_cells = info_residuals.all_cells

        # Number of simulations
        self.n_simuls = info_residuals.n_simuls

        # Eigenval
        self.malthus = info_residuals.malthus
        self.coeff_var = info_residuals.coeff_var

        # Residuals
        self.time_step = info_residuals.time_step
        self.times_residuals = info_residuals.times_residuals
        self.all_residuals = info_residuals.all_residuals
        self.all_residuals_renormalized = info_residuals.all_residuals_renormalized
