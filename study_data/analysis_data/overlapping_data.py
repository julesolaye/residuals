# Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from scipy.stats import shapiro
import seaborn as sb
from sklearn.linear_model import LinearRegression
from residuals import Residuals


class Overlapping_data(object):
    """
    This class allows to plot the overlapping of residuals between the simulations 
    with the parameters we found, and experimental data. We will also compare these
    elements with a gaussian, because renormalized residuals must converge to 
    a gaussian the dynamic is a Bellman-Harris process. The residuals of experimental 
    data has been computed with binning. We will do the same for simulations in
    order to be coherent.

    Attributes
    ------------
    times : "np.array"
        Times linked to the experiment which has been done.

    intensities : "np.array"
        Fluorescence's intensities measured during the experiment.

    time_step_residuals : float
        Time step we will use for residuals.
    
    step_array : float
        Be careful to not confuse with time step of residuals. It is the time step
        in the times array of experiment.
    """

    def __init__(
        self,
        times: "np.array",
        intensities: "np.array",
        time_step_residuals: float,
        step_array: float,
    ):

        # Load data
        self.times = times
        self.intensities = intensities

        # Save other params
        self.time_step_residuals = time_step_residuals
        self.step_array = step_array

        # Parameters linked to residuals
        self.time_step_residuals = time_step_residuals
        self.index_time_step_residuals = int((time_step_residuals) / step_array)

        # Creation directory
        self.create_path()

    def compute_residuals_bin(
        self,
        intensity_start: float = 10,
        intensity_end: float = 70,
        intensity_step: float = 5,
    ):
        """
        This function allows to bin experimental data: data are no more sorted by
        time but by intensity. Residuals seems to converge quickly that's why 
        we use this. The binning will be done in a range of intensities.

        Parameters
        ------------
        intensity_start : float
            Lower bound for the intensities range we will use for the binning.

        intensity_end : float 
            Higher bound for the intensities range we will use for the binning.

        intensity_step : float 
            Step for the intensities range we will use for the binning.
        """

        ## Compute residuals
        self.residuals_class = Residuals(
            self.times, self.intensities, self.time_step_residuals, self.step_array
        )
        self.residuals_class.compute_residuals("all")
        self.residuals = self.residuals_class.residuals
        self.intensities_for_residuals = np.arange(
            intensity_start, intensity_end, intensity_step
        )

        ## Creation of a liste with len(self.intensities_for_residuals) lists in it
        self.residuals_by_intensities = []
        for _ in range(0, len(self.intensities_for_residuals)):
            self.residuals_by_intensities.append(list())

        ## We range residuals by intensities
        for j in range(0, self.residuals.shape[0]):
            for k in range(0, self.residuals.shape[1]):

                value_intensity = self.intensities[j, k]

                if (value_intensity >= intensity_start) & (
                    value_intensity < intensity_end
                ):
                    index = int(value_intensity // intensity_step - 2)

                    self.residuals_by_intensities[index].append(self.residuals[j, k])

        # Suppresion of NaN
        for index in range(0, len(self.residuals_by_intensities)):
            self.residuals_by_intensities[index] = np.delete(
                self.residuals_by_intensities[index],
                np.isnan(self.residuals_by_intensities[index]),
            )

    def compute_simulated_residuals(
        self, time_file_name: str, cells_file_name: str,
    ) -> "np.array":

        """
        This functions allows with simulated dynamics'cells (created before):

        - to estimate the malthusian coefficient, by compute the mean of individual
        estimations made with linear regression on a range of times,

        - to estimate residuals on a range of stopping time.

        This simulated residuals will be use when we will overlap the different
        histograms.

        Parameters
        ------------
        time_file_name : str  
            Name of the file which contains all the times of simulations we have 
            done.

        cells_file_name : str
            Name of the file which contains all the cells'number of simulations we have 
            done.
        """

        all_times = np.load(time_file_name)
        all_cells = np.load(cells_file_name)

        n_simuls = len(all_times)
        all_malthus = np.zeros(n_simuls)

        # Range for linear regression to estimate malthusian coefficient
        start_cells = 100
        end_cells = 1000

        ### Malthusian coefficient estimation
        for simul in range(0, n_simuls):

            times = all_times[simul]
            n_cells = all_cells[simul]

            index_min_time = np.where(n_cells == start_cells)[0][0]
            index_max_time = np.where(n_cells == end_cells)[0][0]

            # Times we will use for our linear regression
            times_reg = times[index_min_time:index_max_time].reshape(-1, 1)
            n_cells_reg = n_cells[index_min_time:index_max_time]

            # Linear regression
            log_cells = np.log(n_cells_reg)
            linear_regression = LinearRegression().fit(times_reg, log_cells)

            all_malthus[simul] = linear_regression.coef_[0]
        # Mean of all estimations
        malthus = np.mean(all_malthus)

        ### Residuals estimation
        max_cells = 6000
        step_max_cells = 1000
        max_cells_residuals = np.arange(
            step_max_cells, max_cells + 1, step_max_cells
        )  # Range where we will compute residuals
        all_residuals = np.zeros((n_simuls, len(max_cells_residuals)))

        for simul in range(0, n_simuls):

            times = all_times[simul]
            n_cells = all_cells[simul]

            for index_max_cells_residu in range(0, len(max_cells_residuals)):

                n_cells_stopping_time = max_cells_residuals[index_max_cells_residu]
                index_time_step_res = np.where(
                    times[n_cells_stopping_time - 1] + self.time_step_residuals >= times
                )[0][-1]
                all_residuals[simul, index_max_cells_residu] = (
                    n_cells[index_time_step_res]
                    - np.exp(malthus * self.time_step_residuals) * n_cells_stopping_time
                ) / np.sqrt(n_cells_stopping_time)

        self.simulated_residuals = all_residuals

    def plot_overlapped_histograms(
        self, index_to_plot: int, n_bins: int = 30,
    ):
        """
        This function allows to make the comparison between three elements:
        - residuals of simulated data,
        - residuals of experimental data,
        - a gaussian.

        In order to do this, we will overlap:
        - histogram of residuals of simulated data and residuals of experimental data,
        - residuals of experimental data and gaussian (with shapiro test in order
        to see the gaussianity),
        - residuals of simulated data and gaussian (with shapiro test in order
        to see the gaussianity).

        All of the overlapping will be plot and save.

        Parameters
        ------------
        index_to_plot : int
            Index of the histogram we will plot.

        n_bins : int
            Number of bins we will use for the histogram (do not confuse with the
            binning).
        """

        ## Compute gaussian
        n_points_gaussian = 1000
        points_gaussian = np.linspace(
            np.min(self.residuals_by_intensities[index_to_plot]),
            np.max(self.residuals_by_intensities[index_to_plot]),
            n_points_gaussian,
        )  # Gaussienne théorique

        ## Overlapping experimental data and simulation
        size_fig = 12
        fig = plt.figure(figsize=(size_fig, size_fig))
        sb.distplot(
            self.residuals_by_intensities[index_to_plot],
            bins=n_bins,
            kde=True,
            color="blue",
            kde_kws=dict(linewidth=1.5, color="blue"),
            label="Experimental data, var = {}".format(
                np.round(np.var(self.residuals_by_intensities[index_to_plot]), 3)
            ),
        )
        sb.distplot(
            self.simulated_residuals[:, 0],
            bins=n_bins,
            kde=True,
            color="green",
            kde_kws=dict(linewidth=1.5, color="green"),
            label="Simulated data, var = {}".format(
                np.round(np.var(self.simulated_residuals[:, 0]), 3)
            ),
        )
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title(
            "Overlap of simulations and experimental data".format(), fontweight="bold",
        )
        plt.legend()
        plt.savefig(os.path.join(self.path_name, "overlap_experimental_simulation.png"))

        ## Overlapping experimental data and gaussian
        size_fig = 12
        fig = plt.figure(figsize=(size_fig, size_fig))
        sb.distplot(
            self.residuals_by_intensities[index_to_plot],
            bins=n_bins,
            kde=True,
            color="blue",
            kde_kws=dict(linewidth=1.5, color="blue"),
            label="Experimental data, var = {}".format(
                np.round(np.var(self.residuals_by_intensities[index_to_plot]), 3)
            ),
        )
        plt.plot(
            points_gaussian,
            scipy.stats.norm.pdf(
                points_gaussian,
                scale=np.std(self.residuals_by_intensities[index_to_plot]),
            ),
            color="red",
            label="Gaussian density of variance {}".format(
                np.round(np.var(self.residuals_by_intensities[index_to_plot]), 3)
            ),
        )
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title(
            "Overlap of experimental data and gaussian, pval Shapiro = {}".format(
                np.round(shapiro(self.residuals_by_intensities[index_to_plot],)[1], 3),
            ),
            fontweight="bold",
        )
        plt.legend()
        plt.savefig(os.path.join(self.path_name, "overlap_experimental_gaussian.png"))

        ## Overlapping simulations and gaussian
        size_fig = 12
        fig = plt.figure(figsize=(size_fig, size_fig))
        sb.distplot(
            self.simulated_residuals[:, 0],
            bins=n_bins,
            kde=True,
            color="green",
            kde_kws=dict(linewidth=1.5, color="green"),
            label="Simulated data, var = {}".format(
                np.round(np.var(self.simulated_residuals[:, 0]), 3)
            ),
        )
        plt.plot(
            points_gaussian,
            scipy.stats.norm.pdf(
                points_gaussian,
                scale=np.std(self.residuals_by_intensities[index_to_plot]),
            ),
            color="red",
            label="Gaussian density of variance {}".format(
                np.round(np.var(self.residuals_by_intensities[index_to_plot]), 3)
            ),
        )
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title(
            "Overlap of simulation and gaussian, pval Shapiro = {}".format(
                shapiro(self.simulated_residuals[:, 0],)[1],
            ),
            fontweight="bold",
        )
        plt.legend()
        plt.savefig(os.path.join(self.path_name, "overlap_simulation_gaussian.png"))

    def plot_experimental_histogram(self, index_to_plot: int, n_bins: int = 30):
        """
        This function allows to compute one histogram of experimental data.

        Parameters
        ------------- 
        index_to_plot : int
            Index of the histogram we will plot.

        n_bins : int
            Number of bins we will use for the histogram (do not confuse with the
            binning). 
        """
        ## Plotting
        size_fig = 12
        fig = plt.figure(figsize=(size_fig, size_fig))
        sb.distplot(
            self.residuals_by_intensities[index_to_plot],
            bins=n_bins,
            kde=True,
            color="blue",
            kde_kws=dict(linewidth=1.5, color="black"),
        )
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title(
            "Histogram of experimental residuals, T.A = {}, variance = {}".format(
                self.intensities_for_residuals[index_to_plot],
                np.round(np.var(self.residuals_by_intensities[index_to_plot]), 3),
            ),
            fontweight="bold",
        )

        ## Save
        file_name = (
            "one_histogram_experimental_"
            + str(self.intensities_for_residuals[index_to_plot])
            + ".png"
        )
        plt.savefig(os.path.join(self.path_name, file_name))

    def create_path(self, path_name: str = "output/superposition_data"):
        """
        This function allows to create the directory where figures will be saved 
        if it doesn't exist. 

        Parameters
        ------------
        path_name : str 
            Name of the directory where figures will be saved.
        """
        # Create directory
        if not os.path.isdir(path_name):
            os.mkdir(path_name)

        # Save for the class
        self.path_name = path_name
