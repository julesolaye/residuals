import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Malthus_estimator_exp(object):
    """
    This class allows to estimate the Malthusian coefficient, and to plot a curve
    with it and experimental data. The estimation can be done with two different
    methods, coded in two distinct functions. This class is different from the one
    for the simulation, do not confuse the two versions.

    Attributes
    ------------
    times : "np.array"
        Times of experimental data.

    intensities : "np.array"
        Intensities of experimental data.
    """

    def __init__(self, times: "np.array", intensities: "np.array"):

        # Save attributes
        self.times = times
        self.intensities = intensities

        # Creation directory
        self.create_path()

    def estim_malthus(
        self, fluo_start: float = 15, fluo_end: float = 50,
    ):
        """
        This function allows to estimate malthusian coefficient with experimental
        data. The method will be as follows: for each experiment, we will 
        use a range of intensities (to avoid errors linked to lag or saturation
        period), and perform a linear regression between the log of intensities 
        and the time. The estimated directing coefficient gives an estimation of 
        the malthusian coefficient for each experiment. The final estimator will 
        be the mean of all of the estimated coefficient.

        Parameters
        ------------
        fluo_start : float 
            Start of the range of intensity we will use for the linear regression.

        fluo_end : float 
            End of the range of intensity we will use for the linear regression.     
        """

        # Estimation of malthusian coefficient for each experiment
        all_malthus_estimated = list()
        for k in range(0, self.times.shape[0]):

            times_used = self.times[k]
            intensities_used = self.intensities[k]

            index_start = np.where(intensities_used >= fluo_start)[0]
            index_end = np.where(intensities_used >= fluo_end)[0]

            # The regression is done only if index with this condition exists,
            # otherwise the experiment is ignored
            if (len(index_start) > 0) and (len(index_end) > 0):

                # Linear regression to estimate the malthusian coefficient
                times_reg = times_used[index_start[0] : index_end[0]].reshape(-1, 1)
                log_cells = np.log(intensities_used[index_start[0] : index_end[0]])

                linear_regression = LinearRegression().fit(times_reg, log_cells)
                all_malthus_estimated.append(linear_regression.coef_[0])

        all_malthus_estimated = np.array(all_malthus_estimated)
        self.malthus = np.mean(all_malthus_estimated)

    def estim_malthus_mean(
        self, fluo_start: float = 15, fluo_end: float = 50, plot: bool = True,
    ) -> float:
        """
        This function allows to estimate malthusian coefficient with experimental
        data. The method will be as follows: we will perform a linear regression 
        between the log of intensities versus times (on a range of intensities).
        The estimated directing coefficient will be the estimator of the malthusian
        coefficient.

        Parameters
        ------------
        fluo_start : float 
            Start of the range of intensity we will use for the linear regression.

        fluo_end : float 
            End of the range of intensity we will use for the linear regression.   

        plot : bool
            If it is True, we will plot the curve of mean of intensities 
            versus time.
        """
        ## Estimation of malthusian coefficient with mean
        mean_intensities = np.nanmean(self.intensities, axis=0)
        index_start = np.where(mean_intensities >= fluo_start)[0][0]
        index_end = np.where(mean_intensities >= fluo_end)[0][0]
        linear_regression = LinearRegression().fit(
            self.times[0, index_start:index_end].reshape(-1, 1),
            np.log(mean_intensities[index_start:index_end]),
        )
        self.malthus = linear_regression.coef_[0]

        ## Plot regression range if it is specified
        if plot:
            fig = plt.figure()
            plt.plot(self.times[0], mean_intensities)
            plt.xlabel("Time")
            plt.ylabel("Intensities")
            plt.title(
                "Mean of intensities versus time", fontweight="bold",
            )
            plt.axvline(
                x=[self.times[0, index_start]],
                linestyle="dashed",
                color="red",
                label="Range regression",
            )
            plt.axvline(x=[self.times[0, index_end]], linestyle="dashed", color="red")
            plt.legend()
            plt.savefig(
                os.path.join(self.path_name, "range_mean_regression_experimental.png")
            )

    def plot_coeff_estim(
        self, type_estim: str, intercept: float = -1.8,
    ):
        """
        This function allows to plot the log of data, with a line of directing 
        coefficient the estimated malthusian coefficient. This allow to verify 
        if the estimation is good or not. We will launch this function by choosing 
        which method of estimation we will use.

        Parameters
        ------------
        type_estim: str
            Type of the estimation we will use here.

        intercept : float 
            Valeur of the intercept in the line we will plot, with directing coefficient
            the estimated malthusian coefficient.  
        """
        ## Estimation
        if type_estim == "all":
            self.estim_malthus()
            file_name = "estimation_malthus_experimental_all.png"

        if type_estim == "mean":
            self.estim_malthus_mean()
            file_name = "estimation_malthus_experimental_mean.png"

        ## Plot
        fig = plt.figure()
        plt.plot(self.times.T, np.log(self.intensities.T))
        plt.plot(
            self.times[0, :],
            self.times[0, :] * self.malthus + intercept,
            color="black",
            linewidth=2.5,
            label="Estimated coefficient = {}".format(np.round(self.malthus, 3)),
        )
        plt.xlabel("Times")
        plt.ylabel("Log intensities")
        plt.title(
            "Comparaison between experimental data and estimated malthusian coefficient",
            fontweight="bold",
        )
        plt.legend()
        plt.savefig(os.path.join(self.path_name, file_name))

    def create_path(self, path_name: str = "output/regression"):
        """
        This function allows to create the directory where figures will be saved 
        if it doesn't exist. 

        Parameters
        ------------
        path_name : str 
            Name of the directory where figures will be saved.
        """
        # Create directory
        if os.path.isdir(path_name):
            os.mkdir(path_name)

        # Save for the class
        self.path_name = path_name
