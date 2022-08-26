import numpy as np
import matplotlib.pyplot as plt
from malthus_estimator_experimental import Malthus_estimator_exp


class Residuals(object):
    """
    This class contains functions which can compute residuals, and make renormalization
    test after that.
    To compute residuals, firstly we will estimate the malthusian
    coefficient with Malthus_estimator_exp and and after we will use the usual 
    formula.
    After that we can make the renormalization test: we will plot ln(var(residuals))
    versus time and see if it is of order malthus or not. We can also plot some 
    individuals curves of renormalized residuals (but it is difficult to interpret).

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
        step_time_residuals: float,
        step_array: float,
    ):
        # Load data
        self.times = times
        self.intensities = intensities

        # Parameters linked to residuals
        self.step_time_residuals = step_time_residuals
        self.index_step_time_residuals = int((step_time_residuals) / step_array)

        # Creation directory
        self.create_path()

    def compute_residuals(self, type_estim: str):
        """
        This function allows to compute residuals. First, we will estimate the 
        malthusian coefficient with one of the two methods. After that, we use 
        the usual formula in order to compute residuals.

        Parameters
        ------------
        type_estim : str 
            Method of estimation we will use in order to have the malthusian 
            coefficient. 
        """
        ## Estimation malthusian coefficient
        malthus_estimator = Malthus_estimator_exp(self.times, self.intensities)
        if type_estim == "all":
            malthus_estimator.estim_malthus()

        if type_estim == "mean":
            malthus_estimator.estim_malthus_esperance()
        self.malthus = malthus_estimator.malthus

        ## Residuals
        self.times_residuals = self.times[0, : -self.index_step_time_residuals]
        self.residuals_not_renormalized = (
            self.intensities[:, self.index_step_time_residuals :]
            - np.exp(self.malthus * self.step_time_residuals)
            * self.intensities[:, : -self.index_step_time_residuals]
        )
        self.residuals = self.residuals_not_renormalized / np.sqrt(
            self.intensities[:, : -self.index_step_time_residuals]
        )  # Renormalization

    def plot_var_logvar(self):
        """
        This function is the principal function for the renormalization test. 
        Firstly, we compute variance (nanvariance) of the residuals not renormalized.
        After that, we compare np.log(var(residuals)) with a line of directing
        coefficient malthus. If the order is the same, it is good.
        """

        var_residuals = np.nanvar(self.residuals_not_renormalized, axis=0)
        fig = plt.figure()
        plt.plot(self.times_residuals, var_residuals)
        plt.xlabel("Time")
        plt.ylabel("Variance residuals")
        plt.title("Variance residuals versus time", fontweight="bold")
        plt.show()

        fig = plt.figure()
        intercept = -1.6
        plt.plot(
            self.times_residuals, np.log(var_residuals), label="Variance residuals",
        )
        plt.plot(
            self.times_residuals,
            self.malthus * self.times_residuals + intercept,
            label="Line with directing coefficient: malthusian coeff",
            color="black",
            linewidth=2,
        )
        plt.xlabel("Time")
        plt.ylabel("Log variance of residuals")
        plt.title("Log variance of residuals versus time", fontweight="bold")
        plt.legend()
        plt.savefig(os.path.join(self.path_name, "log_var_experimental_vs_malthus.png"))

    def plot_some_residuals(self):
        """ 
        This function allows to plot some individual renormalized residuals. But 
        it is very difficult to interpret, but can be useful in order to make 
        the renormalisation test. 
        """
        indexes_simul = [0, 1, 6, 7, 8, 25]
        n_rows = 2
        n_cols = 3
        size_fig = 12
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(size_fig, size_fig))
        n_plot = 0  # Counter

        for index_simul in indexes_simul:
            ax[n_plot // n_cols, n_plot % n_cols].plot(
                self.times_residuals, self.residuals[index_simul],
            )
            ax[n_plot // n_cols, n_plot % n_cols].set_xlabel("Times")
            ax[n_plot // n_cols, n_plot % n_cols].set_ylabel("Renormalized residuals")
            n_plot += 1

        plt.suptitle(
            "Renormalized residuals by square root for some experiments",
            fontweight="bold",
        )
        plt.savefig(os.path.join(self.path_name, "indiv_residuals_experimental.png"))

    def create_path(self, path_name: str = "output/test_renormalization"):
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
