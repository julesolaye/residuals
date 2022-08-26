# Packages
import numpy as np
from sklearn.linear_model import LinearRegression


class Info_residuals(object):
    """
    This function allows allows to compute/estimate all of the informations we 
    need in order to plot our figures. The information are:

    - the malthusian coefficient, which is estimated here with a linear regression
    on a range of times,

    - time step, which have the value ln(2)/(2.malthus) 

    - the residuals, which are computed with the usual formula, that we can use
    because we have the malthusian coefficient and the time step,

    - the renormalization we must apply to residuals, which we will have with a 
    linear regression on the variance of residuals. 

    When we have all the informations, we will use an other plot in order to 
    plot our results.

    Attributes
    ------------
    all_times : "np.array"
        2D array which contains all times of all cells dynamics for each simulation.  

    all_cells : "np.array"
        2D array which contains the number of cells of all cells dynamics for 
        each simulation. 
    """

    def __init__(
        self, all_times: "np.array", all_cells: "np.array",
    ):

        # Number of simulations
        self.n_simuls = len(all_times)

        # Save initialization
        self.all_times = all_times
        self.all_cells = all_cells

    def estimate_malthus_and_timestep(
        self, start_cells: int = 1000, end_cells: int = 6000
    ):
        """ 
        This function allows to estimate malthusian coefficient and time step 
        for simulated data. To have the malthusian coefficient, we will make 
        an "individual coefficient estimation": for each simulation, we will 
        make a linear regression between log(number of cells) and times 
        on a range of times, and the directing coefficient found is the "individual 
        coefficient estimation". The mean of all the "individual coefficient 
        estimation" will be the final estimation of the malthusian coefficient.

        Parameters
        ------------
        start_cells : int 
            Lower bound for the range of times of linear regression.
    
        end_cells : int 
            Higher bound for the range of times of linear regression.
        """
        ### Malthusian coefficient estimation
        all_malthus = np.zeros(self.n_simuls)
        for simul in range(0, self.n_simuls):

            times = self.all_times[simul]
            n_cells = self.all_cells[simul]

            index_min_time = np.where(n_cells == start_cells)[0][0]
            index_max_time = np.where(n_cells == end_cells)[0][0]

            # Times we will use for the linear regression
            times_reg = times[index_min_time:index_max_time].reshape(-1, 1)
            n_cells_reg = n_cells[index_min_time:index_max_time]

            # Linear regression to estimate individual malthusian coefficient
            log_cells = np.log(n_cells_reg)
            linear_regression = LinearRegression().fit(times_reg, log_cells)

            all_malthus[simul] = linear_regression.coef_[0]
        # Final malthusian coefficient is the mean of all estimated malthusian coefficient
        self.malthus = np.mean(all_malthus)

        ### Time step
        self.time_step = (np.log(2) / self.malthus) / 2

    def compute_residuals(self):
        """
        This function allows to compute residuals of simulated data. The residuals
        are computed for each simulation and each time (when it is possible), 
        with the usual formula. 
        """

        ### Range of times where we will compute residuals
        max_time_residuals = np.min(self.all_times[:, -1] - 1.1 * self.time_step)

        ### Arrays that we will fill
        n_times_residuals = 1001
        self.times_residuals = np.linspace(0, max_time_residuals, n_times_residuals)
        self.n_cells_residuals = np.zeros((self.n_simuls, n_times_residuals))
        self.all_residuals = np.zeros((self.n_simuls, n_times_residuals))

        ### Residuals computing
        for simul in range(0, self.n_simuls):

            times = self.all_times[simul]
            n_cells = self.all_cells[simul]

            for index_time_residu in range(0, len(self.times_residuals)):
                index_in_times = np.where(
                    self.times_residuals[index_time_residu] >= times
                )[0][-1]
                index_time_step = np.where(
                    self.times_residuals[index_time_residu] + self.time_step >= times
                )[0][-1]
                self.n_cells_residuals[simul, index_time_residu] = n_cells[
                    index_in_times
                ]
                self.all_residuals[simul, index_time_residu] = (
                    n_cells[index_time_step]
                    - np.exp(self.malthus * self.time_step) * n_cells[index_in_times]
                )

    def estimate_renormalization(
        self,
        start_cells: int = 1000,
        end_cells: int = 6000,
        thresold_test_renormalisation: float = 0.05,
    ):
        """
        This function allows to know which renormalization must been applied 
        on the residuals. To estimate this, we make a linear regression between
        log(var(n_cells)) and times. From this point on, two cases emerge:

        - if the directing coefficient estimated if of the same order of the malthusian
        coefficient ("same order" is caracterised by a thresold), then the 
        renormalization is in square root,

        - else the renormalization is in exponential.

        Parameters
        ------------
        start_cells : int
            Lower bound where we will make our linear regression in order to have 
            the exponential coefficient of the variance of residuals.

        end_cells : int
            Higher bound where we will make our linear regression in order to have 
            the exponential coefficient of the variance of residuals.
        
        thresold_test_renormalisation : float 
            Thresold we will use to determine if the directing coefficient found 
            is "of the same order" or not of the malthusian coeff.
        """

        ### Linear regression to estimate the order of the variance
        # Times used for the linear regression
        index_min_time = np.where(
            np.mean(self.n_cells_residuals, axis=0) >= start_cells
        )[0][0]
        index_max_time = np.where(np.mean(self.n_cells_residuals, axis=0) <= end_cells)[
            0
        ][-1]
        times_reg = self.times_residuals[index_min_time:index_max_time].reshape(-1, 1)
        all_residuals_reg = np.var(
            self.all_residuals[:, index_min_time:index_max_time], axis=0
        )

        # Linear regression
        log_residuals = np.log(all_residuals_reg)
        linear_regression_residuals = LinearRegression().fit(times_reg, log_residuals)
        self.coeff_var = linear_regression_residuals.coef_[0] / 2

        ### Test renormalization
        self.root_renormalization = (
            np.abs(2 * self.coeff_var - self.malthus) / self.malthus
            < thresold_test_renormalisation
        )

        if self.root_renormalization:
            self.all_residuals_renormalized = self.all_residuals / np.sqrt(
                self.n_cells_residuals
            )
        else:
            self.all_residuals_renormalized = self.all_residuals / np.exp(
                (self.coeff_var) * self.times_residuals
            )
