# Packages
import os
import matplotlib.pyplot as plt
from malthus_estimator_experimental import Malthus_estimator_exp
from residuals import Residuals
from overlapping_data import Overlapping_data


class Data_analyser(object):
    """
    This class allows to analyse data. We will do the following analysis:

    - compute malthusian coefficient with a linear regression in a range of 
    times,

    - compute residuals,

    - see if we have a renormalization in square root, by seeing if the variance
    of the residuals is of order malthus,

    - superpose data with some parameters we have estimated, see if is gaussian
    or not.

    Some figure will be plot and saved linked to all these analysis. This class
    will do all of the analysis thanks to some subclasses.

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

        #  Data
        self.times = times
        self.intensities = intensities

        # Important classes
        self.malthus_estimator = Malthus_estimator_exp(times, intensities)
        self.residuals = Residuals(times, intensities, time_step_residuals, step_array)
        self.overlapping_data = Overlapping_data(
            times, intensities, time_step_residuals, step_array
        )

        # Creation of directory where the figure will be plot
        self.create_path()

    def plot_data(self):
        """
        This function simply allows to plot data after they have been loaded. 
        """
        fig = plt.figure()
        plt.plot(self.times.T, self.intensities.T)
        plt.xlabel("Time")
        plt.ylabel("Intensities")
        plt.title(
            "Intensities versus time for experimental data", fontweight="bold",
        )
        plt.savefig("output/all_data/all_data_experimental.png")

    def create_path(self, path_name: str = "output/all_data"):
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
