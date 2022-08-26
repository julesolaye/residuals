# This script allows to make analysis linked to experimental data (malthusian
# coefficient, residuals, renormalization test, superposition).

# Packages
import scipy.io
import warnings

warnings.filterwarnings("ignore")
from data_analyser import Data_analyser


if __name__ == "__main__":

    ### Data load and parameters
    données_projet = scipy.io.loadmat("input/experimental/matlab.mat")
    times = données_projet["times"]
    intensities = données_projet["resc_cleaned"]
    time_step_residuals = 1
    step_array = 0.5

    ### Data plotting
    data_analyser = Data_analyser(times, intensities, time_step_residuals, step_array)
    data_analyser.plot_data()

    ### Malthusian estimator
    data_analyser.malthus_estimator.plot_coeff_estim("mean")
    data_analyser.malthus_estimator.plot_coeff_estim("all")

    ### Renormalization test
    data_analyser.residuals.compute_residuals("all")
    data_analyser.residuals.plot_var_logvar()
    data_analyser.residuals.plot_some_residuals()

    ### Superposition
    index_to_plot = 4  # Corresponds to stopping time at 30
    data_analyser.overlapping_data.compute_residuals_bin()  # Binning
    data_analyser.overlapping_data.compute_simulated_residuals(
        time_file_name="input/simulation/times.npy",
        cells_file_name="input/simulation/cells.npy",
    )  # Simulated residuals
    data_analyser.overlapping_data.plot_experimental_histogram(index_to_plot)
    data_analyser.overlapping_data.plot_overlapped_histograms(index_to_plot)

    ### Plot other bin
    index_to_plot = 3
    data_analyser.overlapping_data.plot_experimental_histogram(index_to_plot)

    index_to_plot = 6
    data_analyser.overlapping_data.plot_experimental_histogram(index_to_plot)
