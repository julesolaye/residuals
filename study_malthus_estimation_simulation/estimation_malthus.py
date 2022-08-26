# This scripts allows to make the comparison between all of the estimators of
# malthusian coefficient. It plots some curves linked to this test, and
# save all results in a DataFrame

# Packages
import os
import numpy as np
import pandas as pd
from malthus_estimator_simulation import Malthus_estimator_sim
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    ### All of the parameters we will use for our test
    all_means = np.array([12, 15, 19, 32, 7, 23, 15, 17, 21, 50, 17, 13, 70, 34, 15])
    all_var = np.array([6, 8, 3, 12, 3, 18, 12, 6, 16, 36, 10, 8, 64, 7, 12])

    ### Arrays we will use to save scores
    scores_norange_method_no_reg = np.zeros(len(all_means))
    scores_norange_method_reg = np.zeros(len(all_means))
    scores_range_method_no_reg = np.zeros(len(all_means))
    scores_range_method_reg = np.zeros(len(all_means))

    for index_param in range(0, len(all_means)):

        ### Parameters
        mean = all_means[index_param]
        var = all_var[index_param]
        param_1_k = (mean ** 2) / var
        param_2_theta = var / mean

        ### Load data and class
        all_times = np.load(
            "input/all_times/all_times_gamma_"
            + str(mean)
            + "_variance_"
            + str(var)
            + ".npy"
        )
        all_cells = np.load(
            "input/all_cells/all_cells_gamma_"
            + str(mean)
            + "_variance_"
            + str(var)
            + ".npy",
        )
        malthus_estimator = Malthus_estimator_sim(
            all_times, all_cells, param_1_k, param_2_theta
        )

        ### Indew we will use for our comparison
        index_to_plot = np.random.randint(len(all_cells))

        ## Method 1 curves and scores
        malthus_estimator.estim_malthus_first_method()
        malthus_estimator.plot_norange_method(index_to_plot)
        scores_norange_method_no_reg[index_param] = malthus_estimator.scores

        malthus_estimator.estim_malthus_first_method(with_regression=True)
        malthus_estimator.plot_norange_method(index_to_plot, with_regression=True)
        scores_norange_method_reg[index_param] = malthus_estimator.scores

        ## Method 2 curves and scores
        malthus_estimator.estim_malthus_second_method()
        malthus_estimator.plot_range_method(index_to_plot)
        scores_range_method_no_reg[index_param] = malthus_estimator.scores

        malthus_estimator.estim_malthus_second_method(with_regression=True)
        malthus_estimator.plot_range_method(index_to_plot, with_regression=True)
        scores_range_method_reg[index_param] = malthus_estimator.scores

        ### Influence number of simulations
        malthus_estimator.plot_influence_n_simulation()

    ### Save score in a DataFrame
    scores_dataframe = pd.DataFrame(
        data=np.array(
            [
                all_means,
                all_var,
                scores_norange_method_no_reg,
                scores_norange_method_reg,
                scores_range_method_no_reg,
                scores_range_method_reg,
            ]
        ).T,
        columns=[
            "Means",
            "Variances",
            "Method_1_no_reg",
            "Method_1_reg",
            "Method_2_no_reg",
            "Method_2_reg",
        ],
    )

    path_name = "output/scores"
    if not os.path.isdir(path_name):
        os.mkdir(path_name)

    scores_dataframe.to_csv(os.path.join(path_name, "scores.csv"))
