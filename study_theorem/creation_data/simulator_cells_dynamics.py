# This class contains all functions wich can be used in order to simulate cells
# dynamics with Bellman-Harris process. It may be useful for the creation of data.

import numpy as np


def gamma(param_1_k: float, param_2_theta: float) -> float:
    """
    This function allows to draw a gamma distribution thanks to scipy.stats.

    Parameters
    ------------
    param_1_k : float
      Parameter k of gamma distribution. 

    param_2_b : float
      Parameter theta of gamma distribution.
    """

    return scipy.stats.gamma.rvs(param_1_k, scale=param_2_theta)


def truncated_gaussian(mean_normal: float, sd_normal: float) -> float:
    """
    This function allows to draw a random variable following a truncated gaussian 
    distribution (truncature at 0). This allows to have only positive values. 
    We do this with scipy.stats.

    Parameters
    ------------
    mean_normal : float
      Mean of the gaussian we will truncate.

    sd_normal : float
      Standard deviation of the gaussian we will truncate.
    """
    clip_1 = -mean_normal / sd_normal
    clip_2 = np.inf

    return scipy.stats.truncnorm.rvs(clip_1, clip_2, mean_normal, sd_normal)


def lognorm(param_1_mu: float, param_2_sigma: float) -> float:
    """
    This function allows to draw a random variable following a lognormal distribution.

    Parameters
    ------------
    param_1_mu : float
      Mean of the normal distribution linked to this lognormal.
    
    param_2_sigma : float
      Standard deviation of the normal distribution linked to this lognormal.
    """

    norm = scipy.stats.norm.rvs()
    return np.exp(param_1_mu + norm * param_2_sigma)


def simulate_division(
    n_cells_init: int,
    age_cells_init: "np.array",
    law_time_div: "function",
    stopping_criteria: str,
    max_time: float = 0,
    max_cells: float = 0,
    time_before_div: "np.darray" = np.array([-1]),
) -> ("np.ndarray", "np.ndarray", "np.ndarray"):
    """
    This function allows to simulate cells division when we suppose that the model
    is not markovian (in cells). It works as follows: we have an array which for 
    each cells contains the time before its division (draw with gamma distibution 
    when they have been created). The time of the next division is the minimum 
    of this array. At each division:

    - we add one cell in the cells number,
    - draw a new time before extinction with a gamma distribution,
    - move the time forward the next division, which is the new minimum of the 
    array.

    The stopping criteria of this function is the time (when we arrive at a 
    certain time, we stop the simulation). This is the difference with the other 
    function.

    Parameters
    ------------
    n_cells_init: int
      Cells number at the beginning of the simulation.

    law_time_div : "function"
      Function which simulate the distribution we will use for our Bellman-Harris
      process.

    stopping_criteria : str
      Stopping criteria that we will use for this simulation, it may be when we 
      arrive at a certain number of cells, or a certain time.

    max_time : float
      If the stopping criteria is on time, it will be the stopping time of the 
      simulation.

    max_cells : int
      If the stopping criteria is on cells, it will be the stopping cells number
      of the simulation.

    time_before_div : "np.darray"
      Time before next division for each cell at the beginning of the simulation.
    
    Returns
    ------------
    times : "np.ndarray"
      Values of T1, T2, T3 ... which are the times where the first, second, third
      ... simulation took place.
    
    n_cells : "np.ndarray"
      Number of cells at each time.

    time_before_division : "np.ndarray"
      Time before next division for each cells at the end of the simulation.
    """

    ### Initialization
    n_cells = [n_cells_init]
    if np.min(time_before_div) == -1:  # To verify is the parameter is specified or not
        time_before_division = np.zeros(n_cells_init)
        for k in range(0, n_cells_init):
            time_before_division[k] = law_time_div()

    else:
        time_before_division = time_before_div

    times_between_div = [0]  # Time between each division (T1, T2-T1, T3-T2 etc......)

    # Stopping criteria
    if stopping_criteria == "cells":
        stopping_test = n_cells[-1] > max_cells
    else:
        stopping_test = sum(times_between_div) + np.min(time_before_division) > max_time

    ### Loop
    while not stopping_test:

        # The cell with the lowest time to division divides
        times_between_div.append(np.min(time_before_division))

        # Update of time before division
        time_before_division -= np.min(time_before_division)

        # Time before division for each cells
        time_before_division[np.argmin(time_before_division)] = law_time_div()
        time_before_division = np.append(time_before_division, law_time_div())

        n_cells.append(n_cells[-1] + 1)

        if stopping_criteria == "cells":
            stopping_test = n_cells[-1] > max_cells
        else:
            stopping_test = (
                sum(times_between_div) + np.min(time_before_division) > max_time
            )

    ### Last update
    time_before_division -= max_time - sum(times_between_div)
    times_between_div = np.array(times_between_div)
    times = np.cumsum(times_between_div)
    return times, n_cells, time_before_division
