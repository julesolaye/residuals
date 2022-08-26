# Packages
import numpy as np
import scipy.linalg


class Multitype(object):
    """
    This class allows to simulate a multitype markovian branching process with
    a Gillespie algorithm. For each iteration:

    - we draw the time before the next event with a random variable following 
    an exponential distribution

    - we draw the type of event with an uniform distribution.

    We will use this class in order to have the evolution of the two terms 
    composing residuals, when the time between each division is modelised 
    by a gamma ditribution with an integer parameter k (in this case, the 
    Bellman-Harris process is a multitype markovian branching process).

    Attributes
    ------------
    n_types: int
        Types number in our markovian multitype branching process (parameter k 
        of gamma distribution).

    rate_event : float
        Rate between each event (parameter 1/theta of gamma distribution).
    """

    def __init__(
        self,
        n_types: int = 80,
        rate_event: float = 3,
        time_start: float = 0,
        time_end: float = 370,
        time_step: float = 0.1,
    ):

        self.n_types = n_types
        self.rate_event = rate_event
        self.time_start = time_start
        self.time_end = time_end
        self.time_step = time_step
        self.compute_mean_matrix_and_eig()

    def simulate(self) -> ("np.array", "np.array"):
        """
        This function allows to simulate the process with a gillespie algorithm. 
        The parameters have already been given in the initialization.

        Returns
        ------------
        times : "np.array" 
            Times at which we simulated the number of cells.

        n_cells : "np.array"
            Number of cells for each time.
        """
        ## Arrays we will return
        n_points = int((self.time_end - self.time_start) / self.time_step) + 1
        times = np.linspace(self.time_start, self.time_end, n_points, endpoint=True)

        n_cells = np.zeros((self.n_types, len(times)))
        n_cells[0, :] = 1

        time = 0
        index_time_event = 0

        while time < self.time_end:

            time_next_event = np.random.exponential(
                scale=1 / (self.rate_event * np.sum(n_cells[:, index_time_event]))
            )  # We draw an exponential random variable for the next division because minimum of several exponential random variable

            if (
                time + time_next_event > self.time_end
            ):  # No update because the event is too late
                time = self.time_end

            else:

                # We compute which event will occurs with propensities scores
                cumulate_propensities = np.cumsum(
                    n_cells[:, index_time_event] / np.sum(n_cells[:, index_time_event])
                )
                which_division_occurs = np.random.uniform()

                # Update time and time index
                time = time + time_next_event
                index_time_event = np.where(times - time >= 0)[0][0]

                index_type_division = np.where(
                    cumulate_propensities - which_division_occurs > 0
                )[0][
                    0
                ]  # First index such that cumulate propensity > 0

                if index_type_division == self.n_types - 1:
                    n_cells[0, index_time_event:] += 2
                    n_cells[self.n_types - 1, index_time_event:] -= 1

                else:
                    n_cells[index_type_division, index_time_event:] -= 1
                    n_cells[index_type_division + 1, index_time_event:] += 1

        return times, n_cells

    def simulate_residuals(
        self,
        time_step_residuals: float,
        max_time: float = 325,
        with_separate_term: bool = True,
    ) -> ("np.array", "np.array", "np.array", "np.array"):
        """
        This function allows to get renormalizer residuals for one simulation, for 
        a given time step, since a given time step. The particularity of this function
        is that it will get residuals, by computing value of each term composing
        residuals. It is useful because we can get the values of each of these 
        terms versus time.

        Parameters
        ------------
        time_step_residuals : float
            Be careful not to confuse it with the time step of the simulation.
            It is the time step we will use in order to compute residuals.

        max_time : float 
            Maximal time for which we want to have residuals.

        with_separate_term: bool
            If True, we will also return the two terms composing residuals.

        Returns
        ------------
        times_to_return : "np.array"
            Array with times where we compute residuals.
            Tableau contenant les temps pour lesquels nous avons calculé la valeur 
            des résidus.
        
        residuals : "np.array"
            Array with all of the residuals we computed.
        
        first_term : "np.array"
            Array with the first term composing the residuals we computed.

        second_term : "np.array"
            Array with the second term composing the residuals we computed.   
        """

        # Index linked to these times
        index_max_time = int(max_time / self.time_step)
        index_time_step_residuals = int(time_step_residuals / self.time_step)

        # Simulation for which we will compute residuals
        times, n_cells = self.simulate()

        # Mean matrix for this time step
        estim_means = scipy.linalg.expm(
            self.infinitesimal_generator * time_step_residuals
        )

        # Compute terms
        first_term = (
            np.sum(
                n_cells[
                    :,
                    index_time_step_residuals : index_max_time
                    + index_time_step_residuals,
                ],
                axis=0,
            )
            - np.sum(estim_means, axis=1) @ n_cells[:, :index_max_time]
        )
        second_term = np.sum(estim_means, axis=1) @ n_cells[
            :, :index_max_time
        ] - np.exp(self.malthus * time_step_residuals) * np.sum(
            n_cells[:, :index_max_time], axis=0
        )

        # Appropriate renormalization according to the spectral gap
        if self.second_eigenval > self.malthus / 2:
            first_term = first_term / np.exp(
                self.second_eigenval * times[:index_max_time]
            )
            second_term = second_term / np.exp(
                self.second_eigenval * times[:index_max_time]
            )

        else:
            first_term = first_term / np.sqrt(n_cells[:, :index_max_time])
            second_term = second_term / np.sqrt(n_cells[:, :index_max_time])

        # Times we will return
        times_to_return = times[:index_max_time]

        # Residuals
        residuals = first_term + second_term

        # Return according to the value of boolean
        if with_separate_term:
            return times_to_return, residuals, first_term, second_term
        else:
            return times_to_return, residuals

    def compute_mean_matrix_and_eig(self):
        """
        This function allow to compute:
        
        - the infinitesimal generator of mean matrix
        linked to the multitype branching process we want to simulate,
        
        - the first eigenvalue of this generator, and the real part of the second.

        This function will be useful when we will split the two terms of residuals,
        because we need the mean matrix for that (which is computed with infinitesimal
        generator).
        """
        ## Real part of the two first eigenvalues
        self.malthus = (2 ** (1 / self.n_types) - 1) * self.rate_event
        self.second_eigenval = (
            (2 ** (1 / self.n_types)) * np.cos(2 * np.pi / self.n_types) - 1
        ) * self.rate_event

        ## Infinitesimal generator
        self.infinitesimal_generator = -np.eye(self.n_types) + np.diag(
            np.ones(self.n_types - 1), 1
        )
        self.infinitesimal_generator[-1, 0] = 2
        self.infinitesimal_generator *= self.rate_event

