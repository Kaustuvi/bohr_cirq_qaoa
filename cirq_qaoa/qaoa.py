from collections import Counter
from scipy import optimize
import numpy as np
from CirQ import Circuit, InsertStrategy, H
from .paulis import PauliSum, exponential_map
from .vqe import VQE
from functools import reduce


class QAOA(object):
    def __init__(self, qubits, steps=1, init_betas=None,
                 init_gammas=None, cost_ham=None,
                 ref_ham=None, driver_ref=None,
                 minimizer=None, minimizer_args=None,
                 minimizer_kwargs=None, rand_seed=None,
                 vqe_options=None, store_basis=False):
        """
        QAOA object.

        Contains all information for running the QAOA algorthm to find the
        ground state of the list of cost clauses.

        N.B. This only works if all the terms in the cost Hamiltonian commute with each other.

        :param qubits: (list of ints) The number of qubits to use for the algorithm.
        :param steps: (int) The number of mixing and cost function steps to use.
                      Default=1.
        :param init_betas: (list) Initial values for the beta parameters on the
                           mixing terms. Default=None.
        :param init_gammas: (list) Initial values for the gamma parameters on the
                            cost function. Default=None.
        :param cost_ham: list of clauses in the cost function. Must be
                    PauliSum objects
        :param ref_ham: list of clauses in the mixer function. Must be
                    PauliSum objects
        :param driver_ref: (CirQ.Circuit()) object to define state prep
                           for the starting state of the QAOA algorithm.
                           Defaults to tensor product of |+> states.
        :param rand_seed: integer random seed for initial betas and gammas
                          guess.
        :param minimizer: (Optional) Minimization function to pass to the
                          Variational-Quantum-Eigensolver method
        :param minimizer_kwargs: (Optional) (dict) of optional arguments to pass to
                                 the minimizer.  Default={}.
        :param minimizer_args: (Optional) (list) of additional arguments to pass to the
                               minimizer. Default=[].
        :param minimizer_args: (Optional) (list) of additional arguments to pass to the
                               minimizer. Default=[].
        :param vqe_options: (optinal) arguents for VQE run.
        :param store_basis: (optional) boolean flag for storing basis states.
                            Default=False.
        """

        # Seed the random number generator, if a seed is provided.
        if rand_seed is not None:
            np.random.seed(rand_seed)

        # Set attributes values, considering their defaults
        self.steps = steps
        self.qubits = qubits
        self.nstates = 2 ** len(qubits)

        self.cost_ham = cost_ham or []
        self.ref_ham = ref_ham or []

        self.minimizer = minimizer or optimize.minimize
        self.minimizer_args = minimizer_args or []
        self.minimizer_kwargs = minimizer_kwargs or {
            'method': 'Nelder-Mead',
            'options': {
                'disp': True,
                'ftol': 1.0e-2,
                'xtol': 1.0e-2
            }
        }

        self.betas = init_betas or np.random.uniform(
            0, np.pi, self.steps)[::-1]
        self.gammas = init_gammas or np.random.uniform(0, 2*np.pi, self.steps)
        self.vqe_options = vqe_options or {}

        self.ref_state_prep = self.create_initial_state(driver_ref)

        if store_basis:
            self.states = [
                np.binary_repr(i, width=len(self.qubits))
                for i in range(self.nstates)
            ]

        # Check argument types
        if not isinstance(self.cost_ham, (list, tuple)):
            raise TypeError("cost_ham must be a list of PauliSum objects.")
        if not all([isinstance(x, PauliSum) for x in self.cost_ham]):
            raise TypeError("cost_ham must be a list of PauliSum objects")

        if not isinstance(self.ref_ham, (list, tuple)):
            raise TypeError("ref_ham must be a list of PauliSum objects")
        if not all([isinstance(x, PauliSum) for x in self.ref_ham]):
            raise TypeError("ref_ham must be a list of PauliSum objects")

        if not isinstance(self.ref_state_prep, Circuit):
            raise TypeError("Please provide a CirQ Circuit object "
                            "to generate initial state.")

    def create_initial_state(self, driver_ref):
        if driver_ref is not None:
            return driver_ref

        circuit = Circuit()
        circuit.append([H(i) for i in self.qubits],
                       strategy=InsertStrategy.EARLIEST)
        return circuit

    def get_parameterized_circuit(self):
        """
        Return a function that accepts parameters and returns a new CirQ Circuit.

        :returns: a function
        """
        cost_para_circuits = []
        driver_para_circuits = []
        for idx in range(self.steps):
            cost_list = []
            driver_list = []
            for cost_pauli_sum in self.cost_ham:
                for term in cost_pauli_sum.terms:
                    cost_list.append(exponential_map(term))

            for driver_pauli_sum in self.ref_ham:
                for term in driver_pauli_sum.terms:
                    driver_list.append(exponential_map(term))

            cost_para_circuits.append(cost_list)
            driver_para_circuits.append(driver_list)

        def psi_ref(params):
            """
            Construct a CirQ Circuit for the vector (beta, gamma).

            :param params: array of 2 . p angles, betas first, then gammas
            :return: a CirQ Circuit object
            """
            if len(params) != 2*self.steps:
                raise ValueError(
                    "params doesn't match the number of parameters set by `steps`")
            betas = params[:self.steps]
            gammas = params[self.steps:]

            circuit = Circuit()
            circuit += self.ref_state_prep
            for idx in range(self.steps):
                for fcircuit in cost_para_circuits[idx]:
                    circuit += fcircuit(gammas[idx])

                for fcircuit in driver_para_circuits[idx]:
                    circuit += fcircuit(betas[idx])

            return circuit

        return psi_ref

    def get_angles(self):
        """
        Finds optimal angles with the quantum variational eigensolver method.

        Stored VQE result

        :returns: ([list], [list]) A tuple of the beta angles and the gamma
                  angles for the optimal solution.
        """
        stacked_params = np.hstack((self.betas, self.gammas))
        vqe = VQE(self.minimizer, minimizer_args=self.minimizer_args,
                  minimizer_kwargs=self.minimizer_kwargs)
        cost_ham = reduce(lambda x, y: x + y, self.cost_ham)
        # maximizing the cost function!
        param_prog = self.get_parameterized_circuit()
        result = vqe.vqe_run(self.qubits, param_prog, cost_ham, stacked_params,
                             **self.vqe_options)
        print(cost_ham)
        self.result = result
        betas = result.x[:self.steps]
        gammas = result.x[self.steps:]
        return betas, gammas
