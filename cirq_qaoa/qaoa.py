import numpy as np

from scipy import optimize
from copy import deepcopy
from cirq import Circuit, InsertStrategy, H
from .pauli_operations import exponentiate_pauli_string
from .vqe import VQE


class QAOA:
    """
    Quantum Approximate Optimization Algoirthm (QAOA)

    Contains all information for running the QAOA algorthm to find the
    ground state of the list of cost clauses.

    Parameters
    ----------

    qubits              :   (list of GridQubits) number of qubits to use for the algorithm.
    steps               :   (int) number of mixing and cost function steps to use.
                            Default=1.
    cost_ham            :   (list) clauses in the cost function. Must be
                            CirqPauliSum objects
    ref_ham             :   (list) clauses in the mixer function. Must be
                            CirqPauliSum objects
    driver_ref          :   (Circuit object) defines the starting state of the QAOA algorithm.
                            Defaults to tensor product of |+> states.
    minimizer           :   (optional) minimization function to pass to the
                            Variational-Quantum-Eigensolver method
    minimizer_kwargs    :   (optional) (dict) optional arguments to pass to
                            the minimizer.  Default={}.
    vqe_options         :   (optional) arguents for VQE run.

    References
    ----------
    Farhi, Edward & Goldstone, Jeffrey & Gutmann, Sam. (2014).
    A Quantum Approximate Optimization Algorithm.
    <https://arxiv.org/abs/1411.4028>
    """

    def __init__(self, qubits, steps=1, cost_ham=None, ref_ham=None,
                 driver_ref=None, minimizer=None, minimizer_kwargs=None,
                 vqe_options=None):
        self.steps = steps
        self.qubits = qubits
        self.number_of_states = 2 ** len(qubits)
        self.cost_ham = cost_ham or []
        self.ref_ham = ref_ham or []

        self.minimizer = minimizer or optimize.minimize
        self.minimizer_kwargs = minimizer_kwargs or {
            'method': 'Nelder-Mead',
            'options': {
                'disp': True,
                'ftol': 1.0e-2,
                'xtol': 1.0e-2
            }
        }

        self.betas = np.random.uniform(0, np.pi, self.steps)[::-1]
        self.gammas = np.random.uniform(0, 2*np.pi, self.steps)
        self.vqe_options = vqe_options or {}

        self.initial_state = self.create_initial_state(driver_ref)

        self.states = [np.binary_repr(i, width=len(self.qubits))
                       for i in range(self.number_of_states)]

    def create_initial_state(self, driver_ref):
        """
        Generates a circuit representing the initial state of the QAOA algorithm
        on which the cost and driver operators are alternatively applied

        Parameters
        ----------
        driver_ref          :   (Circuit object) circuit representing the initial state. If driver_ref is None,
                                the function returns state representing equal superposition over
                                all qubits

        Returns
        -------
        circuit             :   (Circuit object) represents the initial state for the QAOA algorithm
        """
        if driver_ref is not None:
            return driver_ref

        circuit = Circuit()
        circuit.append([H(i) for i in self.qubits],
                       strategy=InsertStrategy.EARLIEST)
        return circuit

    def get_parameterized_circuit(self):
        """
        Returns a function that accepts parameters and returns a new cirq Circuit.

        Returns
        -------
        parameterized_circuit() :   (function)  a function that constructs a parameterized circuit 
                                    with the input parameters. With given parameters, the function returns
                                    a circuit consisting of gates which take the paramters as rotation angles

        """
        cost_parametric_circuits = []
        driver_parametric_circuits = []
        iteration = 0
        while iteration < self.steps:
            cost_list = []
            driver_list = []
            for cost_pauli_sum in self.cost_ham:
                for cost_pauli_string in cost_pauli_sum.pauli_strings:
                    cost_list.append(
                        exponentiate_pauli_string(cost_pauli_string))

            for driver_pauli_sum in self.ref_ham:
                for driver_pauli_string in driver_pauli_sum.pauli_strings:
                    driver_list.append(
                        exponentiate_pauli_string(driver_pauli_string))
            iteration = iteration + 1

            cost_parametric_circuits.append(cost_list)
            driver_parametric_circuits.append(driver_list)

        def parameterized_circuit(params):
            """
            Constructs a Circuit for the array `params` containing angles (beta, gamma) for the optimal solution.

            Parameters
            ----------
            params                  :   (ndarray) contains 2*steps angles, betas first, then gammas

            Returns
            -------
            parameterized_circuit   :   (Circuit object) parameterized Circuit object for 
                                        the input parameters `params`

            """
            if len(params) != 2*self.steps:
                raise ValueError(
                    "params doesn't match the number of parameters set by `steps`")
            betas = params[:self.steps]
            gammas = params[self.steps:]
            circuit = Circuit()
            circuit += self.initial_state
            for i in range(self.steps):
                for cost_circuit in cost_parametric_circuits[i]:
                    circuit += cost_circuit(gammas[i])

                for driver_circuit in driver_parametric_circuits[i]:
                    circuit += driver_circuit(betas[i])
            return circuit

        return parameterized_circuit

    def get_angles(self):
        """
        Finds optimal betas and gammas with the variational quantum eigensolver method.

        Returns
        -------
        betas, gammas   :   (list, list) tuple of the beta angles and the gamma
                            angles for the optimal solution.

        """
        stacked_params = np.hstack((self.betas, self.gammas))
        vqe = VQE(self.minimizer, minimizer_kwargs=self.minimizer_kwargs)

        temp_pauli_sum = deepcopy(self.cost_ham[0])
        for cirq_pauli_sum in self.cost_ham[1:]:
            temp_pauli_sum += cirq_pauli_sum
        cost_ham = temp_pauli_sum

        parameterized_circuit = self.get_parameterized_circuit()
        result = vqe.vqe_run(self.qubits, parameterized_circuit, cost_ham, stacked_params,
                             **self.vqe_options)
        self.result = result
        betas = result.x[:self.steps]
        gammas = result.x[self.steps:]
        return betas, gammas
