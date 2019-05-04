from .paulis import PauliSum, PauliTerm
from .gates import STANDARD_GATES

from cirq import Circuit,  GridQubit, InsertStrategy, Simulator, unitary, I
import numpy as np
from collections import Counter
import funcsigs


class OptResults(dict):
    """
    Object for holding optimization results from VQE.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class VQE(object):
    """
    The Variational-Quantum-Eigensolver algorithm

    VQE is an object that encapsulates the VQE algorithm (functional
    minimization). The main components of the VQE algorithm are a minimizer
    function for performing the functional minimization, a function that takes a
    vector of parameters and returns a CirQ circuit, and a
    Hamiltonian of which to calculate the expectation value.

    Using this object:

        1) initialize with `inst = VQE(minimizer)` where `minimizer` is a
        function that performs a gradient free minization--i.e
        scipy.optimize.minimize(. , ., method='Nelder-Mead')

        2) call `inst.vqe_run(variational_state_evolve, hamiltonian,
        initial_parameters)`. Returns the optimal parameters and minimum
        expecation

    :param minimizer: function that minimizes objective f(obj, param). For
                      example the function scipy.optimize.minimize() needs
                      at least two parameters, the objective and an initial
                      point for the optimization.  The args for minimizer
                      are the cost function (provided by this class),
                      initial parameters (passed to vqe_run() method, and
                      jacobian (defaulted to None).  kwargs can be passed
                      in below.
    :param minimizer_args: (list) arguments for minimizer function. Default=None
    :param minimizer_kwargs: (dict) arguments for keyword args.
                              Default=None

    """

    def __init__(self, minimizer, minimizer_args=[], minimizer_kwargs={}):
        self.minimizer = minimizer
        self.minimizer_args = minimizer_args
        self.minimizer_kwargs = minimizer_kwargs
        self.n_qubits = None

    def vqe_run(self, all_qubits_in_circuit, variational_state_evolve, hamiltonian, initial_params,
                gate_noise=None, measurement_noise=None,
                jacobian=None, disp=None, samples=None, return_all=False):
        """
        functional minimization loop.

        :param variational_state_evolve: function that takes a set of parameters
                                        and returns a cirQ Circuit.
        :param hamiltonian: (PauliSum) object representing the hamiltonian of
                            which to take the expectation value.
        :param initial_params: (ndarray) vector of initial parameters for the
                               optimization
        :param gate_noise: list of Px, Py, Pz probabilities of gate being
                           applied to every gate after each get application
        :param measurement_noise: list of Px', Py', Pz' probabilities of a X, Y
                                  or Z being applied before a measurement.
        :param jacobian: (optional) method of generating jacobian for parameters
                         (Default=None).
        :param disp: (optional, bool) display level. If True then each iteration
                     expectation and parameters are printed at each
                     optimization iteration.
        :param samples: (int) Number of samples for calculating the expectation
                        value of the operators.  If `None` then faster method
                        ,dotting the wave function with the operator, is used.
                        Default=None.
        :param return_all: (optional, bool) request to return all intermediate
                           parameters determined during the optimization.
        :return: (vqe.OptResult()) object :func:`OptResult <vqe.OptResult>`.
                 The following fields are initialized in OptResult:
                 -x: set of w.f. ansatz parameters
                 -fun: scalar value of the objective function

                 -iteration_params: a list of all intermediate parameter vectors. Only
                                    returned if 'return_all=True' is set as a vqe_run()
                                    option.

                 -expectation_vals: a list of all intermediate expectation values. Only
                                    returned if 'return_all=True' is set as a
                                    vqe_run() option.
        """
        self._disp_fun = disp if disp is not None else lambda x: None
        iteration_params = []
        expectation_vals = []
        self._current_expectation = None
        if samples is None:
            print("""WARNING: Fast method for expectation will be used. Noise
                     models will be ineffective""")

        def objective_function(params):
            """
            closure representing the functional

            :param params: (ndarray) vector of parameters for generating the
                           the function of the functional.
            :return: (float) expectation value
            """
            cirq_circuit = variational_state_evolve(params)
            mean_value = self.expectation(
                all_qubits_in_circuit, cirq_circuit, hamiltonian, samples)
            self._current_expectation = mean_value  # store for printing
            return mean_value

        def print_current_iter(iter_vars):
            self._disp_fun("\tParameters: {} ".format(iter_vars))
            if jacobian is not None:
                grad = jacobian(iter_vars)
                self._disp_fun(
                    "\tGrad-L1-Norm: {}".format(np.max(np.abs(grad))))
                self._disp_fun(
                    "\tGrad-L2-Norm: {} ".format(np.linalg.norm(grad)))

            self._disp_fun("\tE => {}".format(self._current_expectation))
            if return_all:
                iteration_params.append(iter_vars)
                expectation_vals.append(self._current_expectation)

        # using self.minimizer
        arguments = funcsigs.signature(self.minimizer).parameters.keys()

        if disp is not None and 'callback' in arguments:
            self.minimizer_kwargs['callback'] = print_current_iter

        args = [objective_function, initial_params]
        args.extend(self.minimizer_args)
        if 'jac' in arguments:
            self.minimizer_kwargs['jac'] = jacobian

        result = self.minimizer(*args, **self.minimizer_kwargs)
        if hasattr(result, 'status'):
            if result.status != 0:
                self._disp_fun("Classical optimization exited with an error index: %i"
                               % result.status)

        results = OptResults()
        if hasattr(result, 'x'):
            results.x = result.x
            results.fun = result.fun
        else:
            results.x = result

        if return_all:
            results.iteration_params = iteration_params
            results.expectation_vals = expectation_vals
        return results

    @staticmethod
    def expectation(all_qubits_in_circuit, cirq_circuit, pauli_sum, samples):
        """
        Computes the expectation value of pauli_sum over the distribution
        generated from cirq_circuit.

        :param all_qubits_in_circuit: (list) list of all qubits in cirq_circuit
        :param cirq_circuit: (CirQ circuit)
        :param pauli_sum: (PauliSum, ndarray) PauliSum representing the
                          operator of which to calculate the expectation value
                          or a numpy matrix representing the Hamiltonian
                          tensored up to the appropriate size.
        :param samples: (int) number of samples used to calculate the
                        expectation value.  If samples is None then the expectation
                        value is calculated by calculating <psi|O|psi> .

        :returns: (float) representing the expectation value of pauli_sum given
                  given the distribution generated from cirq_circuit.
        """
        if isinstance(pauli_sum, np.ndarray):
            # debug mode by passing an array
            simulator = Simulator()
            simulation_result = simulator.simulate(cirq_circuit)
            cirq_circuit_amplitudes = simulation_result.final_state
            cirq_circuit_amplitudes = np.reshape(
                cirq_circuit_amplitudes, (-1, 1))
            average_exp = np.conj(cirq_circuit_amplitudes).T.dot(
                pauli_sum.dot(cirq_circuit_amplitudes)).real
            return average_exp
        else:
            if not isinstance(pauli_sum, (PauliTerm, PauliSum)):
                raise TypeError("pauli_sum variable must be a PauliTerm or"
                                "PauliSum object")

            if isinstance(pauli_sum, PauliTerm):
                pauli_sum = PauliSum([pauli_sum])

            if samples is None:
                operator_circuits = []
                operator_coeffs = []
                for p_term in pauli_sum.terms:
                    op_circuit = Circuit()
                    for qindex, op in p_term:
                        gate = STANDARD_GATES[op]
                        op_circuit.append(
                            [gate(qindex)], strategy=InsertStrategy.EARLIEST)
                    operator_circuits.append(op_circuit)
                    operator_coeffs.append(p_term.coefficient)
                result_overlaps = VQE.calculate_expectation(
                    all_qubits_in_circuit, cirq_circuit, operator_circuits)
                result_overlaps = list(result_overlaps)
                assert len(result_overlaps) == len(operator_circuits), """Somehow we
                didn't get the correct number of results"""
                expectation = sum(
                    list(map(lambda x: x[0]*x[1], zip(result_overlaps, operator_coeffs))))
                return expectation.real

    @staticmethod
    def calculate_expectation(all_qubits_in_circuit, cirq_circuit, operator_circuits):
        simulator = Simulator()
        result_overlaps = []
        simulation_result = simulator.simulate(cirq_circuit)
        cirq_circuit_amplitudes = simulation_result.final_state
        for op_circuit in operator_circuits:
            qubits = all_qubits_in_circuit[:]
            if(len(op_circuit._moments) == 0):
                result_overlaps.append(
                    np.conj(cirq_circuit_amplitudes).T.dot(cirq_circuit_amplitudes))
            else:
                if(len(all_qubits_in_circuit) != len(op_circuit.all_qubits())):
                    op_circuit = VQE.modify_operator_circuit(
                        qubits, op_circuit)
                unitary_matrix_op = unitary(op_circuit)
                result_overlaps.append(np.conj(cirq_circuit_amplitudes).T.dot(
                    unitary_matrix_op).dot(cirq_circuit_amplitudes))
        return result_overlaps

    @staticmethod
    def modify_operator_circuit(all_qubits_in_circuit, operator_circuit):
        for qubits_in_circuit in operator_circuit.all_qubits():
                all_qubits_in_circuit.remove(qubits_in_circuit)
        for qubit in all_qubits_in_circuit:
            operator_circuit.append(
                [I(qubit)], strategy=InsertStrategy.EARLIEST)
        return operator_circuit
