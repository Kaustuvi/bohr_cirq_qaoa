import numpy as np

from .pauli_operations import CirqPauliSum
from cirq import Circuit, InsertStrategy, Simulator, unitary, I


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


class VQE:
    """
    The Variational-Quantum-Eigensolver algorithm

    VQE is an object that encapsulates the VQE algorithm (functional
    minimization). The main components of the VQE algorithm are a minimizer
    function for performing the functional minimization, a function that takes a
    vector of parameters and returns a Cirq circuit, and a
    Hamiltonian of which to calculate the expectation value.

    Parameters
    ----------
    minimizer           :   (function) a function that minimizes objective f(obj, param). For
                            example the function scipy.optimize.minimize() needs
                            at least two parameters, the objective and an initial
                            point for the optimization.  The args for minimizer
                            are the cost function (provided by this class),
                            initial parameters (passed to vqe_run() method
    minimizer_kwargs    :   (dict) arguments for keyword args. Default=None

    """

    def __init__(self, minimizer, minimizer_kwargs={}):
        self.minimizer = minimizer
        self.minimizer_kwargs = minimizer_kwargs

    def vqe_run(self, all_qubits_in_circuit, variational_state_evolve, hamiltonian, initial_params,
                disp=None, return_all=False):
        """
        Runs the VQE algorithm that minimizes the objective function for the cost hamiltonian 
        of a parameterized circuit represented by `variational_state_evolve`

        Parameters
        ----------
        variational_state_evolve    :   function that takes a set of parameters 
                                        and returns a Circuit object.
        hamiltonian                 :   (CirqPauliSum) object representing the hamiltonian 
                                        of which to take the expectation value.
        initial_params              :   (ndarray) vector of initial parameters for the 
                                        optimization
        disp                        :   (optional, bool) display level. If True then each iteration 
                                        expectation and parameters are printed at each 
                                        optimization iteration.
        return_all                  :   (optional, bool) request to return all intermediate
                                        parameters determined during the optimization.

        Returns
        -------
        results                     :   (vqe.OptResult object) The following fields are initialized in OptResult:
                                        -x: set of betas and gammas (parameters for the ansatz)
                                        -fun: scalar value of the objective function
                                        -iteration_params   :   a list of all intermediate parameter vectors. Only
                                                                returned if 'return_all=True' is set as a vqe_run()
                                                                option.
                                        -expectation_vals:      a list of all intermediate expectation values. Only
                                                                returned if 'return_all=True' is set as a
                                                                vqe_run() option.
        """
        self._disp_fun = disp
        iteration_params = []
        expectation_vals = []
        self._current_expectation = None

        def objective_function(params):
            """
            Generates a parameterized circuit with the given parameters `params` and calculates
            the expectation value of the cost hamiltonian for the generated circuit

            Parameters
            ----------
            params      :   (ndarray) vector of parameters for generating the parametrized circuit.

            Return
            ------
            mean_value  :   (float) expectation value of the cost hamiltonian for the generated circuit
            """
            cirq_circuit = variational_state_evolve(params)
            mean_value = self.expectation(
                all_qubits_in_circuit, cirq_circuit, hamiltonian)
            self._current_expectation = mean_value
            return mean_value

        def print_current_iter(iter_vars):
            self._disp_fun("\tParameters: {} ".format(iter_vars))
            self._disp_fun("\tE => {}".format(self._current_expectation))
            if return_all:
                iteration_params.append(iter_vars)
                expectation_vals.append(self._current_expectation)

        self.minimizer_kwargs['callback'] = print_current_iter

        args = [objective_function, initial_params]

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
    def expectation(all_qubits_in_circuit, cirq_circuit, cirq_pauli_sum):
        """
        Computes the expectation value of `cirq_pauli_sum` over the distribution
        generated from `cirq_circuit`.
        The expectation value is calculated by calculating <psi|O|psi>

        Parameters
        ---------
        all_qubits_in_circuit   :   (list) list of all qubits in `cirq_circuit`
        cirq_circuit            :   (Circuit) represents the paramterized circuit for the ansatz 
        cirq_pauli_sum          :   (CirqPauliSum or ndarray) CirqPauliSum representing the Pauli
                                    operator for which the expectation value is to be calculated
                                    or a numpy matrix representing the Hamiltonian 
                                    tensored up to the appropriate size.

        Returns
        -------
        expectation.real        :   (float) represents the expectation value of cirq_pauli_sum 
                                    given the distribution generated from `cirq_circuit`.
        """
        if isinstance(cirq_pauli_sum, np.ndarray):
            simulator = Simulator()
            simulation_result = simulator.simulate(cirq_circuit)
            cirq_circuit_amplitudes = simulation_result.final_state
            cirq_circuit_amplitudes = np.reshape(
                cirq_circuit_amplitudes, (-1, 1))
            average_exp = np.conj(cirq_circuit_amplitudes).T.dot(
                cirq_pauli_sum.dot(cirq_circuit_amplitudes)).real
            return average_exp
        else:
            operator_circuits = []
            operator_coeffs = []
            for p_string in cirq_pauli_sum.pauli_strings:
                op_circuit = Circuit()
                for qubit, pauli in p_string.items():
                    gate = pauli
                    op_circuit.append(
                        [gate(qubit)], strategy=InsertStrategy.EARLIEST)
                operator_circuits.append(op_circuit)
                operator_coeffs.append(p_string.coefficient)
            result_overlaps = VQE.calculate_expectation(
                all_qubits_in_circuit, cirq_circuit, operator_circuits)
            result_overlaps = list(result_overlaps)
            expectation = sum(
                list(map(lambda x: x[0]*x[1], zip(result_overlaps, operator_coeffs))))
            return expectation.real

    @staticmethod
    def calculate_expectation(all_qubits_in_circuit, cirq_circuit, operator_circuits):
        """
        Computes the expectation value according to the formula <psi|O|psi> where O is a Pauli operator
        represented by the state generated from `operator_circuits` for which the expectation is to 
        be calculated over psi where psi is the state generated from `cirq_circuit`

        Parameters
        ---------
        all_qubits_in_circuit   :   (list) list of all qubits in `cirq_circuit`
        cirq_circuit            :   (Circuit) represents the paramterized circuit for the ansatz 
        operator_circuits       :   (list of Circuit objects) represents a circuit generated by 
                                    applying a Pauli operator on a qubit.

        Returns
        -------
        result_overlaps         :   (list) represents the expectation values of each Pauli operator in
                                    `operator_circuits` given the state represented by `cirq_circuit`.
        """
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
        """
        Modifies the `operator_circuit` by appending the Identity operator for qubits on which 
        no Pauli operator is applied so that the state represented by `operator_circuit` consists of
        all qubits used in the algorithm. This is done to ensure that the calculation of expectation for the
        Pauli operators does not run into unmatched dimension errors.

        Parameters
        ---------
        all_qubits_in_circuit   :   (list) list of all qubits used in the algorithm
        operator_circuits       :   (list of Circuit objects) represents a circuit generated by 
                                    applying a Pauli operator on a qubit.

        Returns
        -------
        operator_circuit        :   (Circuit object) represents the modified circuit with the Identity operator
                                    appended for qubits that are not present in `operator_circuit`.
        """
        for qubits_in_circuit in operator_circuit.all_qubits():
                all_qubits_in_circuit.remove(qubits_in_circuit)
        for qubit in all_qubits_in_circuit:
            operator_circuit.append(
                [I(qubit)], strategy=InsertStrategy.EARLIEST)
        return operator_circuit
