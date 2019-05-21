import networkx as nx
import numpy as np

from scipy.optimize import minimize
from cirq import PauliString, Pauli, Simulator, GridQubit
from .qaoa import QAOA
from .pauli_operations import CirqPauliSum, add_pauli_strings


def print_fun(x):
    print(x)


class CirqMaxCutSolver:
    """
    CirqMaxCutSolver creates the cost operators and the mixing operators for the input graph
    and returns a QAOA object that solves the Maxcut problem for the input graph 

    Parameters
    ----------
    steps               :   (int) number of mixing and cost function steps to use. Default=1
    qubit_pairs         :   (list of GridQubit pairs) represents the edges of the graph on which Maxcut 
                            is to be solved
    minimizer_kwargs    :   (optional) (dict) arguments to pass to the minimizer.  Default={}.
    vqe_option          :   (optional) arguments for VQE run.
    """

    def __init__(self, qubit_pairs, steps=1, minimizer_kwargs=None,
                 vqe_option=None):

        self.steps = steps
        self.graph = self.create_input_graph(qubit_pairs=qubit_pairs)
        self.cost_operators = self.create_cost_operators()
        self.driver_operators = self.create_driver_operators()

        self.minimizer_kwargs = minimizer_kwargs or {'method': 'Nelder-Mead',
                                                     'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                                                 'disp': False}}
        self.vqe_option = vqe_option or {'disp': print_fun, 'return_all': True}

    def create_input_graph(self, qubit_pairs):
        """
        Creates graph from list of GridQubit pairs

        Parameters
        ----------
        qubit_pairs     :   (list of GridQubit pairs) representing edges of the graph to be constructed

        Returns
        -------
        graph           :   (Graph object) represents the graph containing edges defined in qubit_pairs
        """
        if not isinstance(qubit_pairs, nx.Graph) and isinstance(qubit_pairs, list):
            maxcut_graph = nx.Graph()
            for qubit_pair in qubit_pairs:
                maxcut_graph.add_edge(*qubit_pair)
        graph = maxcut_graph.copy()
        return graph

    def create_cost_operators(self):
        """
        Creates family of phase separation operators that depend on the objective function to be optimized

        Returns
        -------
        cost_operators  :   (list) cost clauses for the graph on which Maxcut needs to be solved
        """
        cost_operators = []
        for i, j in self.graph.edges():
            qubit_map_i = {i: Pauli.by_index(2)}
            qubit_map_j = {j: Pauli.by_index(2)}
            pauli_z_term = PauliString(
                qubit_map_i, coefficient=0.5)*PauliString(qubit_map_j)
            pauli_identity_term = PauliString(coefficient=-0.5)
            cost_pauli_sum = add_pauli_strings(
                [pauli_z_term, pauli_identity_term])
            cost_operators.append(cost_pauli_sum)
        return cost_operators

    def create_driver_operators(self):
        """
        Creates family of mixing operators that depend on the domain of the problem and its structure

        Returns
        -------
        driver_operators    :   (list) mixing clauses for the graph on which Maxcut needs to be solved
        """
        driver_operators = []
        for i in self.graph.nodes():
            qubit_map_i = {i: Pauli.by_index(0)}
            driver_operators.append(CirqPauliSum(
                [PauliString(qubit_map_i, coefficient=-1.0)]))
        return driver_operators

    def solve_max_cut_qaoa(self):
        """
        Initialzes a QAOA object with the required information for performing Maxcut on the input graph

        Returns
        -------
        qaoa_inst   :   (QAOA object) represents all information for running the QAOA algorthm to find the
                        ground state of the list of cost clauses. 
        """
        qaoa_inst = QAOA(list(self.graph.nodes()), steps=self.steps, cost_ham=self.cost_operators,
                         ref_ham=self.driver_operators, minimizer=minimize,
                         minimizer_kwargs=self.minimizer_kwargs,
                         vqe_options=self.vqe_option)

        return qaoa_inst


def define_grid_qubits(size=2):
        """
        Defines qubits on a square grid of given size

        Parameters
        ----------
        size    :       (int) size of the grid. Default=2 ,i.e, a grid containing four qubits
                        (0,0), (0,1), (1,0) and (1,1)

        Returns
        -------
        a list of GridQubits defined on a grid of given size
        """
        return [GridQubit(i, j) for i in range(size) for j in range(size)]


def define_graph(qubits=[(GridQubit(0, 0), GridQubit(0, 1))], number_of_vertices=2):
        """
        Creates a cycle graph as a list of GridQubit pairs for the given number of vertices

        Parameters
        ----------
        qubits                  :       (list of GridQubits). Default is 
                                        one pair of qubits (0,0) and (0,1) representing a
                                        two vertex graph
        number_of_vertices      :       (int) number of vertices the cycle graph must contain. 
                                        Default=2

        Returns
        -------
        a list of GridQubit pairs representing a cycle graph containing the given number of vertices
        """
        if len(qubits) == 1:
                return qubits

        return [(qubits[i % number_of_vertices], qubits[(i+1) % number_of_vertices]) for i in range(number_of_vertices)]


def display_maxcut_results(qaoa_instance, maxcut_result):
        """
        Displays results in the form of states and corresponding probabilities from solving 
        the maxcut problem using QAOA represented by the input qaoa_instance

        Parameters
        ----------
        qaoa_instance   :       (QAOA object) contains all information about the problem instance on which
                                QAOA is to be applied
        maxcut_result   :       (SimulationTrialResults object) obtained from solving the maxcut problem on an input graph 
        """
        print("State\tProbability")
        for state_index in range(qaoa_instance.number_of_states):
                print(qaoa_instance.states[state_index], "\t", np.conj(
                    maxcut_result.final_state[state_index])*maxcut_result.final_state[state_index])


def solve_maxcut(qubit_pairs, steps=1):
        """
        Solves the maxcut problem on the input graph

        Parameters
        ----------
        qubit_pairs :       (list of GridQubit pairs) represents the graph on which maxcut is to be solved
        steps       :       (int) number of mixing and cost function steps to use. Default=1 
        """
        cirqMaxCutSolver = CirqMaxCutSolver(
            qubit_pairs=qubit_pairs, steps=steps)
        qaoa_instance = cirqMaxCutSolver.solve_max_cut_qaoa()
        betas, gammas = qaoa_instance.get_angles()
        t = np.hstack((betas, gammas))
        param_circuit = qaoa_instance.get_parameterized_circuit()
        circuit = param_circuit(t)
        sim = Simulator()
        result = sim.simulate(circuit)
        display_maxcut_results(qaoa_instance, result)
