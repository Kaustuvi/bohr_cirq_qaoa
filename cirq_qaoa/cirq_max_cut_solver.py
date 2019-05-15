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
    steps               :   (int) The number of mixing and cost function steps to use. Default=1
    graph               :   (list of GridQubit tuples) list of qubit pairs representing the nodes of
                            the graph on which Maxcut is to be solved
    minimizer_kwargs    :   (Optional) (dict) of optional arguments to pass to
                            the minimizer.  Default={}.
    vqe_option          :   (optional) arguents for VQE run.
    """

    def __init__(self, graph, steps=1, minimizer_kwargs=None,
                 vqe_option=None):

        self.steps = steps
        self.graph = self.create_input_graph(graph=graph)
        self.cost_operators = self.create_cost_operators()
        self.driver_operators = self.create_driver_operators()

        self.minimizer_kwargs = minimizer_kwargs or {'method': 'Nelder-Mead',
                                                     'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                                                 'disp': False}}
        self.vqe_option = vqe_option or {'disp': print_fun, 'return_all': True}

    def create_input_graph(self, graph):
        """
        create graph from input list of nodes

        Parameters
        ----------
        graph   :   list of GridQubits representing nodes of the graph to be constructed

        Returns
        -------
        a Graph() object
        """
        if not isinstance(graph, nx.Graph) and isinstance(graph, list):
            maxcut_graph = nx.Graph()
            for edge in graph:
                maxcut_graph.add_edge(*edge)
        graph = maxcut_graph.copy()
        return graph

    def create_cost_operators(self):
        """
        create family of phase separation operators that depend on the objective function to be optimized

        Returns
        -------
        list of cost clauses for the graph on which Maxcut needs to be solved
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
        create family of mixing operators that depend on the domain of the problem and its structure

        Returns
        -------
        list of mixing clauses for the graph on which Maxcut needs to be solved
        """
        driver_operators = []
        for i in self.graph.nodes():
            qubit_map_i = {i: Pauli.by_index(0)}
            driver_operators.append(CirqPauliSum(
                [PauliString(qubit_map_i, coefficient=-1.0)]))
        return driver_operators

    def solve_max_cut_qaoa(self):
        """
        initialzes a QAOA object with the required information for performing Maxcut on the input graph

        Returns
        -------
        a QAOA instance
        """
        qaoa_inst = QAOA(list(self.graph.nodes()), steps=self.steps, cost_ham=self.cost_operators,
                         ref_ham=self.driver_operators, minimizer=minimize,
                         minimizer_kwargs=self.minimizer_kwargs,
                         vqe_options=self.vqe_option)

        return qaoa_inst


def define_grid_qubits(length=2):
        """
        defines qubits on a grid of given length

        Parameters
        ----------
        length  :       length of the grid. Default=2 ,i.e, a grid containing four qubits
                        (0,0), (0,1), (1,0) and (1,1)

        Returns
        -------
        a list of qubits defined on a grid of given length
        """
        return [GridQubit(i, j) for i in range(length) for j in range(length)]


def define_graph(qubits=[(GridQubit(0, 0), GridQubit(0, 1))], number_of_vertices=2):
        """
        creates a graph as a list of qubit pairs for the given number of vertices

        Parameters
        ----------
        qubits                  :       list of GridQubits defined on a grid. Default is 
                                        one pair of qubits (0,0) and (0,1) representing a
                                        two vertex graph
        number_of_vertices      :       number of vertices the input graph must contain. Default=2

        Returns
        -------
        a list of qubit pairs representing a graph containing the given number of vertices
        """
        if len(qubits) == 1:
                return qubits

        return [(qubits[i % number_of_vertices], qubits[(i+1) % number_of_vertices]) for i in range(number_of_vertices)]


def display_maxcut_results(qaoa_instance, maxcut_result):
        """
        displays results in the form of states and corresponding probabilities from solving 
        the maxcut problem using QAOA represented by the input qaoa_instance

        Parameters
        ----------
        qaoa_instance   :       a QAOA object containing all information about the problem instance on which
                                QAOA is to be applied
        maxcut_result   :       the result obtained from solving the maxcut problem on an input graph 
        """
        print("State\tProbability")
        for state_index in range(qaoa_instance.nstates):
                print(qaoa_instance.states[state_index], "\t", np.conj(
                    maxcut_result.final_state[state_index])*maxcut_result.final_state[state_index])


def solve_maxcut(graph, steps=1):
        """
        solves the maxcut problem on the input graph

        Parameters
        ----------
        graph   :       list of qubit pairs representing the graph on which maxcut is to be solved
        steps   :       the number of mixing and cost function steps to use. Default=1 
        """
        cirqMaxCutSolver = CirqMaxCutSolver(graph=graph, steps=steps)
        qaoa_instance = cirqMaxCutSolver.solve_max_cut_qaoa()
        betas, gammas = qaoa_instance.get_angles()
        t = np.hstack((betas, gammas))
        param_circuit = qaoa_instance.get_parameterized_circuit()
        circuit = param_circuit(t)
        sim = Simulator()
        result = sim.simulate(circuit)
        display_maxcut_results(qaoa_instance, result)
