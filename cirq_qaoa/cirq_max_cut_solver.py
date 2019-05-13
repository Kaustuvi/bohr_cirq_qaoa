import networkx as nx

from scipy.optimize import minimize
from cirq import PauliString, Pauli
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
