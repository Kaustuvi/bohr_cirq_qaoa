import networkx as nx
from scipy.optimize import minimize

from .paulis import PauliTerm, PauliSum
from .qaoa import QAOA


def print_fun(x):
    print(x)


class CirqMaxCutSolver:
    def __init__(self, graph, steps=1, rand_seed=None, samples=None,
                 initial_beta=None, initial_gamma=None, minimizer_kwargs=None,
                 vqe_option=None):

        self.steps = steps
        self.graph = self.create_input_graph(graph=graph)
        self.cost_operators = self.create_cost_operators()
        self.driver_operators = self.create_driver_operators()
        self.initial_beta = initial_beta
        self.initial_gamma = initial_gamma
        self.rand_seed = rand_seed

        if minimizer_kwargs is None:
            self.minimizer_kwargs = {'method': 'Nelder-Mead',
                                     'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                                 'disp': False}}
        if vqe_option is None:
            self.vqe_option = {'disp': print_fun, 'return_all': True,
                               'samples': samples}

    def create_input_graph(self, graph):
        if not isinstance(graph, nx.Graph) and isinstance(graph, list):
            maxcut_graph = nx.Graph()
            for edge in graph:
                maxcut_graph.add_edge(*edge)
        graph = maxcut_graph.copy()
        return graph

    def create_cost_operators(self):
        cost_operators = []
        for i, j in self.graph.edges():
            cost_operators.append(PauliTerm("Z", i, 0.5)
                                  * PauliTerm("Z", j)+PauliTerm("I", 0, -0.5))

        return cost_operators

    def create_driver_operators(self):
        driver_operators = []
        for i in self.graph.nodes():
            driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))

        return driver_operators

    def solve_max_cut_qaoa(self):
        qaoa_inst = QAOA(list(self.graph.nodes()), steps=self.steps, cost_ham=self.cost_operators,
                         ref_ham=self.driver_operators, store_basis=True,
                         rand_seed=self.rand_seed,
                         init_betas=self.initial_beta,
                         init_gammas=self.initial_gamma,
                         minimizer=minimize,
                         minimizer_kwargs=self.minimizer_kwargs,
                         vqe_options=self.vqe_option)
                         
        return qaoa_inst
