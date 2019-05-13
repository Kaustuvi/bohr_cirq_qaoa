import cirq
import numpy as np
import networkx as nx
from cirq.ops import Z
from cirq import Simulator

from cirq_qaoa.cirq_max_cut_solver import CirqMaxCutSolver

# define the length of the grid.
length = 2
# define qubits on the grid.
qubits = [cirq.GridQubit(i, j) for i in range(length) for j in range(length)]

# define the graph on which MAXCUT is to be solved
graph = [(qubits[0], qubits[1]), (qubits[1], qubits[2]), (qubits[0], qubits[2])]

cirqMaxCutSolver = CirqMaxCutSolver(graph=graph, steps=2)
qaoa_instance = cirqMaxCutSolver.solve_max_cut_qaoa()
betas, gammas = qaoa_instance.get_angles()
t = np.hstack((betas, gammas))
param_circuit = qaoa_instance.get_parameterized_circuit()
circuit = param_circuit(t)

sim = Simulator()
result = sim.simulate(circuit)

for state_index in range(qaoa_instance.nstates):
    print(qaoa_instance.states[state_index], np.conj(
        result.final_state[state_index])*result.final_state[state_index])
