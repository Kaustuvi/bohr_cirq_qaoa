import numpy as np
from cirq import Simulator, GridQubit
from cirq_qaoa.cirq_max_cut_solver import define_grid_qubits, define_graph, solve_maxcut


def main():
        size = 2
        steps = 2
        qubits = define_grid_qubits(size=size)
        input_graph = [(qubits[0], qubits[1]), (qubits[0], qubits[2]), (qubits[0], qubits[3]),
                       (qubits[1], qubits[2]), (qubits[1], qubits[3]), (qubits[2], qubits[3])]
        solve_maxcut(graph=input_graph, steps=steps)


if __name__ == '__main__':
        main()
