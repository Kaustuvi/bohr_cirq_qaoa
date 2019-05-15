import numpy as np
from cirq import Simulator, GridQubit
from cirq_qaoa.cirq_max_cut_solver import define_grid_qubits, create_input_graph, solve_maxcut


def main():
        length = 2
        number_of_vertices = 3
        steps = 2
        qubits = define_grid_qubits(length=length)
        input_graph = create_input_graph(
            qubits=qubits, number_of_vertices=number_of_vertices)
        solve_maxcut(graph=input_graph, steps=steps)


if __name__ == '__main__':
        main()
