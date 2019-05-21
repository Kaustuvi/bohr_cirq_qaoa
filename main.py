from cirq_qaoa.cirq_max_cut_solver import define_grid_qubits, solve_maxcut


def main():
        size = 2
        steps = 2
        qubits = define_grid_qubits(size=size)
        qubit_pairs = [(qubits[0], qubits[1]), (qubits[0],
                                                qubits[2]), (qubits[1], qubits[2])]
        solve_maxcut(qubit_pairs=qubit_pairs, steps=steps)


if __name__ == '__main__':
        main()
