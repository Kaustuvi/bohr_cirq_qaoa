# Cirq QAOA
### Introduction
The goal of this project was to implement the **Quantum Approximate Optimization Algorithm (QAOA)** by *Edward Farhi, Jeffrey Goldstone and Sam Gutmann* in the **Cirq framework** from Google. As an application, this implmentation was used to solve the Maximum Cut problem. 

### References
- The original [QAOA paper](https://arxiv.org/abs/1411.4028)
- [Cirq docs](https://cirq.readthedocs.io/en/stable/) 
- [Maximum Cut overview](https://en.wikipedia.org/wiki/Maximum_cut)

### Modules
- `cirq_qaoa`   :   
    Contains classes and routines to implement QAOA and solve the Max-Cut problem using **Variational Quantum Eigensolver** ([VQE](https://arxiv.org/abs/1304.3061))
    -  `cirq_maxcut_solver.py` :   contains class `CirqMaxCutSolver`
        `CirqMaxCutSolver` creates the cost operators and the mixing operators for the input graph and returns a QAOA object that solves the Maxcut problem for the input graph
    - `qaoa.py`    : contains class `QAOA`
       `QAOA` constains all information for running the QAOA algorthm to find the ground state of the list of cost clauses.
    -   `vqe.py`    :   contains class `VQE`
         `VQE` is an object that encapsulates the VQE algorithm (functional minimization). The main components of the VQE algorithm are a minimizer function for performing the functional minimization, a function that takes a vector of parameters and returns a Cirq circuit, and a Hamiltonian of which to calculate the expectation value.
         The code for implementing VQE is copied from Rigetti's [Grove](https://grove-docs.readthedocs.io/en/latest/) project
    -   `pauli_operations.py`   :   contains class `CirqPauliSum`
         `CirqPauliSum` is a utitlity class that adds PauliStrings according to Pauli algebra rules. This file also contains the function `exponentiate_pauli_string()` for constructing a circuit corresponding to ![equation](https://latex.codecogs.com/gif.latex?e%5E%7B-j*%5Calpha*term%7D) for the cost or mixing clause `term` and a parameter ![equation](https://latex.codecogs.com/gif.latex?%5Calpha)
-   `main.py`   :   runs an implementation of QAOA to solve the Max-Cut problem on an input graph

### Work in Progress
QAOA has been implemented successfully in the Cirq framework and has also been used to solve the Max-Cut problem for unweighted input graphs. 

### Further Steps
The next step of this experiment is to compare the results of the [QAOA implementation](https://grove-docs.readthedocs.io/en/latest/qaoa.html) from Grove with the current implementation in terms of the possible following paramters:
- Runtimes
- Number of steps in optimization
- Circuit depth (after compilation)
- Possibly introduce noise
