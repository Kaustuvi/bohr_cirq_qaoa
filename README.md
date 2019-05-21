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
         `VQE` is an object that encapsulates the VQE algorithm (functional minimization). The main components of the VQE algorithm are a minimizer function for performing the functional minimization, a function that takes a vector of parameters and returns a Cirq circuit, and a Hamiltonian for which the expectation value is to be calculated.
    -   `pauli_operations.py`   :   contains class `CirqPauliSum`  
         `CirqPauliSum` is a utitlity class that adds PauliStrings according to Pauli algebra rules. This file also contains the function `exponentiate_pauli_string()` for constructing a circuit corresponding to ![equation](https://latex.codecogs.com/gif.latex?e%5E%7B-j*%5Calpha*term%7D) for the cost or mixing clause `term` and a parameter ![equation](https://latex.codecogs.com/gif.latex?%5Calpha)
-   `main.py`   :   runs an implementation of QAOA to solve the Max-Cut problem on an input graph

### Commit Hash and Disclaimer
The implementation of QAOA in this project is based on Rigetti's implementation of QAOA in Rigetti's [Grove](https://github.com/rigetti/grove.git) project with the commit hash `dc6bf6ec63e8c435fe52b1e00f707d5ce4cdb9b3`.  
The function `get_parameterized_circuit()` in `qaoa.py` and the code for the class `VQE` in `vqe.py` are copied from Grove with the original copyright disclaimer:  

Copyright 2016-2017 Rigetti Computing  
Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0  
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
        
### Work in Progress
QAOA has been implemented successfully in the Cirq framework and has also been used to solve the Max-Cut problem for unweighted input graphs. 

### Possible Further Steps
The next step of this project is to compare the results of the [QAOA implementation](https://grove-docs.readthedocs.io/en/latest/qaoa.html) from Grove with the current implementation in terms of the possible following paramters:
- Runtimes
- Number of steps in optimization
- Circuit depth (after compilation)
- Possibly introduce noise
