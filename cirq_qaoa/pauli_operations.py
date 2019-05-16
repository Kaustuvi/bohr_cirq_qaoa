import numpy as np
import warnings

from collections.abc import Sequence
from cirq import PauliString, Circuit, GridQubit, inverse, InsertStrategy, ZPowGate, Rx, Rz, X, Y, H, CNOT


class CirqPauliSum:
    """
    CirqPauliSum is an utitlity class that adds PauliStrings according to Pauli algebra rules
    """

    def __init__(self, pauli_strings):
        if not (isinstance(pauli_strings, Sequence)
                and all([isinstance(pauli_string, PauliString) for pauli_string in pauli_strings])):
            raise ValueError(
                "CirqPauliSum's are currently constructed from Sequences of PauliStrings.")
        self.pauli_strings = pauli_strings

    def __str__(self):
        return " + ".join([str(pauli_string) for pauli_string in self.pauli_strings])

    def __add__(self, other):
        new_pauli_strings = self.pauli_strings
        new_pauli_strings.extend(other.pauli_strings)
        new_pauli_strings = CirqPauliSum(new_pauli_strings)
        return new_pauli_strings.simplify()

    def simplify(self):
        return simplify_cirq_pauli_sum(self)


def add_pauli_strings(pauli_strings):
    """
    adds PauliString objects and returns a CirqPauliSum object simplified
    according to Pauli algebra rules

    Parameters
    ----------
    pauli_strings   :   [list of PauliString objects] whose sum is to be calculated

    Returns
    -------
    CirqPauliSum object simplified according to Pauli algebra rules
    """

    sum_of_pauli_strings = CirqPauliSum(pauli_strings)
    return sum_of_pauli_strings.simplify()


def simplify_cirq_pauli_sum(cirq_pauli_sum):
    """
    simplifies the input CirqPauliSum object according to Pauli algebra rules
    
    Parameters
    ----------
    cirq_pauli_sum  :   CirqPauliSum object that needs to be simplified according to 
                        Pauli algebra rules
                
    Returns
    -------
    simplified CirqPauliSum object
    """

    pauli_strings = []
    identity_strings = []
    for pauli_string in cirq_pauli_sum.pauli_strings:
        if not pauli_string._qubit_pauli_map == {} and not np.isclose(pauli_string.coefficient, 0.0):
            pauli_strings.append(pauli_string)
        else:
            identity_strings.append(pauli_string)
    coeff = sum(i.coefficient for i in identity_strings)
    if not np.isclose(coeff, 0.0):
        total_identity_string = PauliString(
            qubit_pauli_map={}, coefficient=coeff)
        pauli_strings.append(total_identity_string)
    return CirqPauliSum(pauli_strings)


def exponentiate_pauli_string(pauli_string):
    """
    Returns a function f(alpha) that constructs the Circuit corresponding to exp(-1j*alpha*pauli_string).

    Parameters
    ----------
    pauli_string    :   A PauliString to exponentiate

    Returns
    -------
    A function that takes an angle parameter and returns a circuit.
    """
    coeff = pauli_string.coefficient.real
    pauli_string._coefficient = pauli_string.coefficient.real

    def exponentiation_circuit(param):
        circuit = Circuit()
        if is_identity(pauli_string):
            PHASE = ZPowGate(exponent=(-param * coeff)/np.pi)
            circuit.append([X(GridQubit(0, 0))],
                           strategy=InsertStrategy.EARLIEST)
            circuit.append([PHASE(GridQubit(0, 0))],
                           strategy=InsertStrategy.EARLIEST)
            circuit.append([X(GridQubit(0, 0))],
                           strategy=InsertStrategy.EARLIEST)
            circuit.append([PHASE(GridQubit(0, 0))],
                           strategy=InsertStrategy.EARLIEST)
        elif is_zero(pauli_string):
            pass
        else:
            circuit += exponentiate_general_case(pauli_string, param)
        return circuit

    return exponentiation_circuit


def exponentiate_general_case(pauli_string, param):
    """
    Returns a cirq (Circuit()) object corresponding to the exponential of
    the pauli_string object, i.e. exp[-1.0j * param * pauli_string]

    Parameters:
    ----------
    pauli_string    :   A PauliString to exponentiate
    param           :   scalar, non-complex, value

    Returns:
    -------
    A cirq Circuit object
    """
    def reverse_circuit_operations(c):
        reverse_circuit = Circuit()
        operations_in_c = []
        reverse_operations_in_c = []
        for operation in c.all_operations():
            operations_in_c.append(operation)

        reverse_operations_in_c = inverse(operations_in_c)

        for operation in reverse_operations_in_c:
            reverse_circuit.append(
                [operation], strategy=InsertStrategy.EARLIEST)
        return reverse_circuit

    circuit = Circuit()
    change_to_z_basis = Circuit()
    change_to_original_basis = Circuit()
    cnot_seq = Circuit()
    prev_qubit = None
    highest_target_qubit = None

    for qubit, pauli in pauli_string.items():
        if pauli == X:
            change_to_z_basis.append(
                [H(qubit)], strategy=InsertStrategy.EARLIEST)
            change_to_original_basis.append(
                [H(qubit)], strategy=InsertStrategy.EARLIEST)

        elif pauli == Y:
            RX = Rx(np.pi/2.0)
            change_to_z_basis.append(
                [RX(qubit)], strategy=InsertStrategy.EARLIEST)
            RX = inverse(RX)
            change_to_original_basis.append(
                [RX(qubit)], strategy=InsertStrategy.EARLIEST)

        if prev_qubit is not None:
            cnot_seq.append([CNOT(prev_qubit, qubit)],
                            strategy=InsertStrategy.EARLIEST)

        prev_qubit = qubit
        highest_target_qubit = qubit

    circuit += change_to_z_basis
    circuit += cnot_seq
    RZ = Rz(2.0 * pauli_string.coefficient *
            param)
    circuit.append([RZ(highest_target_qubit)],
                   strategy=InsertStrategy.EARLIEST)
    circuit += reverse_circuit_operations(cnot_seq)
    circuit += change_to_original_basis

    return circuit


def is_zero(pauli_string):
    """
    Checks if a PauliString is zero.

    Parameters
    ----------
    pauli_string    :   a PauliString object

    Returns
    -------
    True if PauliString is zero, False otherwise
    """
    if isinstance(pauli_string, PauliString):
        return np.isclose(pauli_string.coefficient, 0)
    else:
        raise TypeError("is_zero only checks PauliString objects!")


def is_identity(pauli_string):
    """
    Checks if a PauliString is a scalar multiple of identity

    Parameters
    ----------
    pauli_string    :   a PauliString object

    Returns
    -------
    True if PauliString is a scalar multiple of identity, False otherwise
    """
    if isinstance(pauli_string, PauliString):
        return (len(pauli_string) == 0) and (not np.isclose(pauli_string.coefficient, 0))
    else:
        raise TypeError("is_identity only checks PauliString objects!")
