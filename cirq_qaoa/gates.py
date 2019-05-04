from cirq import X, Y, Z, H, S, T, I

QUANTUM_GATES = {
    'I': I,
    'X': X,
    'Y': Y,
    'Z': Z,
    'H': H,
    'S': S,
    'T': T
}
"""
Dictionary of quantum gate functions keyed by gate names.
"""

STANDARD_GATES = QUANTUM_GATES
"""
Alias for the above dictionary of quantum gates.
"""
