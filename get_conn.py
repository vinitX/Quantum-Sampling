import numpy as np
import cudaq

def apply_pauli(op, bitstring):
    new_bitstring = np.copy(bitstring)
    phase_factor = np.ones(len(bitstring), dtype=complex)

    for qubit_idx in range(len(op)):
        pauli_op = op[qubit_idx]

        if pauli_op == 'I':
            continue
        elif pauli_op == 'X':
            new_bitstring[:,qubit_idx] = - new_bitstring[:,qubit_idx]
        elif pauli_op == 'Y':
            new_bitstring[:,qubit_idx] = - new_bitstring[:,qubit_idx]
            phase_factor *= 1j*bitstring[:,qubit_idx]
        elif pauli_op == 'Z':
            phase_factor *= bitstring[:,qubit_idx]
        else:
            print("Invalid Pauli Operator!")
            break

    return new_bitstring, phase_factor


def get_conn(hamiltonian: cudaq.SpinOperator, bitstring: np.ndarray):  
    spin_list = []
    val_list = []

    def process_term(term):
        op = term.to_string(False).strip()  # Get Pauli op without coefficient
        coeff = term.get_coefficient()  # Extract the coefficient

        new_bitstring, phase_factor = apply_pauli(op, bitstring)

        spin_list.append(new_bitstring)
        val_list.append(phase_factor * coeff)

    hamiltonian.for_each_term(process_term)
    
    spin_list = np.transpose(spin_list, [1,0,2])
    val_list = np.transpose(val_list)

    return spin_list, val_list