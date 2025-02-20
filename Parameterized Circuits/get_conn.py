import numpy as np
import cudaq

def get_conn(hamiltonian: cudaq.SpinOperator, bitstring: np.ndarray):
    #print(hamiltonian)
    """
    Get the connected states and matrix elements for a given bitstring under the action
    of a Hamiltonian represented as a sum of Pauli words.

    Parameters:
    -----------
    hamiltonian : cudaq.SpinOperator
        The Hamiltonian, represented as a sum of Pauli words.
    bitstring : np.ndarray
        A binary string (numpy array of 0s and 1s) representing the computational basis state.

    Returns:
    --------
    connected_states : list of np.ndarray
        List of connected bitstrings.
    matrix_elements : list of complex
        List of corresponding matrix elements for the connected bitstrings.
    """
        
    #connected_states = []
    #matrix_elements = []
    state_dict = {}

    def process_term(term):
        term_str = term.to_string(False).strip()  # Get Pauli string without coefficient
        coeff = term.get_coefficient()  # Extract the coefficient

        new_bitstring = np.copy(bitstring)
        phase_factor = 1.0
        valid = True
        

        # Parse the term string to identify Pauli operators and their qubit indices

        for qubit_idx in range(len(term_str)):
            pauli_op = term_str[qubit_idx]

            if pauli_op == 'I':
                continue
            elif pauli_op == 'X':
                new_bitstring[qubit_idx] = - new_bitstring[qubit_idx]
            elif pauli_op == 'Y':
                new_bitstring[qubit_idx] = - new_bitstring[qubit_idx]
                if bitstring[qubit_idx] == 1:
                    phase_factor *= 1j
                else:
                    phase_factor *= -1j
            elif pauli_op == 'Z':
                if bitstring[qubit_idx] == -1:
                    phase_factor *= -1.0
            else:
                valid = False
                print("Invalid Pauli Operator!")
                break

        #print(coeff, term_str, new_bitstring, phase_factor)

        if tuple(new_bitstring) in state_dict: 
            state_dict[tuple(new_bitstring)] += coeff * phase_factor
        else: 
            state_dict[tuple(new_bitstring)] = coeff * phase_factor

    hamiltonian.for_each_term(process_term)

    ## Make them unique
    connected_states = [np.array(state) for state in state_dict.keys()]
    matrix_elements = list(state_dict.values())

    return np.array(connected_states), np.array(matrix_elements)

