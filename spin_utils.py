import numpy as np
from collections import defaultdict
from scipy.sparse import kron, identity, csr_matrix


class PauliTerm:
    def __init__(self, n_qubits, ops_dict=None, coeff=1.0):
        self.n_qubits = n_qubits
        self.coeff = coeff
        self.ops = ops_dict if ops_dict else {}  # maps qubit index -> 'X','Y','Z','I'
        self._normalize()

    def _normalize(self):
        """ Remove identity operators and sort keys """
        self.ops = {k: v for k, v in self.ops.items() if v != 'I'}
        self.ops = dict(sorted(self.ops.items()))

    def __mul__(self, other):
        if isinstance(other, PauliTerm):
            assert self.n_qubits == other.n_qubits, "Mismatched qubit counts"
            new_ops = self.ops.copy()
            coeff = self.coeff * other.coeff

            for k, op in other.ops.items():
                if k not in new_ops:
                    new_ops[k] = op
                else:
                    a, b = new_ops[k], op
                    if a == b:
                        new_ops[k] = 'I'
                    else:
                        rule = {
                            ('X', 'Y'): ('Z', 1j),
                            ('Y', 'X'): ('Z', -1j),
                            ('Y', 'Z'): ('X', 1j),
                            ('Z', 'Y'): ('X', -1j),
                            ('Z', 'X'): ('Y', 1j),
                            ('X', 'Z'): ('Y', -1j)
                        }
                        new_op, phase = rule.get((a, b), ('I', 1))
                        new_ops[k] = new_op
                        coeff *= phase

            return PauliTerm(self.n_qubits, new_ops, coeff)
        elif isinstance(other, (int, float, complex)):
            return PauliTerm(self.n_qubits, self.ops.copy(), self.coeff * other)
        else:
            raise TypeError("Unsupported multiplication")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, PauliSum):
            return other + self
        elif isinstance(other, PauliTerm):
            return PauliSum(self.n_qubits, [self, other])
        else:
            raise TypeError("Unsupported addition")

    def to_dense(self):
        pauli_map = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
        ops = [pauli_map[self.ops.get(i, 'I')] for i in range(self.n_qubits)]
        mat = ops[0]
        for op in ops[1:]:
            mat = np.kron(mat, op)
        return self.coeff * mat

    def to_sparse(self):
        pauli_map = {
            'I': identity(2, format='csr', dtype=complex),
            'X': csr_matrix([[0, 1], [1, 0]], dtype=complex),
            'Y': csr_matrix([[0, -1j], [1j, 0]], dtype=complex),
            'Z': csr_matrix([[1, 0], [0, -1]], dtype=complex)
        }
        ops = [pauli_map[self.ops.get(i, 'I')] for i in range(self.n_qubits)]
        mat = ops[0]
        for op in ops[1:]:
            mat = kron(mat, op, format='csr')
        return self.coeff * mat

    def __str__(self):
        if not self.ops:
            return f"{self.coeff:.3g} * I"
        parts = [f"{self.coeff:.3g}"]
        for k, v in sorted(self.ops.items()):
            parts.append(f"{v}{k}")
        return " * ".join(parts)

    def __hash__(self):
        return hash((tuple(sorted(self.ops.items())), self.n_qubits))

    def __eq__(self, other):
        return isinstance(other, PauliTerm) and self.n_qubits == other.n_qubits and self.ops == other.ops


class PauliSum:
    def __init__(self, n_qubits, terms=None):
        self.n_qubits = n_qubits
        self.terms = terms if terms else []
        self._simplify()

    def __add__(self, other):
        if isinstance(other, PauliTerm):
            return PauliSum(self.n_qubits, self.terms + [other])
        elif isinstance(other, PauliSum):
            return PauliSum(self.n_qubits, self.terms + other.terms)
        else:
            raise TypeError("Unsupported addition")

    def _simplify(self):
        term_map = defaultdict(complex)
        for term in self.terms:
            key = tuple(sorted(term.ops.items()))
            term_map[key] += term.coeff
        self.terms = [PauliTerm(self.n_qubits, dict(k), v) for k, v in term_map.items() if abs(v) > 1e-12]

    def to_dense(self):
        dim = 2 ** self.n_qubits
        mat = np.zeros((dim, dim), dtype=complex)
        for term in self.terms:
            mat += term.to_dense()
        return mat

    def to_sparse(self):
        dim = 2 ** self.n_qubits
        mat = csr_matrix((dim, dim), dtype=complex)
        for term in self.terms:
            mat += term.to_sparse()
        return mat

    def __str__(self):
        return " + ".join(str(term) for term in self.terms) if self.terms else "0"


class SpinOperator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def x(self, i): return PauliTerm(self.n_qubits, {i: 'X'})
    def y(self, i): return PauliTerm(self.n_qubits, {i: 'Y'})
    def z(self, i): return PauliTerm(self.n_qubits, {i: 'Z'})
    def i(self, i): return PauliTerm(self.n_qubits, {i: 'I'})


# Example usage
'''spin = SpinOperator(n_qubits=4)
Op = 2.5 * spin.x(1) * spin.y(0) + 0.1 * spin.i(3) * spin.z(1) + 2.5 * spin.y(0) * spin.x(1)  # duplicates

# Pretty-print
op_str = str(Op)

# Dense and sparse matrices
dense_op = Op.to_dense()
sparse_op = Op.to_sparse()

op_str, dense_op, sparse_op.shape'''



def apply_pauli(op, bitstring):
    print("Applying Pauli:", op)
    print(len(op))
    new_bitstring = np.copy(bitstring)
    phase_factor = np.ones(len(bitstring), dtype=complex)

    for qubit_idx in op.keys():
        pauli_op = op[qubit_idx]
        print(pauli_op, qubit_idx)

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


def get_conn(hamiltonian: SpinOperator, bitstring: np.ndarray):  
    spin_list = []
    val_list = []

    for term in hamiltonian.terms:
        op = term.ops  # Get Pauli op without coefficient
        coeff = term.coeff  # Extract the coefficient

        #print(term, op, coeff)

        new_bitstring, phase_factor = apply_pauli(op, bitstring)

        spin_list.append(new_bitstring)
        val_list.append(phase_factor * coeff)
    
    spin_list = np.transpose(spin_list, [1,0,2])
    val_list = np.transpose(val_list)

    return spin_list, val_list


# Example usage
'''spin = SpinOperator(n_qubits=2)
Hamiltonian = 2.5 * spin.x(1) * spin.y(0) + 0.1 * spin.z(1) + 2.5 * spin.y(0) * spin.x(1)  
bitstring = np.array([[-1, -1], [1, 1]])  
get_conn(Hamiltonian, bitstring)  '''