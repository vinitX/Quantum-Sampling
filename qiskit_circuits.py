import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import scipy
from qiskit.quantum_info import SparsePauliOp
import itertools

def Euler_angle_decomposition_qiskit(unitary:np.ndarray):
    #Given a 2*2 unitary matrix as np.array, this function computes the Euler angles (theta, phi, lambda) required
    # to implement that unitary as a U3 gate where U3(theta, phi, lambda)
    from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
    gate_decomposer = OneQubitEulerDecomposer('U3')
    theta_val, phi_val, lambda_val, _ = gate_decomposer.angles_and_phase(unitary)
    return (theta_val, phi_val, lambda_val)

def two_qubit_gate_qiskit(circuit, angle:float, qubit_1:int, qubit_2:int, mode="no_decomposition"):
    # This function provides circuit description of RZZ(theta) - This is the 2-qubit gate used for H_Z
    if mode == "no_decomposition":
        circuit.rzz(angle, qubit_1, qubit_2)
    if mode == "CNOT_decomposition":
        circuit.cx(qubit_1, qubit_2)
        circuit.rz(angle, qubit_2)
        circuit.cx(qubit_1, qubit_2)
    if mode == "RZX_decomposition":
        circuit.h(qubit_2)
        circuit.rzx(angle, qubit_1, qubit_2)   # can be implemented natively by pulse-stretching
        circuit.h(qubit_2)
    return circuit


def Trotter_circuit_qiskit(N: int, k:int, alpha:float,
                    gamma:float, l:float, J:float, time_delta=0.5, initial_config=None):
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens

    from collections import defaultdict

    qreg = QuantumRegister(N)
    creg = ClassicalRegister(N)
    circuit = QuantumCircuit(qreg, creg)

    circuit.x(qreg[N-1-np.argwhere(initial_config).flatten()])  # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have no_spins-1-index

    angle_dict = defaultdict(tuple)
    for qubit in np.arange(N):
        one_body_Ham = SparsePauliOp(["X", "Z"], [gamma, alpha*(1-gamma)*l[qubit]]).simplify()
        angle_dict[qubit]= Euler_angle_decomposition_qiskit(scipy.linalg.expm(-1.0j*time_delta*one_body_Ham.to_matrix()))   # always 2*2 so no problem of exponentiation, storage


    for _ in range(k-1):
        for qubit in np.arange(N):
            circuit.u(angle_dict[qubit][0],angle_dict[qubit][1],angle_dict[qubit][2], qreg[qubit])

        for qubit_tuple in list(itertools.combinations(np.arange(N),r=2)):
            circuit = two_qubit_gate_qiskit(circuit,
                                2*J[qubit_tuple[0],qubit_tuple[1]]*(1-gamma)*alpha*time_delta,
                                qreg[qubit_tuple[0]], qreg[qubit_tuple[1]], mode="no_decomposition")
                                # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have no_spins-1-index
        circuit.barrier()

    for qubit in np.arange(N):
        circuit.u(angle_dict[qubit][0],angle_dict[qubit][1],angle_dict[qubit][2], qreg[qubit])

    return circuit
