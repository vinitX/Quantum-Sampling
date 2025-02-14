import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import scipy
from qiskit.quantum_info import SparsePauliOp
import itertools

'''def Euler_angle_decomposition_qiskit(unitary:np.ndarray):
    #Given a 2*2 unitary matrix as np.array, this function computes the Euler angles (theta, phi, lambda) required
    # to implement that unitary as a U3 gate where U3(theta, phi, lambda)
    from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
    gate_decomposer = OneQubitEulerDecomposer('U3')
    theta_val, phi_val, lambda_val, _ = gate_decomposer.angles_and_phase(unitary)
    return (theta_val, phi_val, lambda_val)'''

def Euler_angle_decomposition(U:np.ndarray):
    theta = 2 * np.arccos(np.abs(U[0, 0]))

    phi = np.angle(U[1, 0]) - np.angle(U[0, 0])
    lam = np.angle(U[1, 1]) - np.angle(U[1, 0])
    
    return theta, phi, lam

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


def Fixed_Trotter_qiskit(N, k, alpha, gamma, time_delta, theta, phi, lam, J,):
    circuit = QuantumCircuit(N)

    for _ in range(k-1):
        for qubit in np.arange(N):
            circuit.u(theta[qubit],phi[qubit],lam[qubit], qubit)

        for qubit_tuple in list(itertools.combinations(np.arange(N),r=2)):
            circuit = two_qubit_gate_qiskit(circuit,
                                2*J[qubit_tuple[0],qubit_tuple[1]]*(1-gamma)*alpha*time_delta,
                                qubit_tuple[0], qubit_tuple[1], mode="no_decomposition")
                                # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have no_spins-1-index
        circuit.barrier()

    for qubit in np.arange(N):
        circuit.u(theta[qubit],phi[qubit],lam[qubit], qubit)

    return circuit


def initialize_from_bitstring(qc, N, initial_config):
    for i in range(N):
        if int(initial_config[i]) == 1: qc.x(N-1-i)   # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have no_spins-1-index


def Trotter_circuit_qiskit(N, fixed_circuit, initial_config):
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens

    #qreg = QuantumRegister(N)
    #creg = ClassicalRegister(N)
    circuit = QuantumCircuit(N) #, creg)
    initialize_from_bitstring(circuit, N, initial_config)
    circuit.compose(fixed_circuit, inplace=True)

    return circuit
