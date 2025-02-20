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

def initialize_from_bitstring(qc, N, angles_ry):
    for i in range(N):
        qc.ry(angles_ry[i], i)

def Trotter_circuit_qiskit(N, k, angles_ry, angles_u3, angles_2q):
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens

    #qreg = QuantumRegister(N)
    #creg = ClassicalRegister(N)
    circuit = QuantumCircuit(N) #, creg)
    initialize_from_bitstring(circuit, N, angles_ry)

    for _ in range(k-1):
        for i in np.arange(N):
            circuit.u(angles_u3[i*3], angles_u3[i*3+1], angles_u3[i*3+2], i)

        for i in range(N):
            for j in range(i + 1, N):
                circuit = two_qubit_gate_qiskit(circuit, angles_2q[i,j], i, j, mode="no_decomposition")

        circuit.barrier()

    for i in np.arange(N):
        circuit.u(angles_u3[i*3], angles_u3[i*3+1], angles_u3[i*3+2], i)
        
    return circuit
