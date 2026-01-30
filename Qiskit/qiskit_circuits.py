import argparse
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

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

def Trotter_circuit_qiskit(N, k, angles_ry, angles_u3, angles_2q):
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
    circuit = QuantumCircuit(N) #, creg)
    for i in range(N):
        circuit.ry(angles_ry[i], i)

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

def _apply_bitstring_initialization(circuit, bitstring):
    # Qiskit bitstrings are reported with the highest-index classical bit on the left.
    # We reverse to map left-to-right into qubit indices [0..N-1].
    bits = list(reversed(bitstring.strip()))
    for q, b in enumerate(bits):
        if b == "1":
            circuit.x(q)
    return circuit

def run_sequential_qasm(N, k, angles_ry, angles_u3, angles_2q, runs=10, shots=1, seed=None):
    simulator = AerSimulator()
    results = []
    prev_bitstring = None

    for _ in range(runs):
        circuit = Trotter_circuit_qiskit(N, k, angles_ry, angles_u3, angles_2q)
        if prev_bitstring is not None:
            _apply_bitstring_initialization(circuit, prev_bitstring)

        creg = ClassicalRegister(N)
        circuit.add_register(creg)
        circuit.measure(range(N), range(N))

        transpiled = transpile(circuit, simulator, seed_transpiler=seed)
        job = simulator.run(transpiled, shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()
        print("Counts:", counts)

        # Use the most frequent outcome (shots=1 yields the single observed bitstring).
        prev_bitstring = max(counts, key=counts.get)
        print("Selected bitstring:", prev_bitstring)
        results.append(prev_bitstring)

    return results

def main():
    parser = argparse.ArgumentParser(description="Run sequential QASM simulations with measurement feedback.")
    parser.add_argument("--num-qubits", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    N = args.num_qubits
    k = args.trotter_steps
    angles_ry = np.zeros(N)
    angles_u3 = np.zeros(N * 3)
    angles_2q = np.zeros((N, N))

    results = run_sequential_qasm(
        N,
        k,
        angles_ry,
        angles_u3,
        angles_2q,
        runs=args.runs,
        shots=args.shots,
        seed=args.seed,
    )
    print("Sequential measurement bitstrings:", results)

if __name__ == "__main__":
    main()
