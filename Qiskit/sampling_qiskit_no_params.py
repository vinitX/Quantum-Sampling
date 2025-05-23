from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import time


def Euler_angle_decomposition(U:np.ndarray):
    theta = 2 * np.arccos(np.abs(U[0, 0]))

    phi = np.angle(U[1, 0]) - np.angle(U[0, 0])
    lam = np.angle(U[1, 1]) - np.angle(U[1, 0])
    
    return theta, phi, lam
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import time

def two_qubit_gate_qiskit(circuit, angle: float, qubit_1: int, qubit_2: int):
    # Use CNOT decomposition for RZZ as it often transpiles better
    circuit.cx(qubit_1, qubit_2)
    circuit.rz(angle, qubit_2)
    circuit.cx(qubit_1, qubit_2)
    return circuit

def create_trotter_template(N, k, angles_u3, angles_2q):
    circuit = QuantumCircuit(N)
    # Ry gates replaced with classical decisions later
    for _ in range(k):
        # Fixed U3 gates
        for i in range(N):
            circuit.u(angles_u3[3*i], angles_u3[3*i+1], angles_u3[3*i+2], i)
        # 2-qubit gates
        for i in range(N):
            for j in range(i+1, N):
                two_qubit_gate_qiskit(circuit, angles_2q[i,j], i, j)
    circuit.measure_all()
    return circuit

def dict_to_res(counts):
    res_str = next(iter(counts.keys()))
    return np.array([1.0 if c == '1' else -1.0 for c in res_str[::-1]])

def main(N, sample_size):
    k = 24
    s = np.random.choice([1., -1.], size=N)
    
    # Generate fixed angles
    angles_u3 = np.random.uniform(0, 2*np.pi, 3*N)
    angles_2q = np.random.uniform(0, 2*np.pi, (N, N))
    
    # Create base circuit template (without initial X gates)
    circuit_template = create_trotter_template(N, k, angles_u3, angles_2q)
    
    # Transpile once with optimization
    simulator = AerSimulator() #method='statevector', optimization_level=3)
    #transpiled_circuit = transpile(circuit_template, simulator)
    
    tim = time.time()
    
    for _ in range(sample_size):
        circuit = QuantumCircuit(N)
        for q in range(N):
            if s[q] == 1:
                circuit.x(q)
        circuit.compose(circuit_template, inplace=True)
        
        result = simulator.run(circuit, shots=1).result()
        counts = result.get_counts()
        s = dict_to_res(counts)
    
    print(f"Optimized Sampling Time: {time.time() - tim:.2f}s")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('--sample_size', type=int, default=100)

    args = parser.parse_args()
    main(args.N, args.sample_size)
    