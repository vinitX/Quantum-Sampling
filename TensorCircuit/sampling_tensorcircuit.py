import tensorcircuit as tc
import numpy as np
import time

def two_qubit_gate_tc(circuit, angle: float, qubit_1: int, qubit_2: int):
    # RZZ implementation using CNOT decomposition
    circuit.cnot(qubit_1, qubit_2)
    circuit.rz(qubit_2, theta=angle)
    circuit.cnot(qubit_1, qubit_2)
    return circuit

def Trotter_circuit_tensorcircuit(N, k, angles_ry, angles_u3, angles_2q):
    circuit = tc.Circuit(N)
    for i in range(N):
        circuit.ry(i, theta=angles_ry[i])

    for _ in range(k-1):
        # Apply U3 gates decomposed into RZ-RY-RZ
        for i in range(N):
            theta, phi, lamb = angles_u3[i*3], angles_u3[i*3+1], angles_u3[i*3+2]
            circuit.rz(i, theta=phi)
            circuit.ry(i, theta=theta)
            circuit.rz(i, theta=lamb)

        # Apply 2-qubit gates
        for i in range(N):
            for j in range(i + 1, N):
                two_qubit_gate_tc(circuit, angles_2q[i,j], i, j)

    # Final U3 layer
    for i in range(N):
        theta, phi, lamb = angles_u3[i*3], angles_u3[i*3+1], angles_u3[i*3+2]
        circuit.rz(i, theta=phi)
        circuit.ry(i, theta=theta)
        circuit.rz(i, theta=lamb)

    return circuit

def main(N, sample_size):
    k = 24
    s = np.random.choice([1., -1.], size=N)
    
    angles_u3 = np.random.uniform(0, 2*np.pi, 3*N)
    angles_2q = np.random.uniform(0, 2*np.pi, (N, N))
    
    tim = time.time()
    
    for _ in range(sample_size):
        angles_ry = np.pi * (s + 1) / 2
        circuit = Trotter_circuit_tensorcircuit(N, k, angles_ry, angles_u3, angles_2q)
        bits, _ = circuit.measure(*range(N)) 
        s = bits*2-1 
    
    print("Sampling Time: ", time.time() - tim)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('--sample_size', type=int, default=100)

    args = parser.parse_args()
    main(args.N, args.sample_size)
    