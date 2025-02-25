import numpy as np
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
#from qiskit.providers.aer import AerSimulator


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

    #qreg = QuantumRegister(N)
    #creg = ClassicalRegister(N)
    circuit = QuantumCircuit(N) #, creg)

    ry_params = ParameterVector('ry', N)

    for i in range(N):
        circuit.ry(ry_params[i], i)

    for _ in range(k-1):
        for i in np.arange(N):
            circuit.u(angles_u3[i*3], angles_u3[i*3+1], angles_u3[i*3+2], i)

        for i in range(N):
            for j in range(i + 1, N):
                circuit = two_qubit_gate_qiskit(circuit, angles_2q[i,j], i, j, mode="no_decomposition")

    for i in np.arange(N):
        circuit.u(angles_u3[i*3], angles_u3[i*3+1], angles_u3[i*3+2], i)

    circuit.measure_all()
    return circuit, ry_params


def dict_to_res(counts):
  for key, value in counts.items():
    if value == 1: 
        final_config = key

  res = [1.0 if s == '1' else -1.0 for s in final_config]
  
  return np.array(res)


def main(N,sample_size):
  k = 24
  s = np.random.choice([1.,-1.],size=N)

  angles_u3 = np.random.uniform(0,2*np.pi,3*N)
  angles_2q = np.random.uniform(0,2*np.pi,(N,N))

  angles_ry_params = ParameterVector('theta', N)

  circuit_template, ry_params = Trotter_circuit_qiskit(N, k, angles_ry_params, angles_u3, angles_2q)
  
  simulator = AerSimulator()
  #transpiled_circuit = transpile(circuit_template, simulator)
  
  tim = time.time()

  for _ in range(sample_size):
    angles_ry = np.pi*(s + 1)/2
    bound_circuit = circuit_template.assign_parameters(dict(zip(ry_params, angles_ry)))
    result = simulator.run(bound_circuit, shots=1).result()
    counts = result.get_counts(bound_circuit)
    s = dict_to_res(counts)

  print("Sampling Time: ", time.time()-tim)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('--sample_size', type=int, default=100)

    args = parser.parse_args()
    main(args.N, args.sample_size)
    