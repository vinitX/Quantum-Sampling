import numpy as np
import time
import argparse
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

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

def Trotter_circuit_qiskit(N, k, bitstring, angle_rx, angles_rz, angles_2q):
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
    circuit = QuantumCircuit(N) #, creg)
    for i in np.arange(N):
        if bitstring[i] == 1: circuit.x(i)

    for _ in range(k):
        for i in np.arange(N):
            circuit.rx(angle_rx, i)

        for i in np.arange(N):
            circuit.rz(angles_rz[i], i)

        for i in range(N):
            for j in range(i + 1, N):
                circuit = two_qubit_gate_qiskit(circuit, angles_2q[i,j], i, j, mode="no_decomposition")
        
    return circuit

def dict_to_res(counts):
  for key, value in counts.items():
    if value == 1: 
        final_config = key

  res = [1.0 if s == '1' else -1.0 for s in final_config]
  
  return np.array(res)


def run_sequential_qasm(N, k, angle_rx, angles_rz, angles_2q, runs=10, shots=1, seed=None):
    simulator = AerSimulator()
    results = []
    bitstring = np.random.choice([1,0],size=N)
    print("Initial bitstring:", bitstring)

    for _ in range(runs):
        circuit = Trotter_circuit_qiskit(N, k, bitstring, angle_rx, angles_rz, angles_2q)
        circuit.measure_all()
    
        transpiled = transpile(circuit, simulator, seed_transpiler=seed)
        job = simulator.run(transpiled, device="CPU", precision="single", shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()
        print("Counts:", counts)

        # Use the most frequent outcome (shots=1 yields the single observed bitstring).
        bitstring = max(counts, key=counts.get)
        bitstring = np.array([int(b) for b in bitstring[::-1]])  # Reverse to match Qiskit's qubit ordering 
        print("Selected bitstring:", bitstring)
        results.append(bitstring)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run sequential QASM simulations with measurement feedback.")
    # parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--trotter-steps", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # N = args.num_qubits
    k = args.trotter_steps
    runs = args.sample_size

    angle_rx = np.pi / 2

    for _ in range(1):  
        for N in np.arange(4, 5, 2):

            angles_rz = np.random.uniform(0, 2*np.pi, N)
            angles_2q = np.random.uniform(0, 2*np.pi, (N, N))

            t0 = time.time()
            results = run_sequential_qasm(N, k, angle_rx, angles_rz, angles_2q, runs=runs, shots=1, seed=args.seed,)
            t1 = time.time()
            print(f"Size: {N}  Time: {t1 - t0} sec")
            # print("Sequential measurement bitstrings:", results)

            with open("qiskit_CPU_times.txt", "a") as f:
                f.write(f"{N},{t1 - t0}\n")

if __name__ == "__main__":
    main()

    