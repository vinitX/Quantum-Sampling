import tensorcircuit as tc
import numpy as np
import time
import jax
import jax.numpy as jnp

tc.set_backend("jax")  # Use JAX backend for JIT performance
backend = tc.backend

def two_qubit_gate_tc(circuit, angle: float, qubit_1: int, qubit_2: int):
    circuit.cnot(qubit_1, qubit_2)
    circuit.rz(qubit_2, theta=angle)
    circuit.cnot(qubit_1, qubit_2)
    return circuit

# Wrapper function to fix N and k at definition time
def make_trotter_circuit(N, k):
    @jax.jit
    def Trotter_circuit_tensorcircuit(angles_ry, angles_u3, angles_2q):
        circuit = tc.Circuit(N)  # N is fixed at function creation time
        for i in range(N):
            circuit.ry(i, theta=angles_ry[i])

        for _ in range(k - 1):
            for i in range(N):
                theta, phi, lamb = angles_u3[i * 3], angles_u3[i * 3 + 1], angles_u3[i * 3 + 2]
                circuit.rz(i, theta=phi)
                circuit.ry(i, theta=theta)
                circuit.rz(i, theta=lamb)

            for i in range(N):
                for j in range(i + 1, N):
                    two_qubit_gate_tc(circuit, angles_2q[i, j], i, j)

        for i in range(N):
            theta, phi, lamb = angles_u3[i * 3], angles_u3[i * 3 + 1], angles_u3[i * 3 + 2]
            circuit.rz(i, theta=phi)
            circuit.ry(i, theta=theta)
            circuit.rz(i, theta=lamb)

        # Instead of returning circuit, return measurement results as a JAX array
        bits, _ = circuit.measure(*range(N))
        return jnp.array(bits)  # Ensure output is JAX-compatible

    return Trotter_circuit_tensorcircuit  # Return JIT-compiled function

def main(N, sample_size):
    k = 24
    trotter_circuit_fn = make_trotter_circuit(N, k)  # Create a fixed version of the function
    
    s = np.random.choice([1., -1.], size=N)
    
    angles_u3 = backend.convert_to_tensor(np.random.uniform(0, 2 * np.pi, 3 * N))
    angles_2q = backend.convert_to_tensor(np.random.uniform(0, 2 * np.pi, (N, N)))
    
    for idx in range(sample_size):
        tim = time.time()
        angles_ry = backend.convert_to_tensor(np.pi * (s + 1) / 2)
        bits = trotter_circuit_fn(angles_ry, angles_u3, angles_2q)  # JIT-compiled function returns bits
        s = bits * 2 - 1 
    
        print(idx, "\t", f"{time.time()-tim:.{2}e}")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('--sample_size', type=int, default=100)

    args = parser.parse_args()
    main(args.N, args.sample_size)
    