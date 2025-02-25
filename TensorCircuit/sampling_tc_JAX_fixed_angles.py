import tensorcircuit as tc
import numpy as np
import time
import jax
import jax.numpy as jnp

print(jax.devices())

tc.set_backend("jax")  # Use JAX backend for JIT performance
tc.set_contractor("greedy")
backend = tc.backend
tc.set_dtype("complex64")

def two_qubit_gate_tc(circuit, angle: float, qubit_1: int, qubit_2: int):
    circuit.cnot(qubit_1, qubit_2)
    circuit.rz(qubit_2, theta=angle)
    circuit.cnot(qubit_1, qubit_2)
    return circuit

# Use static_args in JIT to avoid recompilation for fixed angles
def make_trotter_circuit(N, k, angles_u3, angles_2q):
    @jax.jit
    def Trotter_circuit_tensorcircuit(angles_ry):
        circuit = tc.Circuit(N)

        # Apply variable ry gates first
        for i in range(N):
            circuit.ry(i, theta=angles_ry[i])

        # Now apply fixed gates (precomputed values)
        for _ in range(k - 1):
            for i in range(N):
                theta, phi, lamb = angles_u3[i * 3], angles_u3[i * 3 + 1], angles_u3[i * 3 + 2]
                circuit.rz(i, theta=phi)
                circuit.ry(i, theta=theta)
                circuit.rz(i, theta=lamb)

            for i in range(N):
                for j in range(i + 1, N):
                    two_qubit_gate_tc(circuit, angles_2q[i, j], i, j)

        # Final layer of U3 gates
        for i in range(N):
            theta, phi, lamb = angles_u3[i * 3], angles_u3[i * 3 + 1], angles_u3[i * 3 + 2]
            circuit.rz(i, theta=phi)
            circuit.ry(i, theta=theta)
            circuit.rz(i, theta=lamb)

        # Measurement
        bits, _ = circuit.measure(*range(N))
        return jnp.array(bits)  # Ensure output is JAX-compatible

    return Trotter_circuit_tensorcircuit  # Return optimized JIT-compiled function

def main(N, sample_size):
    k = 24
    
    s = np.random.choice([1., -1.], size=N)
    
    angles_u3 = np.random.uniform(0, 2 * np.pi, 3 * N)
    angles_2q = np.random.uniform(0, 2 * np.pi, (N, N))
    
    # Precompute circuit with fixed angles
    trotter_circuit_fn = make_trotter_circuit(N, k, angles_u3, angles_2q)
    
    tim = time.time()
    for idx in range(sample_size):
        angles_ry = backend.convert_to_tensor(np.pi * (s + 1) / 2)
        bits = trotter_circuit_fn(angles_ry)  # JIT-compiled function with only variable input
        s = bits * 2 - 1 
    
    print("Sampling Time: ", time.time()-tim)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('--sample_size', type=int, default=100)

    args = parser.parse_args()
    main(args.N, args.sample_size)
    