import tensorcircuit as tc
import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

tc.set_backend("jax")
tc.set_contractor("greedy")
tc.set_dtype("complex64")

def two_qubit_gate_tc(circuit, angle: float, qubit_1: int, qubit_2: int):
    circuit.cnot(qubit_1, qubit_2)
    circuit.rz(qubit_2, theta=angle)
    circuit.cnot(qubit_1, qubit_2)
    return circuit

def make_trotter_circuit(N, k, angles_u3, angles_2q):
    # Convert fixed parameters to JAX arrays
    angles_u3_jax = jnp.array(angles_u3, dtype=jnp.float32)
    angles_2q_jax = jnp.array(angles_2q, dtype=jnp.float32)
    
    # Define circuit with fixed parameters and variable initial state
    @partial(tc.backend.jit, static_argnames=["N", "k"])
    def _trotter_circuit(angles_ry):
        c = tc.Circuit(N)
        
        # Initial state preparation
        for i in range(N):
            c.ry(i, theta=angles_ry[i])
        
        # Fixed circuit layers
        for _ in range(k - 1):
            # Single-qubit rotations
            for i in range(N):
                theta = angles_u3_jax[i*3]
                phi = angles_u3_jax[i*3+1]
                lamb = angles_u3_jax[i*3+2]
                c.rz(i, theta=phi)
                c.ry(i, theta=theta)
                c.rz(i, theta=lamb)
            
            # Two-qubit gates
            for i in range(N):
                for j in range(i+1, N):
                    c = two_qubit_gate_tc(c, angles_2q_jax[i,j], i, j)
        
        # Final layer
        for i in range(N):
            theta = angles_u3_jax[i*3]
            phi = angles_u3_jax[i*3+1]
            lamb = angles_u3_jax[i*3+2]
            c.rz(i, theta=phi)
            c.ry(i, theta=theta)
            c.rz(i, theta=lamb)
        
        return c.measure(*range(N), with_prob=True)[0]
    
    return _trotter_circuit

def main(N, sample_size):
    k = 24
    
    # Initialize random parameters
    angles_u3 = np.random.uniform(0, 2*np.pi, 3*N).astype(np.float32)
    angles_2q = np.random.uniform(0, 2*np.pi, (N, N)).astype(np.float32)
    
    # Initialize state parameters
    s = np.random.choice([1., -1.], size=N)
    angles_ry = (np.pi * (s + 1) / 2).astype(np.float32)
    
    # Compile circuit
    trotter_circuit_fn = make_trotter_circuit(N, k, angles_u3, angles_2q)
    
    # Warmup run
    tm = time.time()
    _ = trotter_circuit_fn(jnp.array(angles_ry)).block_until_ready()
    print(time.time()-tm)
    
    # Timing
    tim = time.time()
    for idx in range(sample_size):
        angles_ry = jnp.array(np.pi * (s + 1) / 2, dtype=jnp.float32)
        bits = trotter_circuit_fn(angles_ry).block_until_ready()
        s = bits * 2 - 1
    
    print(f"Sampling Time ({sample_size} runs): {time.time() - tim:.2f}s")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('--sample_size', type=int, default=100)

    args = parser.parse_args()
    main(args.N, args.sample_size)
