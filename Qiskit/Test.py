import numpy as np
import time
import argparse
import numpy as np
from qiskit_circuits import quantum_sampler
import pickle
from Sampling_Quantum import prob_Ising, prob_Ising_nv, spin_to_key_nv, computing_norm_ratio, enum

def generate_samples(N, poly, sample_size, tot_time=12, time_delta=0.5,
                      gamma=0.42, beta=1, burn=None, instance=None):
    k = int(tot_time / time_delta)

    l = poly[1:1+N]
    J = np.reshape(poly[1+N:],(N,N))
    alpha = computing_norm_ratio(N,l,J)

    angle_rx = gamma * time_delta
    angles_rz = l[::-1] * (1-gamma) * alpha * time_delta
    angles_2q = 2*J[::-1, ::-1] * (1-gamma) * alpha * time_delta

    s = np.random.choice([1,0],size=N)

    prob_dict = {}
    key_list = []

    if burn is None:
        burn = sample_size//10

    for _ in range(burn):
        # s_new = quantum_sampler(N, k, s, angle_rx, angles_rz, angles_2q)
        s_new = np.random.choice([1,0],size=N)
        p1 = prob_Ising_nv(s, N, poly)
        p2 = prob_Ising_nv(s_new, N, poly)

        if np.random.rand()<min(1.0,p2/p1): s = s_new 

    t0 = time.time()

    for _ in range(sample_size):
        # s_new = quantum_sampler(N, k, s, angle_rx, angles_rz, angles_2q)
        s_new = np.random.choice([1,0],size=N)

        p1 = prob_Ising_nv(s, N, poly)
        p2 = prob_Ising_nv(s_new, N, poly)

        if np.random.rand()<min(1.0,p2/p1): s = s_new     
        
        key = spin_to_key_nv(s)
        if key in prob_dict: prob_dict[key] +=1
        else: prob_dict[key] = 1
        key_list.append(key)

    t1 = time.time()

    for key in prob_dict.keys():
        prob_dict[key] = prob_dict[key] / sample_size

    prob_dict = dict(sorted(prob_dict.items()))

    print(f"Size: {N}  Time: {t1 - t0} sec")

    with open(f"quantum_prob_dict_{N}_{instance}.pkl", "wb") as f:
        pickle.dump(prob_dict, f)

    with open(f"quantum_key_list_{N}_{instance}.txt", "w") as f:
        for key in key_list:
            f.write(f"{key}\n")

    return prob_dict, key_list   


def main():
    parser = argparse.ArgumentParser(description="Run sequential QASM simulations with measurement feedback.")
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--trotter_steps", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--shots", type=int, default=1)
    args = parser.parse_args()

    N = args.num_qubits
    instance = 1
    k = args.trotter_steps
    sample_size = args.sample_size

    np.random.seed(instance)
    l = np.random.randn(N)
    J = np.diag(np.random.randn(N-1), k=1)
    J = (J + J.T) / 2

    poly = np.zeros(1+N+N*N)
    poly[0] = 0.0
    poly[1:1+N] = l
    poly[1+N:] = J.flatten()

    prob_dict, key_list = generate_samples(N, poly, sample_size, instance=instance)

    print(prob_dict)
    print(key_list)
    
    # calculate the expected_distribution
    s = enum(N)
    exact_prob = prob_Ising(s,N,poly)
    exact_prob /= np.sum(exact_prob)
    print(np.round(exact_prob,2))


if __name__ == "__main__":
    main()

