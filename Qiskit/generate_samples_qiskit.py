from RBM_surrogate import *
from MCMC_Proposals import *


import numpy as np
import time

# from qiskit_aer import Aer
# backend = Aer.get_backend('qasm_simulator')
# transpiled_circuit = transpile(Trotter_circuit_qiskit, backend)


def init(seed,N,M,D):
  np.random.seed(seed)
  X=np.random.randn(N+M+N*M+N*D+D+N*N)+1j*np.random.randn(N+M+N*M+N*D+D+N*N)
  X=np.concatenate((np.real(X),np.imag(X)))
  return X


def generate_samples(N, k, s, angles_u3, angles_2q): 
  angles_ry = np.pi*(s + 1)/2

  circuit = Trotter_circuit_qiskit(N, k, angles_ry, angles_u3, angles_2q)
  circuit.measure_all()
  simulator = AerSimulator()
  result = simulator.run(circuit, shots=1).result()
  counts = result.get_counts(circuit)

  for key, value in counts.items():
      if value == 1: 
          final_config = key

  res = [1.0 if s == '1' else -1.0 for s in final_config]
  
  return np.array(res)
  

def Sampling(smpl, sample_size, burn, init_config=[], compute_transition=False):
  N = smpl.N
  one_body_coeffs = -smpl.poly[1:1+N]     #np.random.rand(n)
  two_body_coeffs = -smpl.poly[1+N:]      #np.random.rand(n**2)
  beta=1

  time_delta = 0.5
  total_time = 12
  k = int(total_time / time_delta)

  Proposal_object = All_proposals(inverse_temp=beta, one_body_coeffs=one_body_coeffs,
              two_body_coeffs = two_body_coeffs)
  
  tm = time.time()
  prob_dict = {}
  sample_list = []

  if len(init_config)==0:
    s = np.random.choice([1.,-1.],size=N)
  else: s = init_config

  angles_u3, angles_2q = Proposal_object.calculate_angles()

  for _ in range(burn):
    s = generate_samples(N, k, s, angles_u3, angles_2q)

  for _ in range(sample_size):
    s = generate_samples(N, k, s, angles_u3, angles_2q)
    key = Proposal_object.get_spinconfig_to_int(s)
    if key in prob_dict: prob_dict[key] +=1
    else: prob_dict[key] = 1
    sample_list.append(s)

  print("Sampling Time: ", time.time()-tm)
  
  prob_dict_flipped = {}
  for key in prob_dict.keys():
    prob_dict_flipped[2**N - key - 1] = prob_dict[key] / sample_size

  return np.flip(prob_dict_flipped), sample_list     #This flip is required to make peace with the difference in the convention.



def main(N, sample_size, seed, dir):
    X=init(seed,N,M=N,D=0)/40
    smpl = Sampling_Quantum_vectorized(X,N,M=N,D=0,beta=1)

    prob_dict, sample_list = Sampling(smpl, sample_size, burn=sample_size//10)

    np.save(dir+'TFIM_samples_mode=qiskit_N='+str(N)+'.npy', sample_list)

    import pickle
    with open(dir+'TFIM_prob_dict_N='+str(N)+'.pkl', 'wb') as f:
        pickle.dump(prob_dict, f)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dir', type=str, default='Data/')

    args = parser.parse_args()
    tm=time.time()
    main(args.N, args.sample_size, args.seed, args.dir)
    print("Total Time: ", time.time()-tm)