from Sampling_Quantum import *
from MCMC_Proposals import *
from Sampling_Circuits import *

import numpy as np
import time

def init(seed,N,M,D):
  np.random.seed(seed)
  X=np.random.randn(N+M+N*M+N*D+D+N*N)+1j*np.random.randn(N+M+N*M+N*D+D+N*N)
  X=np.concatenate((np.real(X),np.imag(X)))
  return X


# def generate_samples(N, k, init_config, angles_u3, angles_2q): 
#   angles_ry = np.pi*(init_config + 1)/2

#   counts = cudaq.sample(Trotter_circuit, N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)
#   #counts = Trotter_circuit_builder(N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)

#   for key, value in counts.items():
#       if value == 1: 
#           final_config = key

#   res = [1.0 if s == '1' else -1.0 for s in final_config]
  
#   return np.array(res)

def dict_to_res(counts):
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
  
  tim = time.time()
  prob_dict = {}
  sample_list = []

  if len(init_config)==0:
    s = np.random.choice([1.,-1.],size=N)
  else: s = init_config

  angles_u3, angles_2q = Proposal_object.calculate_angles()

  for k in range(burn):
    angles_ry = np.pi*(s + 1)/2
    #counts = cudaq.sample(Trotter_circuit, N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)
    counts = Trotter_circuit_builder(N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)
    s = dict_to_res(counts)
  
  for k in range(sample_size):
    angles_ry = np.pi*(s + 1)/2
    #counts = cudaq.sample(Trotter_circuit, N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)
    counts = Trotter_circuit_builder(N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)
    s = dict_to_res(counts)
    
    key = Proposal_object.get_spinconfig_to_int(s)
    if key in prob_dict: prob_dict[key] +=1
    else: prob_dict[key] = 1
    sample_list.append(s)

  print("Sampling Time: ", time.time()-tim)
  
  prob_dict_flipped = {}
  for key in prob_dict.keys():
    prob_dict_flipped[2**N - key - 1] = prob_dict[key] / sample_size

  return np.flip(prob_dict_flipped), sample_list     #This flip is required to make peace with the difference in the convention.



def main(N, sample_size, seed, dir):
    X=init(seed,N,M=N,D=0)/40
    smpl = Sampling_Quantum_vectorized(X,N,M=N,D=0,beta=1)

    prob_dict, sample_list = Sampling(smpl, sample_size, burn=sample_size//10)

    np.save(dir+'TFIM_samples_mode=kernel_N='+str(N)+'.npy', sample_list)

    import pickle
    with open(dir+'TFIM_prob_dict_N='+str(N)+'.pkl', 'wb') as f:
        pickle.dump(prob_dict, f)


import argparse
if __name__ == "__main__":

    Trotter_circuit.compile()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dir', type=str, default='Data/')

    args = parser.parse_args()
    tm=time.time()
    main(args.N, args.sample_size, args.seed, args.dir)
    print("Total Time: ", time.time()-tm)