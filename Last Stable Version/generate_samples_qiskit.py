from Sampling_Quantum import *
from MCMC_Proposals_qiskit import *

import numpy as np
import time

def init(seed,N,M,D):
  np.random.seed(seed)
  X=np.random.randn(N+M+N*M+N*D+D+N*N)+1j*np.random.randn(N+M+N*M+N*D+D+N*N)
  X=np.concatenate((np.real(X),np.imag(X)))
  return X


def Sampling(smpl, sample_size, burn=1000, mode='', method='Quantum', init_config=[], compute_transition=False):
  N = smpl.N
  one_body_coeffs = -smpl.poly[1:1+N]     #np.random.rand(n)
  two_body_coeffs = -smpl.poly[1+N:]      #np.random.rand(n**2)
  beta=1

  Proposal_object = All_proposals(inverse_temp=beta, one_body_coeffs=one_body_coeffs,
              two_body_coeffs = two_body_coeffs)
  
  tm = time.time()
  prob_dict = {}
  sample_list = []

  if len(init_config)==0:
    s = np.random.choice([1.,-1.],size=N)
  else: s = init_config

  for k in range(burn):
    s = Proposal_object.generate_MCMC_trajectories(s, mode)

  for k in range(sample_size):
    s = Proposal_object.generate_MCMC_trajectories(s, mode)
    key = Proposal_object.get_spinconfig_to_int(s)
    if key in prob_dict: prob_dict[key] +=1
    else: prob_dict[key] = 1
    sample_list.append(s)


  print("Sampling Time: ", time.time()-tm)
  
  prob_dict_flipped = {}
  for key in prob_dict.keys():
    prob_dict_flipped[2**N - key - 1] = prob_dict[key] / sample_size

  return np.flip(prob_dict_flipped), sample_list     #This flip is required to make peace with the difference in the convention.



def main(N, mode, sample_size, seed, dir):
    X=init(seed,N,M=N,D=0)/40
    smpl = Sampling_Quantum_vectorized(X,N,M=N,D=0,beta=1)

    prob_dict, sample_list = Sampling(smpl, sample_size, mode=mode, burn=sample_size//10)

    np.save(dir+'TFIM_samples_mode='+mode+'_N='+str(N)+'.npy', sample_list)

    import pickle
    with open(dir+'TFIM_prob_dict_N='+str(N)+'.pkl', 'wb') as f:
        pickle.dump(prob_dict, f)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='The system size')
    parser.add_argument('mode', type=str, help='\{kernel, builder, or qiskit\}')
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dir', type=str, default='Data/')

    args = parser.parse_args()
    tm=time.time()
    main(args.N, args.mode, args.sample_size, args.seed, args.dir)
    print("Total Time: ", time.time()-tm)