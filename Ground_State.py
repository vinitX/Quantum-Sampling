from Sampling import *
from RBM import *
from spin_utils import *
from Adam import Adam

import numpy as np
import time
import matplotlib.pyplot as plt
import json
from functools import partial
import re
import pickle

from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh

import argparse

def load_hamiltonian(N, ham_file): 
  ham = np.zeros((2**N,2**N),dtype=complex)

  with open(ham_file, 'r') as f:
      for line in f:
          # Match lines like: m= 1 n= 2 Re=0.0000000000 Im=0.0000000000
          match = re.match(r"m=\s*(\d+)\s*n=\s*(\d+)\s*Re=([-\d\.]+)\s*Im=([-\d\.]+)", line)
          if match:
              m, n, re_val, im_val = match.groups()
              ham[int(m)-1,int(n)-1] = float(re_val) + 1j * float(im_val)

  ham_pauli = hamiltonian_matrix_to_pauli_sum(ham)

  lmbd = None
  if N <= 10:
    ham_csr = csr_array(ham)
    lmbd, _ = eigsh(ham_csr, k=1, which='SA')

    print("Ground State:", lmbd)
    
  return ham_pauli, lmbd


def plot_log(E_hist=[], E_smpl_hist=[], lmbd=None):
    if len(E_smpl_hist) == 0:
        plt.plot(E_hist)
        plt.ylabel("Energy")
        plt.xlabel("Epochs")

    elif len(E_hist) == 0:
        plt.plot(E_smpl_hist)
        plt.ylabel("Energy")
        plt.xlabel("Epochs")

    else: 
        ya = np.real(E_hist)
        xa = np.arange(len(ya))
        ci = np.abs(np.array(E_smpl_hist) - ya)
        _, ax = plt.subplots()
        ax.plot(ya)
        ax.fill_between(xa, (ya-ci), (ya+ci), color='r', alpha=.5)
        ax.set_ylabel("Energy")
        ax.set_xlabel("Epochs")

    if lmbd:
        plt.hlines(lmbd[0],0,len(E_hist)-1,color='r',linestyles='dashed')


def save_data(data_dir,X,E=None,E_smpl=None): # ,epoch=0,prob_dict=None):
  with open(data_dir+"X.txt", "ab") as f:
    np.savetxt(f, np.reshape(X,(1,-1)))
  if E is not None:
    with open(data_dir+"E.txt", "ab") as f:
      np.savetxt(f, [E])
  if E_smpl is not None:
    with open(data_dir+"E_smpl.txt", "ab") as f:
      np.savetxt(f, [E_smpl])
    # with open(data_dir+"prob_dict/epoch_"+str(epoch)+".pkl", "wb") as f:
    #   pickle.dump(prob_dict, f)



# Adam optimizer hyperparameters
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

def main(): 
    ## Write code to input parameters from user and parse them
    parser = argparse.ArgumentParser(description='RBM Ground State Finder')
    parser.add_argument('--N', type=int, help='Number of visible units')
    parser.add_argument('--ham_file', type=str, help='File path to hamiltonian file')
    parser.add_argument('--method', type=str, default='sampling', help='sampling or exact')
    parser.add_argument('--epochs', type=int, default=40, help='Total number of epochs')
    parser.add_argument('--data_dir', type=str, default='Data/', help='File path for saving data. Example: Data/LiCoO_k=33/')
    parser.add_argument('--alpha', type=float, default=1, help='Density of hidden units. M = alpha * N')
    parser.add_argument('--beta', type=float, default=1.0, help='Inverse temperature')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sample size for Monte Carlo sampling')    

    args = parser.parse_args()
    print(args)

    N = args.N
    M = int(args.alpha * N)
    method = args.method
    epochs = args.epochs
    beta = args.beta
    seed = args.seed
    sample_size = args.sample_size
    ham_file = args.ham_file
    data_dir = args.data_dir


    ham_pauli, lmbd = load_hamiltonian(N, ham_file)

    E_hist=[]
    E_smpl_hist=[]

    rbm = RBM(N,M,seed=seed)
    X = rbm.X

    tm=time.time()

    optimizer = Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

    for epoch in range(epochs):
      rbm = RBM(N,M,X=X)

      params = [rbm.a, rbm.b, rbm.w]
      #tm = time.time()
      prob_func = partial(prob_RBM_nv, params=params)
      if method == "sampling":
        prob_dict = Sampling(N=N, prob_func=prob_func, sample_size=sample_size, burn=sample_size//10)
      #print("Sampling Time: ", time.time()-tm)

      if method == "exact":
        E = rbm.Energy_exact(ham_pauli)
        print(epoch, "Energy: ", E)
        grad = np.real(rbm.grad_exact(ham_pauli))
        E_hist.append(E)
        save_data(data_dir,X,E)
        

      elif method == "sampling":
        E_smpl = rbm.Energy_sampling(ham_pauli,prob_dict=prob_dict)
        grad = np.real(rbm.grad_Sampling(ham_pauli,prob_dict=prob_dict))
        E_smpl_hist.append(E_smpl)

        E = None
        if N < 10:
          E = rbm.Energy_exact(ham_pauli)
          print(epoch, "Energy: ", E, "\t +/- \t", np.abs(E_smpl - E))
          E_hist.append(E)
        save_data(data_dir,X,E,E_smpl) #,epoch,prob_dict)
        

      optimizer.apply_gradients([(grad, rbm.X)])

    if E: 
      print("\n\n\nEnergy: ",E,"\n Time",time.time()-tm)
    else: 
      print("\n\n\nEnergy: ",E_smpl,"\n Time",time.time()-tm)


    plot_log(E_hist, E_smpl_hist)


if __name__ == "__main__":
    main()