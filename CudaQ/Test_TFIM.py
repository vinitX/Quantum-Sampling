import cudaq
from cudaq import spin

from RBM_surrogate import *
from CudaQ.get_conn import get_conn
from Sampling import *
from CudaQ.Sampling_Quantum import *
#from Sampling_MPO import *
from Adam import Adam

import numpy as np
import time
# import matplotlib.pyplot as plt
from functools import partial

N=2
M=1
D=0
beta=1


def TFIM(N,g):
    hamiltonian = 0
    for i in range(N-1):
        hamiltonian -= spin.z(i) * spin.z(i+1)

    for i in range(N):
        hamiltonian -= g*spin.x(i)
    return hamiltonian

Ham = TFIM(N,1.0)

# Adam optimizer hyperparameters
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

file = "Data/"

from scipy.sparse import csr_array, coo_matrix
from scipy.sparse.linalg import eigsh

Ham_sp = Ham.to_sparse_matrix()
data, row, col = Ham_sp
Ham_coo = coo_matrix((data, (row, col)), shape=(2**8, 2**8))
Ham_csr = csr_array(Ham_coo)

lmbd, _ = eigsh(Ham_csr, k=1, which='SA')

print("Smallest eigenvalue:", lmbd)


def plot_log(E_hist, E_smpl_hist):
    #ya, iter = smpl.median_filter(np.real(E_hist))
    ya = np.real(E_hist)
    xa = np.arange(len(ya))
    ci = np.abs(np.array(E_smpl_hist) - ya)
    fig, ax = plt.subplots()
    ax.plot(ya)
    ax.fill_between(xa, (ya-ci), (ya+ci), color='r', alpha=.5)
    ax.set_ylabel("Energy")
    ax.set_xlabel("Epochs")
    plt.hlines(lmbd[0],0,len(xa)-1,color='r',linestyles='dashed')
    plt.show()


def save_data(X,smpl,g,seed,E,E_smpl,prob_dist,sample_list):
  with open(file+"X_N="+str(N)+"_g="+str(g)+"_seed="+str(seed)+".txt", "ab") as f:
    np.savetxt(f, np.reshape(X,(1,-1)))
  with open(file+"Poly_N="+str(N)+"_g="+str(g)+"_seed="+str(seed)+".txt", "ab") as f:
    np.savetxt(f, np.reshape(smpl.poly,(1,-1)))
  with open(file+"E_N="+str(N)+"_g="+str(g)+"_seed="+str(seed)+".txt", "ab") as f:
    np.savetxt(f, [E])
  with open(file+"E_smpl_N="+str(N)+"_g="+str(g)+"_seed="+str(seed)+".txt", "ab") as f:
    np.savetxt(f, [E_smpl])
  with open(file+"prob_dist_N="+str(N)+"_g="+str(g)+"_seed="+str(seed)+".txt", "ab") as f:
    np.savetxt(f, np.reshape(prob_dist,(1,-1)))
  with open(file+"prob_dist_N="+str(N)+"_g="+str(g)+"_seed="+str(seed)+".txt", "ab") as f:
    np.savetxt(f, np.reshape(sample_list,(1,-1)))



sample_size = 100

seed = 1

E_hist=[]
E_smpl_hist=[]

rbm = RBM_surrogate(N=N,M=M,seed=seed,)
#x = tf.Variable(rbm.X)
x = np.array(rbm.X, dtype=np.float32)

tmg=time.time()

optimizer = Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

sampling_method = "Uniform"

for epoch in range(40):
  rbm = RBM_surrogate(N,M,X=x)  #.numpy())
  rbm.build_surrogate()

  tm = time.time()
  
  if sampling_method == "Uniform":
    prob_func = partial(prob_Ising_nv, N=N, poly=rbm.poly, log_rho_max=rbm.log_rho_max)
    prob_dict = Sampling(N=N, prob_func=prob_func, sample_size=sample_size, burn=sample_size//10)
    
  elif sampling_method == "Quantum":
    prob_dict, _ = Sampling_Quantum(N=N, poly=rbm.poly, sample_size=sample_size, burn=sample_size//10)
  # elif sampling_method == "TN":
  #   prob_dict, _ = Sampling_MPO(N=N, poly=rbm.poly, sample_size=sample_size, burn=sample_size//10)

  print("\t\t\tSampling Time: ", time.time()-tm)

  E = rbm.Energy_exact(Ham)

  #tm = time.time()
  E_smpl = rbm.Energy_sampling(Ham,prob_dict=prob_dict)
  #print("Energy Time: ", time.time()-tm)
  print(E, E_smpl)
  #print(epoch, "Energy: ", E, "\t +/- \t", np.abs(E_smpl - E))

  #tm = time.time()
  #grad_exact = np.real(rbm.grad_exact(Ham))
  grad = np.real(rbm.grad_Sampling(Ham,prob_dict=prob_dict))
  #print("Gradient Time: ", time.time()-tm)

  optimizer.apply_gradients([(grad, x)])

  E_hist.append(E)
  E_smpl_hist.append(E_smpl)

print("\n\n\nEnergy: ",E,"\n Time",time.time()-tmg, lmbd[0])

# plot_log(E_hist, E_smpl_hist)