import netket as nk
import numpy as np
import jax
import re

from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh

import argparse

def load_hamiltonian(N, ham_file, kappa): 
  # Dilating Hamiltonian to have size (2^N, 2^N)
  # H_dil = H_org \osum kappa * I
  # kappa is the eigenvalue of the fill padded subspace. 
  # It needs to be larger than the eigenvalue of the ground state, 
  # if we want the learner to converge to the true ground state. 

  ham = kappa * np.eye(2**N,dtype=complex)

  with open(ham_file, 'r') as f:
      for line in f:
          # Match lines like: m= 1 n= 2 Re=0.0000000000 Im=0.0000000000
          match = re.match(r"m=\s*(\d+)\s*n=\s*(\d+)\s*Re=([-\d\.]+)\s*Im=([-\d\.]+)", line)
          if match:
              m, n, re_val, im_val = match.groups()
              ham[int(m)-1,int(n)-1] = float(re_val) + 1j * float(im_val)

  lmbd = None
  if N <= 10:
    ham_csr = csr_array(ham)
    lmbd, _ = eigsh(ham_csr, k=1, which='SA')

    print("Ground State:", lmbd)
    
  return ham, lmbd


# def save_data(data_dir,X,E=None,E_smpl=None): # ,epoch=0,prob_dict=None):
#   with open(data_dir+"X.txt", "ab") as f:
#     np.savetxt(f, np.reshape(X,(1,-1)))
#   if E is not None:
#     with open(data_dir+"E.txt", "ab") as f:
#       np.savetxt(f, [E])
#   if E_smpl is not None:
#     with open(data_dir+"E_smpl.txt", "ab") as f:
#       np.savetxt(f, [E_smpl])


def main(): 
    ## Write code to input parameters from user and parse them
    parser = argparse.ArgumentParser(description='RBM Ground State Finder')
    parser.add_argument('--N', type=int, help='Number of visible units')
    parser.add_argument('--ham_file', type=str, help='File path to hamiltonian file')
    parser.add_argument('--epochs', type=int, default=40, help='Total number of epochs')
    parser.add_argument('--data_dir', type=str, default='Data/', help='File path for saving data. Example: Data/LiCoO_k=33/')
    parser.add_argument('--alpha', type=float, default=1, help='Density of hidden units. M = alpha * N')
    parser.add_argument('--beta', type=float, default=1.0, help='Inverse temperature')
    #parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sample size for Monte Carlo sampling')    
    parser.add_argument('--kappa', type=float, default=0, help='The eigenvalue to the fill padded subspace.')
    parser.add_argument("--n_chains", type=int, default=16, help="Number of Markov chains (samples split across chains)")
    parser.add_argument("--lr", type=float, default=6e-3, help="Adam learning rate")

    args = parser.parse_args()
    print(args)

    N = args.N
    sample_size = args.sample_size
    ham_file = args.ham_file
    kappa = args.kappa

    H_mat, lmbd = load_hamiltonian(N, ham_file, kappa)
    print("Ground State energy: ", lmbd)

    hi = nk.hilbert.Spin(s=0.5, N=args.N)
    H = nk.operator.LocalOperator(hi, operators=[H_mat], acting_on=[list(range(N))])

    model = nk.models.RBM(alpha=args.alpha, use_visible_bias=True, param_dtype=complex)
    sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=args.n_chains)
    vstate = nk.vqs.MCState(sampler, model, n_samples=sample_size)
    optimizer = nk.optimizer.Adam(learning_rate=args.lr)
    precond = nk.optimizer.SR(diag_shift=1e-3)

    log = nk.logging.RuntimeLog()

    vmc = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=precond)

    print("Starting VMC...")
    vmc.run(n_iter=args.epochs, out=log, show_progress=True)

    print("Final ground state energy: ", vstate.expect(H))

if __name__ == "__main__":
    main()