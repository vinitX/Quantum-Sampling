import netket as nk
import numpy as np

from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh

import argparse

def load_hamiltonian(N, ham_file): 
  ham = np.loadtxt(ham_file, dtype=complex)

  lmbd = None
  if N <= 10:
    ham_csr = csr_array(ham)
    lmbd, _ = eigsh(ham_csr, k=1, which='SA')

    print("Ground State:", lmbd)
    
  return ham, lmbd


def main(): 
    ## Write code to input parameters from user and parse them
    parser = argparse.ArgumentParser(description='RBM Ground State Finder')
    parser.add_argument('--N', type=int, help='Number of visible units')
    parser.add_argument('--ham_file', type=str, help='File path to hamiltonian matrix file')
    parser.add_argument('--epochs', type=int, default=40, help='Total number of epochs')
    parser.add_argument('--data_dir', type=str, default='Data/', help='File path for saving data. Example: Data/LiCoO_k=33/')
    parser.add_argument('--alpha', type=float, default=1, help='Density of hidden units. M = alpha * N')
    parser.add_argument('--beta', type=float, default=1.0, help='Inverse temperature')
    #parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sample size for Monte Carlo sampling')    
    parser.add_argument("--n_chains", type=int, default=16, help="Number of Markov chains (samples split across chains)")
    parser.add_argument("--lr", type=float, default=6e-3, help="Adam learning rate")

    args = parser.parse_args()
    print(args)

    N = args.N
    sample_size = args.sample_size
    ham_file = args.ham_file

    H_mat, lmbd = load_hamiltonian(N, ham_file)
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