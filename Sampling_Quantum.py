import numpy as np
#from MCMC_Proposals_cudaq import *
from cudaq import spin
import scipy
from sampling_circuits_cudaq import *

def Euler_angle_decomposition(U:np.ndarray):
    theta = 2 * np.arccos(np.abs(U[0, 0]))

    phi = np.angle(U[1, 0]) - np.angle(U[0, 0])
    lam = np.angle(U[1, 1]) - np.angle(U[1, 0])
    
    return theta, phi, lam

def spin_to_key_nv(s):
    N=len(s)
    key=0
    for i in range(len(s)):
      key+=(2**i)*(1-s[N-i-1])/2
    return int(key)

def key_to_spin_nv(key, N):
    s=bin(key)[2:]
    s=(N-len(s))*'0' + s
    s=np.array([1-2*int(x) for x in s])
    return s

def spin_to_key(s):
    N=np.shape(key)[1]
    key=np.zeros(len(s),dtype=int)

    for i in range(N):
      key+=(2**i)*(1-s[:,N-i-1])//2
    return key

def key_to_spin(key, N):
    s = np.zeros((len(key),N),dtype=int)
    for i in range(N):
      s[:,N-i-1] = 1 - 2*(key%2)
      key = key//2
    return s

def enum(N):
    return key_to_spin(np.arange(2**N), N)

def Energy_Ising(s, N, poly):
    E = poly[0] * np.ones(len(s))
    E +=  s @ poly[1:1+N]
    E += np.reshape(np.einsum('ki,kj->kij',s,s,optimize='optimal'),(-1,N*N)) @ poly[1+N:]
    return E

def prob_Ising(s, N, poly, log_rho_max=1):
    E = Energy_Ising(s, N, poly)
    return np.exp(E - log_rho_max)


def prob_Ising_nv(s, N, poly, log_rho_max=1):
    y_pred = poly[0]
    y_pred +=  np.dot(s,poly[1:1+N])
    y_pred += np.dot(np.reshape(np.outer(s,s),-1), poly[1+N:])

    return np.exp(y_pred - log_rho_max)
    

def computing_norm_ratio(N,model_instance_one_body,model_instance_two_body):  #This gives value of alpha = self.computing_norm_ratio()
    #This computes alpha = ||H_x||_F/||H_z||_F using params only. No storing and computing full matrix is necessary
    #Coupling_matrix = np.reshape(np.array(self.model_instance_two_body), (self.no_spins, self.no_spins), order='F')
    alpha = np.sqrt(N)/np.sqrt(sum([J**2 for J in model_instance_two_body[np.tril_indices(N, k = -1)]]) + sum([h**2 for h in model_instance_one_body]))
    return alpha

def compute_angles(poly, N, time_delta=0.5, gamma = 0.42):
    l = poly[1:1+N]
    J = np.reshape(poly[1+N:],(N,N))
       
    alpha = computing_norm_ratio(N,l,J)
    #time_array, time_delta_step = self.scalar_time_sampling(sampling_type="discrete")
    #gamma_array, gamma_step = self.scalar_gamma_sampling(sampling_type="discrete")

    angle_list = []
    for qubit in range(N):
        coeff = -alpha*(1-gamma)*l[N-1-qubit]
        one_body_Ham = gamma * spin.x(0) + coeff * spin.z(0)
        angle_list.append(list(Euler_angle_decomposition(scipy.linalg.expm(-1.0j*time_delta*one_body_Ham.to_matrix()))))   # always 2*2 so no problem of exponentiation, storage

    theta = np.zeros(N)
    phi = np.zeros(N)
    lam = np.zeros(N)
    for qubit in range(N):
        theta[qubit], phi[qubit], lam[qubit] = angle_list[qubit]

    angles_u3 = np.concatenate((theta,phi,lam))

    angles_2q = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            angles_2q[i,j] = 2*J[N-1-i, N-1-j]*(1-gamma)*alpha*time_delta

    return angles_u3, angles_2q


def dict_to_res(counts):
  for key, value in counts.items():
    if value == 1: 
        final_config = key

  res = [1.0 if s == '1' else -1.0 for s in final_config]
  
  return np.array(res)


def Learner_Ham(N, poly, gamma:float, type_of_Ham="with_mixer"): 
    #Technically not required, only useful if prop of H(v) is studied
    #This makes the full learner Ham with mixer i.e. (1-gamma)*alpha*H_z + gamma*H_x  or without mixer where we just have H_z
    # where H_z = H(v) = \sum_i h_i v_i + \sum_{ij} Q_ij v_i v_j with v_i replaced with sigma_z
    #H_x = mixer = \sum_i X_i
    #alpha is the norm ratio of ||H_x||_F/||H_z||_F
    #gamma is unif(gamma1, gamma2). The paper says gamma1 as 0.2 and gamma2 as 0.6
    #CAUTION - USE this function for SMALL qubit size (n) ONLY as it involves exp storage of matrix size 2**n * 2**n when converted to np.array.
    # As a SparsePaulilist object, as I have defined below, it is fine to use it even for higher number of qubits

    l = poly[1:1+N]
    J = np.reshape(poly[1+N:],(N,N))

    alpha = computing_norm_ratio(N,l,J)

    Ham=0

    for i in range(N):
        for j in range(N):
          # Qiskit follows endian order with least sig bit as qubit[0] which is why we have (no_spins-1-index)
          coef = (1-gamma)*alpha*(J[N-1-i, N-1-j])
          Ham += coef * spin.z(i)*spin.z(j)  
                            
    for i in np.arange(N):
        # Qiskit follows endian order with least sig bit as qubit[0] which is why we have (no_spins-1-index)
        coef = -(1-gamma)*alpha*l[N-1-i]
        Ham += coef * spin.z(i)
        
        if type_of_Ham == "with_mixer":
            Ham += gamma * spin.x(i)

    return Ham


def get_transition_matrix_from_proposal(N, Proposal_mat, Energy, acceptance_criteria='metropolis', beta=1):
    # This function gets the full transition matrix P(s'|s) = Q(s'|s) * Acceptance(s'|s) where Q(s'|s)
    #can be a quantum circuit proposal, a local flip proposal, uniform proposal or Haar random proposal
    import math
    

    E_rowstack = np.tile(Energy, (2**N,1))  # E_rowstack[i,j] = E_j for all i
    E_diff = E_rowstack.T - E_rowstack # E_diff[i,j] = E_i - E_j (new E minus old E)

    uphill_moves = (E_diff >= 0) # includes isoenergetic moves
    #downhill_moves = np.invert(uphill_moves)

    if acceptance_criteria =='metropolis':
        if beta > 0:
            A_s_sp = np.exp(-E_diff*beta, where=uphill_moves, out=np.ones_like(E_diff)) #only compute exp for uphill moves, downhill ones are filled with 1 anyway
        if beta == math.inf:
        # reject all uphill, accept all others (level and downhill)
            A_s_sp = np.where(uphill_moves, 0.0, 1.)

    Transition_mat = np.multiply(A_s_sp, Proposal_mat)  #note not np.dot but elem wise multiplication
    np.fill_diagonal(Transition_mat, 0)
    diag = np.ones(2**N) - np.sum(Transition_mat, axis=0) # This step just fills the diag elems with 1-row_sum. This ensures that row_sum=1 for Transition mat
    Transition_mat = Transition_mat + np.diag(diag)
    return Transition_mat




def Sampling_Quantum(N, poly, sample_size, tot_time=12, time_delta=0.5, gamma=0.42, beta=1, burn=1000, init_config=[], compute_proposal_matrix=False, mode='Exact'):
  angles_u3, angles_2q = compute_angles(poly, N, time_delta, gamma)
  k = int(tot_time / time_delta)

  if len(init_config)==0:
      s = np.random.choice([1.,-1.],size=N)
  else: s = init_config

  if compute_proposal_matrix == False:
    prob_dict = {}
    key_list = []

    for _ in range(burn):
      angles_ry = np.pi*(s + 1)/2
      counts = cudaq.sample(Trotter_circuit, N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)
      s = dict_to_res(counts)
  
    print("Burn Complete!")

    tm = time.time()
    for _ in range(sample_size):
      angles_ry = np.pi*(s + 1)/2
      counts = cudaq.sample(Trotter_circuit, N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)
      s = dict_to_res(counts)
      
      key = spin_to_key_nv(s)
      if key in prob_dict: prob_dict[key] +=1
      else: prob_dict[key] = 1
      key_list.append(key)

    print("Sampling Time: ", time.time()-tm)

    for key in prob_dict.keys():
      prob_dict[key] = prob_dict[key] / sample_size

    return prob_dict, key_list     
     

  elif compute_proposal_matrix == True: 
    if mode == "Exact":
      full_Ham_mat = Learner_Ham(N, poly, gamma, type_of_Ham="with_mixer").to_matrix()  
      U_t = scipy.linalg.expm(-1.0j*tot_time*full_Ham_mat)

    elif mode == "Trotter":
      U_t = np.zeros((2**N, 2**N), dtype=np.complex128)

      for key in range(2**N):
          spin = key_to_spin_nv(key, N)
          angles_ry = np.pi*(spin + 1)/2

          U_t[:, key] = np.array(cudaq.get_state(
                        Trotter_circuit, N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1)), copy=False)

    Proposal_mat =  np.real(np.conjugate(U_t) * U_t)

    s = enum(N)
    Energy = Energy_Ising(s, N, poly)
    Transition_matrix = np.abs(get_transition_matrix_from_proposal(N, Proposal_mat, Energy, acceptance_criteria='metropolis', beta=beta))

    key = spin_to_key_nv(init_config)
    for _ in range(burn): 
      key = np.random.choice(np.arange(2**N), p=Transition_matrix[:,key])
    print("Burn Complete!")

    prob_dist = np.zeros(2**N)
    tm = time.time()
    for _ in range(sample_size):
      key = np.random.choice(np.arange(2**N), p=Transition_matrix[:,key])
      prob_dist[key] += 1
    print("Sampling Time: ", time.time()-tm)

    return prob_dist
 

    

  