import numpy as np
from MCMC_Proposals_cudaq import *

def spin_to_key_nv(s):
    N=len(s)
    key=0
    for i in range(len(s)):
      key+=(2**i)*(1-s[N-i-1])/2
    return int(key)

def spin_to_key_nv(s):
    N=len(s)
    key=0
    for i in range(len(s)):
      key+=(2**i)*(1-s[N-i-1])/2
    return int(key)

def key_to_spin(key,N):
    s = np.zeros((len(key),N),dtype=int)
    for i in range(N):
      s[:,N-i-1] = 1 - 2*(key%2)
      key = key//2
    return s

def spin_to_key(s,N):
    key=np.zeros(len(s),dtype=int)

    for i in range(N):
      key+=(2**i)*(1-s[:,N-i-1])//2
    return key

def enum(N):
    return key_to_spin(np.arange(2**N))

def prob_RBM(s,params,beta=1):
    a,b,w = params
    M = len(b)

    def f(s):
        b_vec = np.broadcast_to(b, (len(s), M))
        return beta*(s@w + b_vec)
    
    return np.prod((np.abs(np.cosh(f(s)))**2), axis=1) * np.exp(-2*beta*(s@np.real(a))) 


def prob_RBM_nv(s,params,beta=1):
    a,b,w = params

    def f_nv(s):
        return beta*(s@w+b)
    
    return np.prod(np.abs(np.cosh(f_nv(s)))**2) * np.exp(-2*beta*(np.dot(s,np.real(a))))


def prob_Ising(s,N,params):
    y_pred = params[0] * np.ones(len(s))
    y_pred +=  s @ params[1:1+N]
    y_pred += np.reshape(np.einsum('ki,kj->kij',s,s,optimize='optimal'),(-1,N*N)) @ params[1+N:]

    return y_pred


def prob_Ising_nv(s, N, poly, log_rho_max=1):
    y_pred = poly[0]
    y_pred +=  np.dot(s,poly[1:1+N])
    y_pred += np.dot(np.reshape(np.outer(s,s),-1), poly[1+N:])

    return np.exp(y_pred - log_rho_max)


def sampler(s,prob_func,algo='Metropolis_uniform'):
    N=len(s)
    if algo=='Metropolis_uniform':
        p1 = prob_func(s)

        s_new = np.random.choice([1,-1],size=N)
        p2 = prob_func(s_new)

        accept = min(1.0,p2/p1)

        if np.random.rand()<accept:
            return s_new
        else: return s


def Sampling(N,sample_size=1000,burn=None,algo='Metropolis_uniform'):
    if algo=='Exact' and 2**N < sample_size:
      s = enum(N)
      prob_dist = prob_RBM(s)
      prob_dist = prob_dist / np.sum(prob_dist)
      samples = np.random.choice(np.arange(2**N), size=sample_size, p=prob_dist)
      prob_mat, _ = np.histogram(samples, bins=np.arange(2**N+1))
      return prob_mat/sample_size

    s=np.random.choice([1,-1],size=N)

    if burn is None:
      burn = sample_size//10

    #tm=time.time()
    for k in range(burn):
      s = sampler(s,algo='Metropolis_uniform')
    #print("\n#Burn Complete \n\n")
    #print("\t\t\t\t\t\t Burn Time: ", time.time()-tm)

    #tm=time.time()
    if 2**N < sample_size:
      prob_mat = np.zeros(2**N)

      for k in range(sample_size):
        s = sampler(s,'Metropolis_uniform')
        prob_mat[spin_to_key_nv(s)]+=1
      prob_mat = prob_mat / np.sum(prob_mat)
      return prob_mat

    else: 
      prob_dict = {}

      for k in range(sample_size):
        s = sampler(s,'Metropolis_uniform')
        key = spin_to_key_nv(s)
        if key in prob_dict: prob_dict[key]+=1
        else: prob_dict[key]=1
      
      for key in prob_dict.keys():
        prob_dict[key] = prob_dict[key] / sample_size


      return prob_dict
    #print("\t\t\t\t\t\t Metropolis Sampling: ", time.time()-tm)


def computing_norm_ratio(N,model_instance_one_body,model_instance_two_body):  #This gives value of alpha = self.computing_norm_ratio()
    #This computes alpha = ||H_x||_F/||H_z||_F using params only. No storing and computing full matrix is necessary
    #Coupling_matrix = np.reshape(np.array(self.model_instance_two_body), (self.no_spins, self.no_spins), order='F')
    alpha = np.sqrt(N)/np.sqrt(sum([J**2 for J in model_instance_two_body[np.tril_indices(N, k = -1)]]) + sum([h**2 for h in model_instance_one_body]))
    return alpha

def compute_angles(poly,N):
    l = poly[1:1+N]
    J = poly[1+N:]
       
    alpha = computing_norm_ratio()
    #time_array, time_delta_step = self.scalar_time_sampling(sampling_type="discrete")
    time_delta = 0.5
    #gamma_array, gamma_step = self.scalar_gamma_sampling(sampling_type="discrete")
    gamma = 0.42
    tot_time = 12

    k=int(tot_time/time_delta)


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

    angles_u3 = np.concatenate(theta,phi,lam)

    angles_2q = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            angles_2q[i,j] = 2*J[(N-1-i)*N + N-1-j]*(1-gamma)*alpha*time_delta

    return angles_u3, angles_2q


def Sampling_Quantum(N, poly, sample_size, beta=1, burn=1000, method='Quantum', init_config=None, compute_proposal_matrix=True):
  one_body_coeffs = -poly[1:1+N]     #np.random.rand(n)
  two_body_coeffs = -poly[1+N:]      #np.random.rand(n**2)

  if compute_proposal_matrix == False:
     
    if init_config==None:
        s = np.random.choice([1,-1],size=N)
    else: s = init_config

    for k in range(burn):
        s = Proposal_object.generate_MCMC_trajectories(s, Transition_matrix)

    for k in range(sample_size):
        s = Proposal_object.generate_MCMC_trajectories(s, Transition_matrix)
        key = Proposal_object.get_spinconfig_to_int(s)
        prob_dist[key] +=1
        key_list.append(key)

    return np.flip(prob_dist/np.sum(prob_dist)), key_list     #This flip is required to make peace with the difference in the convention.
    #err_hist
     

  elif compute_proposal_matrix == True: 
    Proposal_object = All_proposals(inverse_temp=beta, one_body_coeffs=one_body_coeffs,
              two_body_coeffs = two_body_coeffs)
    
    Proposal_matrix = Proposal_object.get_quantum_circuit_proposal_matrix(mode="Trotter error/no Sampling error")
    #Proposal_matrix_exact = Proposal_object.get_quantum_circuit_proposal_matrix(mode="Exact-no Trotter error/no Sampling error")

    Proposal_object.get_energy_array().Energy_array
    Transition_matrix = np.real(Proposal_object.get_transition_matrix_from_proposal(Proposal_matrix, acceptance_criteria='metropolis'))

    prob_dist = np.zeros(2**N)

    exact_dist = np.exp(-beta * Proposal_object.Energy_array)
    exact_dist = exact_dist / np.sum(exact_dist)

    key_list = []

  