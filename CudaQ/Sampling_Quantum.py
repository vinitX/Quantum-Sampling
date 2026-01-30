import numpy as np
from cudaq import spin
import cudaq
import scipy

#cudaq.set_target('tensornet')

def Euler_angle_decomposition(U:np.ndarray, eps=1e-4):
    # Pull off a global phase so that U00 is real ≥ 0:
    # gamma = arg(U00)
    u00 = U[0,0]
    gamma = np.angle(u00)
    U_phase = U * np.exp(-1j * gamma)

    # Now U_phase[0,0] = cos(theta/2) ∈ ℝ, ≥ 0
    c = np.real(U_phase[0,0])
    # Clip for numerical safety
    c = np.clip(c, -1.0, 1.0)
    theta = 2 * np.arccos(c)

    s = np.sin(theta/2)
    #print(r"sin($\theta$) = ", s)

    if s < eps:
        # We set φ=0, absorb any relative phase into λ:
        phi = 0.0
        # U11/U00 = exp(i(φ+λ)) ≈ exp(iλ)
        lam = np.angle(U_phase[1,1] / U_phase[0,0])
    else:
        # Standard case: extract φ and λ from off-diagonals
        # U_phase[1,0] = e^{iφ} sin(θ/2)  ⇒ φ = arg(...)
        phi = np.angle(U_phase[1,0] / s)
        # U_phase[0,1] = -e^{iλ} sin(θ/2) ⇒ λ = arg(-U01/s)
        lam = np.angle(-U_phase[0,1] / s)
    
    return theta, phi, lam


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """
    Compute the unitary matrix for the U3 gate with given angles.
    
    Args:
        theta: Rotation angle about the X-axis (radians)
        phi: Phase angle for first Z-rotation (radians)
        lam: Phase angle for second Z-rotation (radians)
    
    Returns:
        2x2 complex numpy array representing the U3 gate
    """
    # Calculate matrix components
    cos_theta = np.cos(theta/2)
    sin_theta = np.sin(theta/2)
    
    return np.array([
        [cos_theta, -np.exp(1j * lam) * sin_theta],
        [np.exp(1j * phi) * sin_theta, np.exp(1j * (phi + lam)) * cos_theta]
        ], dtype=np.complex128)


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
    E = 0  #poly[0] 
    E +=  s @ poly[1:1+N]

    J = poly[1+N:].reshape(N, N) 
    ssT = np.einsum('ki,kj->kij', s, s, optimize='optimal')
    upper_J = np.triu(J, k=1) 
    E += np.tensordot(ssT, upper_J, axes=([1, 2], [0, 1]))

    return E

def prob_Ising(s, N, poly, log_rho_max=0):
    E = Energy_Ising(s, N, poly)
    return np.exp(E - log_rho_max)


def prob_Ising_nv(s, N, poly, log_rho_max=0):
    s = s.reshape(1, len(s))
    return prob_Ising(s, N, poly, log_rho_max)
    

def computing_norm_ratio(N,model_instance_one_body,model_instance_two_body):  #This gives value of alpha = self.computing_norm_ratio()
    #This computes alpha = ||H_x||_F/||H_z||_F using params only. No storing and computing full matrix is necessary
    #Coupling_matrix = np.reshape(np.array(self.model_instance_two_body), (self.no_spins, self.no_spins), order='F')
    alpha = np.sqrt(N)/np.sqrt(sum([J**2 for J in model_instance_two_body[np.tril_indices(N, k = -1)]]) + sum([h**2 for h in model_instance_one_body]))
    return alpha

def compute_angles(poly, N, time_delta, gamma):
    l = poly[1:1+N]
    J = np.reshape(poly[1+N:],(N,N))
       
    alpha = computing_norm_ratio(N,l,J)
    #time_array, time_delta_step = self.scalar_time_sampling(sampling_type="discrete")
    #gamma_array, gamma_step = self.scalar_gamma_sampling(sampling_type="discrete")

    theta = np.zeros(N)
    phi = np.zeros(N)
    lam = np.zeros(N)

    for qubit in range(N):
        coeff = alpha*(1-gamma)*l[N-1-qubit]
        one_body_Ham = gamma * spin.x(0) + coeff * spin.z(0)
        U = scipy.linalg.expm(-1.0j*time_delta*one_body_Ham.to_matrix())
        theta[qubit], phi[qubit], lam[qubit] = Euler_angle_decomposition(U)  
        
        euler_error = 1 - np.abs(np.trace(u3_matrix(theta[qubit], phi[qubit], lam[qubit]) @ U.conj().T))/2
        if euler_error > 1e-8: 
          print("Euler Error: ", euler_error)
    angles_u3 = np.concatenate((theta,phi,lam))

    angles_2q = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            angles_2q[i,j] = 2*J[N-1-i, N-1-j]*(1-gamma)*alpha*time_delta

    return angles_u3, angles_2q



@cudaq.kernel
def two_qubit_gate(angle:float, qubit_1: cudaq.qubit, qubit_2: cudaq.qubit):  # mode: str = "CNOT_decomposition"  [cudaq doesn't support string type]
    x.ctrl(qubit_1, qubit_2)
    rz(angle, qubit_2)
    x.ctrl(qubit_1, qubit_2)

@cudaq.kernel
def Trotter_circuit(N: int, k:int, angles_ry:np.ndarray, angles_u3:np.ndarray, angles_2q:np.ndarray):  #list[int]
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
    # k : Trotter repeat length

    qreg=cudaq.qvector(N)

    for i in range(N):
        ry(angles_ry[i], qreg[i])

    for _ in range(k):
        for i in range(N):
            u3(angles_u3[i], angles_u3[i+N], angles_u3[i+2*N], qreg[i])

        for i in range(N):
            for j in range(i + 1, N): 
                two_qubit_gate(angles_2q[i*N+j], qreg[i], qreg[j])

    #mz(qreg)

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
        for j in range(i+1,N):
          # Qiskit/CudaQ follows endian order with least sig bit as qubit[0] which is why we have (no_spins-1-index)
          coef = (1-gamma)*alpha*(J[N-1-i, N-1-j])

          Ham += coef * spin.z(i)*spin.z(j)
                            
    for i in np.arange(N):
        # Qiskit/CudaQ follows endian order with least sig bit as qubit[0] which is why we have (no_spins-1-index)
        coef = (1-gamma)*alpha*l[N-1-i]
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
    
    downhill_moves = (E_diff <= 0) 
    
    if acceptance_criteria =='metropolis':
        if beta > 0:
            A_s_sp = np.exp(E_diff*beta, where=downhill_moves, out=np.ones_like(E_diff)) #only compute exp for uphill moves, downhill ones are filled with 1 anyway
        if beta == math.inf:
        # reject all downhill, accept all others 
            A_s_sp = np.where(downhill_moves, 0.0, 1.)

    Transition_mat = np.multiply(A_s_sp, Proposal_mat)  #note not np.dot but elem wise multiplication

    np.fill_diagonal(Transition_mat, 0)
    diag = np.ones(2**N) - np.sum(Transition_mat, axis=0) # This step just fills the diag elems with 1-row_sum. This ensures that row_sum=1 for Transition mat
    Transition_mat = Transition_mat + np.diag(diag)
    return Transition_mat


def dict_to_res(counts):
  for key, value in counts.items():
    if value == 1: 
        final_config = key

  res = [1.0 if s == '0' else -1.0 for s in final_config]
  
  return np.flip(res)


def quantum_sampler(N,poly,s,k,angles_u3,angles_2q):
    angles_ry = np.flip(np.pi*(1-s)/2)
    counts = cudaq.sample(Trotter_circuit, N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1), shots_count=1)
    s_new = dict_to_res(counts)

    p1 = prob_Ising_nv(s, N, poly)
    p2 = prob_Ising_nv(s_new, N, poly)

    accept = min(1.0,p2/p1)
    if np.random.rand()<accept: return s_new     
    else: return s


def Sampling_Quantum(N, poly, sample_size, tot_time=12, time_delta=0.5, gamma=0.42, beta=1, burn=None, init_config=[], compute_proposal_matrix=False, mode='Exact'):
  angles_u3, angles_2q = compute_angles(poly, N, time_delta, gamma)
  k = int(tot_time / time_delta)

  if len(init_config)==0:
      s = np.random.choice([1.,-1.],size=N)
  else: s = init_config

  if compute_proposal_matrix == False:
    prob_dict = {}
    key_list = []

    Trotter_circuit.compile()

    if burn is None:
      burn = sample_size//10
      
    for _ in range(burn):
      s = quantum_sampler(N,poly,s,k,angles_u3,angles_2q)
  
    for _ in range(sample_size):
      s = quantum_sampler(N,poly,s,k,angles_u3,angles_2q)
      
      key = spin_to_key_nv(s)
      if key in prob_dict: prob_dict[key] +=1
      else: prob_dict[key] = 1
      key_list.append(key)

    for key in prob_dict.keys():
      prob_dict[key] = prob_dict[key] / sample_size

    prob_dict = dict(sorted(prob_dict.items()))

    return prob_dict     
     

  elif compute_proposal_matrix == True: 
    if mode == "Exact":
      full_Ham_mat = Learner_Ham(N, poly, gamma, type_of_Ham="with_mixer").to_matrix()  
      U_t = scipy.linalg.expm(-1.0j*tot_time*full_Ham_mat)

    elif mode == "Trotter":
      U_t = np.zeros((2**N, 2**N), dtype=np.complex128)

      for key in range(2**N):
          s = key_to_spin_nv(key, N)
          angles_ry = np.flip(np.pi*(1-s)/2)

          U_t[:, key] = (np.array(cudaq.get_state(
                        Trotter_circuit, N, k, angles_ry, angles_u3, np.reshape(angles_2q,-1)), copy=False))

    Proposal_mat =  np.abs(U_t)**2  
    #Proposal_mat = np.ones((2**N,2**N))/(2**N)

    s = enum(N)
    Energy = Energy_Ising(s, N, poly)
    Transition_matrix = np.abs(get_transition_matrix_from_proposal(N, Proposal_mat, Energy, acceptance_criteria='metropolis', beta=beta))

    key = spin_to_key_nv(init_config)
    for _ in range(burn): 
      key = np.random.choice(np.arange(2**N), p=Transition_matrix[:,key])

    prob_dist = np.zeros(2**N)

    for _ in range(sample_size):
      key = np.random.choice(np.arange(2**N), p=Transition_matrix[:,key])
      prob_dist[key] += 1

    return prob_dist / np.sum(prob_dist)
 

    

  