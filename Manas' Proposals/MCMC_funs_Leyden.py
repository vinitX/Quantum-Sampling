import numpy as np
import scipy
import itertools
import scipy.linalg as la
#from collections import defaultdict
#import qiskit
from qiskit.circuit import Parameter 
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit import transpile
#from qiskit.providers.aer import QasmSimulator
# Import qubit states Zero (|0>) and One (|1>), Pauli operators (X, Y, Z), and the identity operator (I)
#from qiskit.opflow import Zero, One, X, Y, Z, I, MatrixEvolution, PauliTrotterEvolution
#from qiskit.quantum_info import Statevector
#from qiskit.quantum_info.operators import Operator, Pauli
#from qiskit.opflow.primitive_ops import PauliOp
#from qiskit.opflow.primitive_ops import PauliSumOp
#from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# Define problem instances -----------------------------------------------------------------------------
class ProblemInstance:
    """
    Encodes a problem instance, which is defined on n spins by an n*n coupling matrix J, an 
    n-dimensional local field vector h, and a non-negative temperature T.

    Attributes:
        J: a real numpy array with dimensions (n,n) with J[i,j] = 0 for all i>=j
        h: a real numpy array with dimensions (n)
        T: a number between 0 and np.inf (inclusive)
        n: an integer number of spins >= 2, inferred from the size of h
        J_quantum: a re-scaled version of J used to define the quantum Hamiltonian
        h_quantum: a re-scaled version of h used to define the quantum Hamiltonian
        E_arr: a (2**n) dimensional numpy array of all classical energies possible for J and h.
               E_arr[0] is the energy of (1,...,1) and E_arr[-1] is that of (-1,...,-1), following
               the integer <-> spin config. convention in the int2spinconf and spinconf2int functions
    """
    def __init__(self, J, h, T=None, precomp_E = True):
        """
        Initiates ProblemInstance with J, h, and (optionally) T. T can also be specified later.
        Infers n and defines J_quantum and h_quantum.
        Can pre-compute E_arr to avoid repeatedly computing energy of any spin configuration later.
        """
        if np.any(np.triu(J, k=1) != J):
            raise ValueError('J must be upper-triangular with 0s along the diagonal')

        self.J = J
        self.h = h
        self.T = T 
        self.n = h.size
        
        scale = ( self.n / (la.norm(J, ord='fro')**2 + la.norm(h, ord=2)**2) )**0.5
        self.J_quantum = self.J * scale 
        self.h_quantum = self.h * scale

        if precomp_E:
            self.E_arr = np.zeros((2**self.n))
            for i in range(2**self.n):
                config = int2spinconf(i, self.n)
                self.E_arr[i] = config @ J @ config + config @ h # note sign convention compared to paper



def RandomProblemInstance(n, connectivity, T=None, precomp_E=True):
    """
    Create a random n-spin ProblemInstance with a desired connectivity. If connectivity='full' all
    J[i,j] above the diagonal will be sampled independently from Normal(0,1), rest will be 0. 
    If connectivity='line' all J[i,i+1] will be sampled independently from Normal(0,1), rest will be 0.
    In both cases, all h[i] will be be sampled independently from Normal(0,1). Temperature T not set.
    """
    h = np.random.randn(n)

    if connectivity=='line':
        J = np.diag(np.random.randn(n-1), k=1)

    if connectivity=='full':
        J = np.triu(np.random.randn(n,n), k=1)

    return ProblemInstance(J, h, T=T, precomp_E=precomp_E)


# Useful quantum definitions --------------------------------------------------------------------------- 
def my_kron(arr_list):
    """Takes a list of numpy arrays [A1, A2, ...] and computes their tensor product A1 (x) A2 (x) ..."""
    if len(arr_list) == 1:
        return arr_list[0]
    else:
        return np.kron(arr_list[0], my_kron(arr_list[1:]))


def X(i, n):
    """n-qubit Pauli X. Acts as sigma_x on qubit i and as the identity on the rest."""
    if i<0 or i>=n or n<1:
        raise ValueError('Bad value of i and/or n.')
    X_list = [np.array([[0,1],[1,0]]) if j==i else np.eye(2) for j in range(n)]
    return my_kron(X_list)


def Y(i, n):
    """n-qubit Pauli Y. Acts as sigma_y on qubit i and as the identity on the rest."""
    if i<0 or i>=n or n<1:
        raise ValueError('Bad value of i and/or n.')
    Y_list = [np.array([[0,-1j],[1j,0]]) if j==i else np.eye(2) for j in range(n)]
    return my_kron(Y_list)


def Z(i, n):
    """n-qubit Pauli Z. Acts as sigma_z on qubit i and as the identity on the rest."""
    if i<0 or i>=n or n<1:
        raise ValueError('Bad value of i and/or n.')
    Z_list = [np.array([[1,0],[0,-1]]) if j==i else np.eye(2) for j in range(n)]
    return my_kron(Z_list)



# Misc. useful functions ------------------------------------------------------------------------------
def int2spinconf(i, n):
    """
    Converts an integer i in [0,2**n-1] into an n-spin configuration using the convention
        0 -> np.array([1,...1,1])
        1 -> np.array([1,...,1,-1])
            ...
        2**n-1 -> np.array([-1,...,-1,-1])
    """
    if i<0 or i>2**n-1:
        raise ValueError('i out of range')

    bin_list = list( np.binary_repr(i,n))
    return np.array([1 if bit=='0' else -1 for bit in bin_list])


def spinconf2int(config):
    """The inverse of int2spinconf. Example: spinconf2int(np.array([1,-1,1])) = 2"""
    bin_str = ''.join(['0' if spin==1 else '1' for spin in config])
    return int(bin_str, 2)


def hamming_dist(int1, int2):
    """Computes the Hamming distance between the binary representations of integers int1 and int2"""
    diff_int = np.bitwise_xor(int1, int2)
    diff_bin = [int(bit) for bit in np.binary_repr(diff_int)]
    return sum(diff_bin)


# Construct proposal matrices --------------------------------------------------------------------------
def local_proposal_mat(n):
    """Represents picking a spin uniformly at random and flipping it."""
    return sum([X(i,n) for i in range(n)])/n


def uniform_proposal_mat(n):
    """Represents picking a new spin configuration uniformly at random."""
    return np.ones((2**n, 2**n))/2**n



def Haar_random_proposal_mat(n, haar_samples=20):
    #Generate a Haar-random matrix using the QR decomposition of an arbitrary complex mattrix with entries from std Gaussian dist (0,1).
    Proposal_mat = 0
    for item in np.arange(haar_samples):
        Arb_complex_mat = np.random.normal(0,1.0,size=(2**n, 2**n)) + 1.0j*np.random.normal(0,1.0,size=(2**n, 2**n))
        Q, R = np.linalg.qr(Arb_complex_mat)
        New_lam_matrix = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(2**n)])
        Proposal_mat += np.real(np.multiply(np.conjugate(np.dot(Q, New_lam_matrix)), np.dot(Q, New_lam_matrix)))
    return np.real(np.multiply(1.0/haar_samples, Proposal_mat))


def Euler_angle_decomposition(unitary:np.ndarray):
    #Given a 2*2 unitary matrix as np.array, this function computes the Euler angles (theta, phi, lambda) required
    # to implement that unitary as a U3 gate where U3(theta, phi, lambda)
    from qiskit.synthesis import OneQubitEulerDecomposer
    gate_decomposer = OneQubitEulerDecomposer('U3')
    theta_val, phi_val, lambda_val, _ = gate_decomposer.angles_and_phase(unitary)
    return (theta_val, phi_val, lambda_val)



def two_qubit_gate(circuit, angle:float, qubit_1:int, qubit_2:int, mode="no_decomposition"):
        # This function provides circuit description of RZZ(theta) - This is the 2-qubit gate used for H_Z
    if mode == "no_decomposition":
        circuit.rzz(angle, qubit_1, qubit_2)
    if mode == "CNOT_decomposition":
        circuit.cx(qubit_1, qubit_2)
        circuit.rz(angle, qubit_2)
        circuit.cx(qubit_1, qubit_2)
    if mode == "RZX_decomposition":
        circuit.h(qubit_2)
        circuit.rzx(angle, qubit_1, qubit_2)   # can be implemented natively by pulse-stretching
        circuit.h(qubit_2)  
    return circuit    


                
def quantum_proposal_time_homogeneous(problem_inst, gamma_lims=[0.25, 0.6], t_lims=[2,20], t_val = None):

    J_Q = problem_inst.J_quantum
    h_Q = problem_inst.h_quantum
    n = problem_inst.n

    H_z = sum([J_Q[i,j]*Z(i,n) @ Z(j,n) for i in range(n) for j in range(n)]) + sum([h_Q[i]*Z(i,n) for i in range(n)])
    H_x = sum([X(i,n) for i in range(n)])

    gamma_steps = 20 # approximate integral over c by Riemann sum with c_steps points
    gamma_starts, step_size = np.linspace(gamma_lims[0], gamma_lims[1], num=gamma_steps, endpoint=False, retstep=True)
    gamma_mids = gamma_starts + step_size/2
    gamma_mid_point = gamma_mids[int(len(gamma_mids)/2)]

    gamma_list = [gamma_mid_point]
    #print(gamma_mid_point)

    t_steps = 50 # approximate integral over t by Riemann sum with t_steps points
    t_starts, step_size = np.linspace(t_lims[0], t_lims[1], num=t_steps, endpoint=False, retstep=True)
    t_mids = t_starts + step_size/2
    #t_mid_point = t_mids[int(len(t_mids)/2)]
    #print(t_mid_point)
    
    t_mid_point = 12

    Proposal_mat = 0

    if  t_val == "t-uplim":
        t_val = t_lims[1]
        t_list = [t_val]
    elif t_val == "t-mid":
        t_list = [t_mid_point]
    elif t_val == "t-llim":
        t_val = t_lims[0]
        t_list = [t_val]
    else:
        t_list = t_mids

    #two_body_op = scipy.linalg.expm(np.multiply(-1.0j*t_mid_point, (1-gamma_mid_point)*sum([J_Q[i,j]*Z(i,n) @ Z(j,n) for i in range(n) for j in range(n)])))
    
    for gamma_ind, gamma in enumerate(gamma_list):
        H = (1-gamma)*H_z + (gamma)*H_x
        for time_ind, time in enumerate(t_list):
            U_t = scipy.linalg.expm(np.multiply(-1.0j*time, H))
            Proposal_mat += np.multiply(np.conjugate(U_t), U_t)
    return np.real(np.multiply(1.0/len(t_list), Proposal_mat))



def Trotter_circuit(problem_inst, Trotter_repeat_length:int, 
                        gamma:float, time_delta=0.5, initial_config=None):
        # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
            
        from collections import defaultdict
        J_Q = problem_inst.J_quantum
        h_Q = problem_inst.h_quantum
        n = problem_inst.n
            
        qreg = QuantumRegister(n)
        creg = ClassicalRegister(n)
        circuit = QuantumCircuit(qreg, creg)
            
        if initial_config != None:
            assert(len(initial_config)==n), "initial config array is not same length as number of spins"    
            circuit.x(qreg[n-1-np.argwhere(return_array==1).flatten()])  # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have no_spins-1-index
        
        #exp_op = 1
        angle_dict = defaultdict(tuple)
    
        for qubit in np.arange(n):
            one_body_Ham = SparsePauliOp(["X", "Z"], [gamma, (1-gamma)*h_Q[qubit]]).simplify()
            #exp_op *= scipy.linalg.expm(-1.0j*time_delta*one_body_Ham)
            angle_dict[qubit]= Euler_angle_decomposition(scipy.linalg.expm(-1.0j*time_delta*one_body_Ham))   # always 2*2 so no problem of exponentiation, storage
        
    
        #print("dummy ham unitary trotter", scipy.linalg.expm(-1.0j*Trotter_repeat_length*time_delta*dummy_ham))
        #import qiskit.quantum_info as qi
    
        for _ in range(Trotter_repeat_length-1):
            for qubit in np.arange(n):
                circuit.u(angle_dict[qubit][0],angle_dict[qubit][1],angle_dict[qubit][2], qreg[n-1-qubit])

            
            for qubit_tuple in list(itertools.combinations(np.arange(n),r=2)):
               circuit = two_qubit_gate(circuit, 
                                   2*J_Q[qubit_tuple[0],qubit_tuple[1]]*(1-gamma)*time_delta, 
                                   qreg[n-1-qubit_tuple[0]], qreg[n-1-qubit_tuple[1]], mode="no_decomposition")

            #two_body_op = np.dot(two_body_op, np.array(qi.Operator(circuit)))
            circuit.barrier()
            
            
        #print(two_body_op)        
        for qubit in np.arange(n):
            circuit.u(angle_dict[qubit][0],angle_dict[qubit][1],angle_dict[qubit][2], qreg[n-1-qubit])

        #exp_op = np.array(qi.Operator(circuit).data)
        
        return circuit




def quantum_proposal_time_homogeneous_Trotter_circuit(problem_inst, delta_step, gamma_lims=[0.25,0.6], t_lims=[2,20], t_val=None):
    gamma_steps = 20 # approximate integral over c by Riemann sum with c_steps points
    gamma_starts, step_size = np.linspace(gamma_lims[0], gamma_lims[1], num=gamma_steps, endpoint=False, retstep=True)
    gamma_mids = gamma_starts + step_size/2
    gamma_mid_point = gamma_mids[int(len(gamma_mids)/2)]

    gamma_list = [gamma_mid_point]

    t_steps = 50 # approximate integral over t by Riemann sum with t_steps points
    t_starts, step_size = np.linspace(t_lims[0], t_lims[1], num=t_steps, endpoint=False, retstep=True)
    t_mids = t_starts + step_size/2
    t_mid_point = t_mids[int(len(t_mids)/2)]
    
    t_mid_point = 12
    #delta_step = 0.05
    Proposal_mat =0
    
    if  t_val == "t-uplim":
        t_list = [t_lims[1]]
    elif t_val == "t-mid":
        t_list = [t_mid_point]
    elif t_val == "t-llim":
        t_list = [t_lims[0]]
    else:
        t_list = t_mids
        
    import qiskit.quantum_info as qi

    for gamma_ind, gamma in enumerate(gamma_list):
        for time_ind, time in enumerate(t_list):
            circuit = Trotter_circuit(problem_inst, Trotter_repeat_length=int(time/delta_step),
                                                 gamma=gamma,time_delta=delta_step,initial_config=None)
            
            #one_body_op = np.dot(one_body_op, np.array(qi.Operator(circuit)))
            #print(one_body_op)

            U_t = qi.Operator(circuit)
            Proposal_mat += np.multiply(np.conjugate(U_t), U_t)

    return np.real(np.multiply(1.0/len(t_list), Proposal_mat))



def quantum_proposal_mat_avg(problem_inst, gamma_lims=[0.25,0.6]):
    """
    Returns a 2**n * 2**n stochastic proposal matrix for our quantum method of suggesting moves,
    with no Trotter, gate or SPAM errors.
    """
    def cont_eig(Dlambda):
        t_0, t_f = 2, 20 # evolution time t ~ unif(t_0, t_f)
        x = np.sin(Dlambda*t_f) - np.sin(Dlambda*t_0) # from analytical expression for transition probabilities
        return np.divide(2*x/(t_f-t_0), Dlambda, out=np.ones_like(Dlambda), where=(Dlambda!=0) ) 

    J_Q = problem_inst.J_quantum 
    h_Q = problem_inst.h_quantum
    n = problem_inst.n 

    H_z = sum([J_Q[i,j]*Z(i,n) @ Z(j,n) for i in range(n) for j in range(n)]) + sum([h_Q[i]*Z(i,n) for i in range(n)])
    H_x = sum([X(i,n) for i in range(n)])

    d = 2**n
    a = np.arange(d**2)
    mask = (a//d >= a%d)
    ones = np.ones(d)

    # c is called gamma in paper
    #c_lims = [0.25, 0.6] # sample c ~ unif(c_lims[0], c_lims[1])
    gamma_steps = 20 # approximate integral over c by Riemann sum with c_steps points
    gamma_starts, step_size = np.linspace(gamma_lims[0], gamma_lims[1], num=gamma_steps, endpoint=False, retstep=True)
    gamma_mids = gamma_starts + step_size/2

    prop_list = [None]*gamma_mids.size

    for gamma_ind, gamma in enumerate(gamma_mids):
        H = (1-gamma)*H_z + gamma*H_x
        vals, vecs = la.eigh(H)

        vals_diff = (np.kron(vals, ones) - np.kron(ones, vals))[mask]
        M = la.khatri_rao(vecs.T, vecs.T)[mask]
        prop_list[gamma_ind] = M.T * cont_eig(vals_diff) @ M 
    
    Proposal_mat = sum(prop_list)/len(gamma_mids)
    return Proposal_mat


# Transition matrices, spectral gap, random moves ------------------------------------------------------
def make_transition_mat(problem_inst, proposal_mat, acceptance='metropolis'):
    """
    Constructs the full 2**n * 2**n MCMC transition matrix for problem_inst, where moves are suggested
    according to proposal_mat and accepted/rejected using either metropolis or glauber probabilities. 
    Both acceptance probability assume proposal_mat is symmetric. 
    Args:
        problem_inst: a ProblemInstance object with the temperature T between 0 and np.inf (inclusive)
        proposal_mat: a 2**n * 2**n stochastic matrix (numpy array) which need not be symmetric
        acceptance: 'metropolis' or 'glauber' (i.e., Gibbs sampler)
    """
    if problem_inst.T is None:
        raise TypeError('Temperature T is undefined.')
    if problem_inst.T < 0:
        raise ValueError('Temperature T cannot be negative.')

    n = problem_inst.n
    T = problem_inst.T
    E_rowstack = np.tile(problem_inst.E_arr, (2**n,1))  # E_rowstack[i,j] = E_j for all i
    E_diff = E_rowstack.T - E_rowstack # E_diff[i,j] = E_i - E_j (new E minus old E)

    uphill_moves = (E_diff >= 0) # includes isoenergetic moves
    downhill_moves = np.invert(uphill_moves)

    if acceptance=='metropolis':
        if T>0:
            # avoid overflows by only computing exp(...) when -E_diff/T =< 0, otherwise set to 1
            pi_ratio = np.exp(-E_diff/T, where=uphill_moves, out=np.ones_like(E_diff)) 
        if T==0:
            # reject all uphill, accept all others (level and downhill)
            pi_ratio = np.where(uphill_moves, 0., 1.)
        A = pi_ratio

    if acceptance=='glauber':
        if T>0:
            pi_ratio = np.exp(-E_diff/T, where=uphill_moves, out=E_diff*np.nan) # compute when -E_diff/T =< 0
            pi_ratio_inv = np.exp(E_diff/T, where=downhill_moves, out=E_diff*np.nan) # compute when +E_diff/T < 0
        if T==0:
            # reject all uphill, accept all others (level and downhill)
            pi_ratio = np.where(uphill_moves, 0., 1.)
            pi_ratio_inv = np.where(downhill_moves, 0., 1.)

        # compute A in two equivalent ways depending on sign(E_diff) to avoid overflows
        A = np.where(uphill_moves, pi_ratio/(1+pi_ratio), (1+pi_ratio_inv)**(-1) )

    P = A * proposal_mat
    np.fill_diagonal(P, 0)
    diag = np.ones(2**n) - np.sum(P, axis=0)
    P = P + np.diag(diag)
    return P


def abs_spectral_gap(transition_mat):
    """
    Returns the absolute spectral gap of transition_mat, and also of its lazy variant (I + transition_mat)/2.
    """
    dist = np.sort( 1-np.abs(la.eigvals(transition_mat)) )
    delta = np.min(dist[1:])
    dist_lazy = np.sort( 1-np.abs(1/2 + la.eigvals(transition_mat)/2) )
    delta_lazy = np.min(dist_lazy[1:])
    return delta, delta_lazy


def generate_move(transition_mat, state):
    """
    Generate a random move from current state according to transition_mat.
    Both the input and output states are represented as integers in [0, 2**n-1].
    """
    return np.random.choice(transition_mat.shape[0], p=transition_mat[:,state])


def make_MCMC_trajectories_th(P, num_trajectories, num_moves, initial_state=None):
    """
    Generate MCMC trajectories based on a provided transition matrix P. Returns a numpy array of dimension 
    [num_trajectories, num_moves+1], where each row is an independent trajectory, represented by a sequence
    of random integers in [0,2^n-1] (for an n-spin problem) encoding spin configurations.
    Args:
        P: a 2^n * 2^n transition matrix for a Markov chain
        num_trajectories: the number of independent MCMC trajectories to create
        num_moves: the number of MCMC moves to be made for each trajectory
        initial_state: an integer in [0, 2^n-1] describing the initial configuration of each trajectory, 
                       or None for each trajectory to start in a (different) uniformly random state
    """
    trajectories = np.empty((num_trajectories, num_moves+1), dtype='int')
    for traj in range(num_trajectories):

        if initial_state is None:
            trajectories[traj, 0] = np.random.randint(P.shape[0])
        else:
            trajectories[traj, 0] = initial_state

        for move in range(1, num_moves+1):
            trajectories[traj, move] = generate_move(P, trajectories[traj, move-1])

    return trajectories


def make_MCMC_trajectories_exp(problem_inst, C_arr, initial_state=None, max_num_moves=None):
    """
    Mimics doing MCMC "online" using buffered quantum transitions, rather than doing accept/reject + feedforward 
    in an experiment.  Generates Metropolis trajectories based on experimental data, comprising of observed quantum 
    transitions between computational states |i> --> |j>, potentially for multiple quantum circuits. Starting from some 
    configuration i, this function draws an observed |i> -> |j> transition at random without replacement from the data 
    (for a circuit chosen uniformly at random in each move, if applicable), and proposes an i->j MCMC move, which is 
    accepted or rejected according to the Metropolis rule. Continues until there are no more |i> -> |j> quantum 
    transitions left for the current state i (and the randomly chosen circuit). Called "Markov chain subsampling" in
    supplementary info.
    
    Returns a numpy array whose rows are independent Metropolis trajectories represented by a sequence of random 
    integers in [0,2^n-1] (for an n-spin problem) encoding spin configurations. Creates as many rows (independent 
    trajectories) of equal length as possible until the data runs out.

    Args:
        problem_inst: a ProblemInstance object on which the data is based
        C_arr: a 3-dimensional numpy array where C_arr[i,j,k] is the number of observed |j> -> |i> transitions for
               circuit k
        initial_state: an integer in [0, 2^n-1] describing the initial configuration of each trajectory, 
                       or None for each trajectory to start in a (different) uniformly random state
        max_num_moves: The desired number of moves in each trajectory. If data runs out before this number is reached,
                       or if None, a single trajectory that is as long as possible is returned. Otherwise, multiple
                       independent trajectories with max_num_moves moves will be returned.
    """

    n = problem_inst.n

    # returns a single trajectory, along with the remaining data
    def make_trajectory(C_arr, initial_state=None, max_num_moves=None):
        C_arr_copy = np.copy(C_arr)
        num_moves = np.sum(C_arr) if max_num_moves is None else max_num_moves

        if initial_state is None:
            initial_state = np.random.randint(2**n)
        trajectory = [initial_state]

        for _ in range(num_moves):
            j = trajectory[-1] # current state
            k = np.random.randint(C_arr.shape[2]) # pick random circuit
            candidates = np.repeat(np.arange(2**n), C_arr_copy[:,j,k]) # list all (remaining) j->i transitions observed for circuit k 
            if candidates.size == 0:
                break
            else:
                i = np.random.choice(candidates) # pick candidate state uniformly from (remaining) observed ones
                C_arr_copy[i,j,k] += -1 # remove proposed transition from data set

                DE = problem_inst.E_arr[i] - problem_inst.E_arr[j] # energy difference
                a = min(1, np.exp(-DE/problem_inst.T)) # acceptance probability
                if np.random.rand() <= a: # move accepted
                    trajectory.append(i)
                else: # move rejected
                    trajectory.append(j)

        return trajectory, C_arr_copy

    trajectories = []
    empty = False
    C_arr_remainder = np.copy(C_arr)
    while not empty:
        trajectory, C_arr_remainder = make_trajectory(C_arr_remainder, max_num_moves=max_num_moves)
        if len(trajectory) < max_num_moves+1:
            empty=True
        else:
            trajectories.append(trajectory)

    return np.array(trajectories)

