import quimb as qu
import quimb.tensor as qtn
import numpy as np
import time
from Sampling_Quantum import *

def build_circuit(n_qubits, depth, gate_params):
    """Build a quantum circuit and convert it to an MPO."""
    # Initialize the quantum circuit
    circuit = qtn.Circuit(n_qubits)

    for layer in range(depth):
        # Apply single-qubit RX rotations
        for i in range(n_qubits):
            theta = gate_params['single'][layer, i]
            circuit.apply_gate('rx', theta, i)

        # Apply two-qubit RZZ gates (nearest neighbors)
        for i in range(n_qubits - 1):
            for j in range(i+1, n_qubits):
                angle = gate_params['rzz'][layer, i, j]
                circuit.apply_gate('rzz', angle, i, j)

    for i in range(n_qubits):
        theta = gate_params['single'][depth-1, i]
        circuit.apply_gate('rx', depth-1, i)

    # Convert the circuit to an MPO
    #mpo = circuit.to_mpo()
    return circuit


def Singular_values(mpo,i):
    #Returns singular values of an MPO at i'th site
    #Requires MPO to be canonicalized at i'th index
    mpo = mpo.canonicalize(i)

    A = mpo[i].data
    chi = A.shape[1]  #Bond Dimension
    A = np.transpose(A, (0,2,3,1))
    A = np.reshape(A, (-1,chi))

    B = mpo[i+1].data
    B = np.reshape(B, (chi,-1))

    S = np.linalg.svd(A@B, compute_uv=False)

    return S



def swap_gate():
    """Returns the SWAP gate as a (2,2,2,2) tensor."""
    SWAP = np.array([[1, 0, 0, 0], 
                     [0, 0, 1, 0], 
                     [0, 1, 0, 0], 
                     [0, 0, 0, 1]])
    return SWAP

def rzz(theta):
    """Create the RZZ gate matrix."""
    ZZ = np.kron(qu.pauli('Z'), qu.pauli('Z'))
    return qu.expm(-1j * theta / 2 * ZZ)


def rx(theta):
    return qu.expm(-1j * theta / 2 * qu.pauli('X'))

def u3(angles):
    theta, phi, lam = angles
    return np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]], dtype=complex)


def decompose_two_qubit_gate(gate, cutoff=1e-14):
    gate = np.reshape(gate, (2,2,2,2))
    gate = np.transpose(gate, (0,2,1,3))
    gate = np.reshape(gate, (4,4))

    U, S, Vh = np.linalg.svd(gate)  
    S_diag = np.diag(S)  
    #print("singular values: ", S)
    bd = np.sum(S>cutoff)
    #print("Bond Dimension: ", bd)

    # Absorb sqrt(S) into U and Vh to get two-site tensors
    U_new = U @ np.sqrt(S_diag)
    Vh_new = np.sqrt(S_diag) @ Vh

    U_new = U_new[:,:bd]
    Vh_new = Vh_new[:bd,:]

    U_new = (U_new.T).reshape(1, -1, 2, 2)
    Vh_new = Vh_new.reshape(-1, 1, 2, 2)

    return U_new, Vh_new, bd


def create_non_local_gate_mpo(gate, i, j, N):
    assert i < j
    assert i >= 0
    assert j < N

    tensors = []
    for _ in range(i):
        tensors.append(np.reshape(np.eye(2),(1,1,2,2)))
    
    U, V, bd = decompose_two_qubit_gate(gate)

    tensors.append(U)

    delta_ij = np.eye(bd)
    delta_ab = np.eye(2)
    I = np.einsum('ij,ab->ijab', delta_ij, delta_ab)

    for _ in range(i+1,j):
        tensors.append(I)
 
    tensors.append(V)

    for _ in range(j+1,N):
        tensors.append(np.reshape(np.eye(2),(1,1,2,2)))

    mpo = qtn.MatrixProductOperator(tensors)
    #mpo.show()

    return mpo


def U3_layer_mpo(angles, N):
    tensors = []
    for i in range(N-1,-1,-1):  #Following the little-endian notation for CudaQ
        gate = u3([angles[i], angles[i+N], angles[i+2*N]])
        #if i==0 or i==N-1: tensors.append(gate.reshape(1,2,2))
        #else: 
        tensors.append(gate.reshape(1,1,2,2))

    mpo = qtn.MatrixProductOperator(tensors)
    return mpo


def RZZ_layer_mpo(theta, N, max_bond=None):
    mpo = qtn.MPO_identity(N, dtype='complex64')

    for i in range(N):
        for j in range(i+1, N):
            #Following the little-endian notation for CudaQ
            rzz_mpo = create_non_local_gate_mpo(rzz(theta[N-i-1,N-j-1]), i, j, N)
            mpo = mpo.apply(rzz_mpo, compress=True, max_bond=max_bond)
            
    return mpo


def build_mpo(n_qubits, depth, params, max_bond=None, rzz_max_bond=None):
    """Build compressed MPO with explicit gate tensors"""

    mpo = qtn.MPO_identity(n_qubits, dtype='complex64')

    u3_mpo = U3_layer_mpo(params['u3'],n_qubits)

    tm = time.time()
    rzz_mpo = RZZ_layer_mpo(params['zz'], n_qubits, max_bond=rzz_max_bond)
    #rzz_mpo.show()

    tm = time.time()
    #The apply function follows the reverse order of application than usual quantum circuits. That's why RZZ is applied first. 
    for _ in range(depth):
        mpo = mpo.apply(rzz_mpo, compress=True, max_bond=max_bond,)
        mpo = mpo.apply(u3_mpo, compress=True)   
        #mpo.show()
 
    print("Circuit creation time: ", time.time()-tm)

    return mpo

def MPS_sampler(N,poly,s_old,mpo):
    s = ((1-s_old)/2)  #np.flip

    #print(s)
    mps = qtn.MPS_computational_state(np.array(s,dtype=int))
    result_mps = mpo.apply(mps)

    samples = result_mps.sample(1)

    for i, sp in enumerate(samples):
        s_new, _ = sp
        s_new = (1 - 2*np.array(s_new)) #np.flip

    p1 = prob_Ising_nv(s_old, N, poly)
    p2 = prob_Ising_nv(s_new, N, poly)

    accept = min(1.0,p2/p1)
    if np.random.rand()<accept: return s_new     
    else: return s_old



def Sampling_MPO(N, poly, sample_size, tot_time=12, time_delta=0.5, gamma=0.42, beta=1, burn=None):
    angles_u3, angles_2q = compute_angles(poly, N, time_delta, gamma)
    k = int(tot_time / time_delta)

    params = {
        'u3': angles_u3,
        'zz': angles_2q + angles_2q.T
    }

    mpo = build_mpo(n_qubits=N, depth=k, params=params, max_bond=32, rzz_max_bond=16)
    mpo.show()

    s = np.random.choice([1.,-1.],size=N)
    
    for _ in range(burn):
        s = MPS_sampler(N,poly,s,mpo)

    prob_dict = {}
    key_list = []
      
    for _ in range(sample_size):
        s = MPS_sampler(N,poly,s,mpo)

        key = spin_to_key_nv(s)
        if key in prob_dict: prob_dict[key] +=1
        else: prob_dict[key] = 1
        key_list.append(key)

    for key in prob_dict.keys():
      prob_dict[key] = prob_dict[key] / sample_size

    prob_dict = dict(sorted(prob_dict.items()))

    return prob_dict, key_list     


