import numpy as np
import scipy
import sys
import os
import itertools
from itertools import product
from scipy import *
import scipy as sp
# import netket as nk, jax
# from netket.hilbert import CustomHilbert
# from netket.operator import LocalOperator
from scipy.sparse import csc_matrix, csr_matrix, linalg
from scipy.sparse import *
import math
import pandas as pd
import matplotlib
import itertools
from typing import Type
from collections import defaultdict
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit import Aer, transpile
#from qiskit.providers.aer import QasmSimulator
# Import qubit states Zero (|0>) and One (|1>), Pauli operators (X, Y, Z), and the identity operator (I)
from qiskit.opflow import Zero, One, X, Y, Z, I, MatrixEvolution, PauliTrotterEvolution
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian
from qiskit.quantum_info import SparsePauliOp
from qutip import *
from qutip.qip.operations import cnot, rx
from qutip.qip.circuit import QubitCircuit, Gate
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16})  # enlarge matplotlib fonts
sys.path.append(r'.')
#os.chdir("/home/msajjan/Desktop/Qiskit_learning/Open_challenge_2023/")
os.getcwd()
from qiskit.providers.fake_provider import *
#from heisenberg_model import *
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
from qiskit import IBMQ
from qiskit.utils import QuantumInstance
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.opflow import X, Y, Z, I, StateFn, CircuitStateFn, SummedOp
from qutip.qip.circuit import QubitCircuit
from qiskit.opflow import PauliExpectation, CircuitSampler
from scipy.optimize import OptimizeResult
#QiskitRuntimeService.save_account(channel="ibm_quantum", instance='ibm-q-afrl/air-force-lab/quantum-sim', token="15713c90074232b9b4f1089b82f9b34995edf382fcc53686ac9a9071ed38f33d3599d639cae3583245f84e7a6bb3ed66317ad653870f7bfcbbce2d46997e4ec2",overwrite=True)
#service=QiskitRuntimeService(channel='ibm_quantum', instance='ibm-q-afrl/air-force-lab/quantum-sim')
#service.backends()
import csv
#from sympy import *
from qutip import *
#import jdc
import re
#from platform import python_version
#from funcy import join
import os
#from __future__ import print_function  # for Python2
import psutil
#print(psutil.__version__)
print (os.getcwd())
from collections import defaultdict
from matplotlib import pyplot as plt
matplotlib.rcParams['mathtext.fontset'] ='stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["font.weight"] = "bold"
print (qiskit.__qiskit_version__)




class All_proposals:
    def __init__(self,**kwargs):
        self.beta= kwargs['inverse_temp']
        if self.beta is None:
            raise TypeError('Temperature T is undefined.')
        if self.beta < 0:
            raise ValueError('Temperature T cannot be negative.')

        # For model instance below, use the H(v) function. This H(v) appears inside the fitted
        #proxy distribution phi(v) which is phi(v)=e^-H(v)/Z.... H(v) = \sum_i h_i v_i + \sum_{ij} Q_ij v_i v_j
        #I shall refer to h_i as one-body terms passed as a list of length(n) and Q_ij as two body terms
        #passed as a list of length(n^2) with i index changing faster than j index. Note h_i and Q_ij are always real

        #assert kwargs['one_body_coeffs'].any != None, "Please provide one-body coeffs from the fitted distribution phi(v)"  #This is the one-body term within the Ham of phi(v). This is a function of all RBM model params
        #assert kwargs['two_body_coeffs'] != None, "Please provide two-body coeffs from the fitted distribution phi(v)"  #This is the two-body term within the Ham of phi(v). This is a function of all RBM model params

        self.no_spins = len(kwargs['one_body_coeffs'])

        self.model_instance_one_body = np.zeros(self.no_spins)
        self.model_instance_two_body  = np.zeros((self.no_spins, self.no_spins))    # If the one and two body elems are labelled such that the first slot in the config is site 0, then use the following. If not PLEASE COMMENT IT OUT !


        for qubit_tuple in list(itertools.combinations(np.arange(self.no_spins),r=2)):
            self.model_instance_two_body[qubit_tuple[0], qubit_tuple[1]]= np.reshape(np.array(kwargs['two_body_coeffs']), (self.no_spins, self.no_spins), order='F')[self.no_spins -1-qubit_tuple[0], self.no_spins -1-qubit_tuple[1]]

        self.model_instance_two_body = self.model_instance_two_body + self.model_instance_two_body.T

        for qubit_index in np.arange(self.no_spins):
            self.model_instance_one_body[qubit_index]= np.array(kwargs['one_body_coeffs'])[self.no_spins -1-qubit_index]


    def get_energy_array(self):
        self.Energy_array = np.zeros(2**self.no_spins)

        for num in np.arange(2**self.no_spins):
             config = self.get_int_to_spinconfig(num, self.no_spins)
             self.Energy_array[num] = np.dot(config, np.dot(self.model_instance_two_body, config)) + np.dot(config,self.model_instance_one_body)
        return self


    def Learner_Ham_maker_w_wo_mixer(self, gamma:float, alpha:float, type_of_Ham="with_mixer"): #Technically not required, only useful if prop of H(v) is studied
        list_op=[]
        import itertools
        #This makes the full learner Ham with mixer i.e. (1-gamma)*alpha*H_z + gamma*H_x  or without mixer where we just have H_z
        # where H_z = H(v) = \sum_i h_i v_i + \sum_{ij} Q_ij v_i v_j with v_i replaced with sigma_z
        #H_x = mixer = \sum_i X_i
        #alpha is the norm ratio of ||H_x||_F/||H_z||_F
        #gamma is unif(gamma1, gamma2). The paper says gamma1 as 0.2 and gamma2 as 0.6
        #CAUTION - USE this function for SMALL qubit size (n) ONLY as it involves exp storage of matrix size 2**n * 2**n when converted to np.array.
        # As a SparsePaulilist object, as I have defined below, it is fine to use it even for higher number of qubits

        for qubit_tuple in list(itertools.combinations(np.arange(self.no_spins),r=2)):
            list_op.append(("ZZ", [qubit_tuple[0], qubit_tuple[1]],
                                (1-gamma)*alpha*(self.model_instance_two_body[qubit_tuple[0], qubit_tuple[1]])))

        for qubit_index in np.arange(self.no_spins):
            list_op.append(("Z", [qubit_index], (1-gamma)*alpha*self.model_instance_one_body[qubit_index]))
            if type_of_Ham == "with_mixer":
                list_op.append(("X", [qubit_index], gamma))

        self.Learner_Ham_w_wo_mixer = SparsePauliOp.from_sparse_list(list_op, num_qubits=self.no_spins).simplify()
        return self


    def computing_norm_ratio(self):  #This gives value of alpha = self.computing_norm_ratio()
        #This computes alpha = ||H_x||_F/||H_z||_F using params only. No storing and computing full matrix is necessary
        #Coupling_matrix = np.reshape(np.array(self.model_instance_two_body), (self.no_spins, self.no_spins), order='F')
        alpha = np.sqrt(self.no_spins)/np.sqrt(sum([J**2 for J in self.model_instance_two_body[np.tril_indices(self.no_spins, k = -1)]]) + sum([h**2 for h in self.model_instance_one_body]))
        return alpha


    def scalar_gamma_sampling(self, gamma_limits:tuple=(0.25,0.6), seed=10, sampling_type="continuous"):
        # This function samples the scalar gamma for defining the unitary U(t, gamma)
        if sampling_type == "continuous":
            np.random.seed(seed)
            gamma = np.random.uniform(gamma_limits[0], gamma_limits[1], 1)[0]
            return gamma
        else:
            gamma_steps = 30 # approximate integral over by Riemann sum
            gamma_start_array, gamma_step_size = np.linspace(gamma_limits[0], gamma_limits[1], num=gamma_steps, endpoint=False, retstep=True)
            gamma_midpt_array = gamma_start_array + gamma_step_size/2
            return gamma_midpt_array, gamma_step_size


    def scalar_time_sampling(self, time_limits:tuple=(4,20), seed=10, time_delta=0.5, sampling_type="continuous"):
        # This function samples the scalar time t for defining the unitary U(t, gamma)
        if sampling_type == "continuous":
            np.random.seed(seed)
            time = np.random.uniform(time_limits[0], time_limits[1], 1)[0]
            return time
        else:
            low_stop = np.ceil(time_limits[0]/time_delta)
            high_stop = np.ceil(time_limits[1]/time_delta) + 1
            time_array = [np.round(r*time_delta,4) for r in np.arange(low_stop, high_stop,1)]
            return time_array, time_delta

    def Euler_angle_decomposition(self, unitary:np.ndarray):
        #Given a 2*2 unitary matrix as np.array, this function computes the Euler angles (theta, phi, lambda) required
        # to implement that unitary as a U3 gate where U3(theta, phi, lambda)
        from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
        gate_decomposer = OneQubitEulerDecomposer('U3')
        theta_val, phi_val, lambda_val, _ = gate_decomposer.angles_and_phase(unitary)
        return (theta_val, phi_val, lambda_val)


    def two_qubit_gate(self, circuit, angle:float, qubit_1:int, qubit_2:int, mode="no_decomposition"):
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


    def Trotter_circuit(self, Trotter_repeat_length:int, alpha:float,
                        gamma:float, time_delta=0.5, initial_config=None):
        # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens

        from collections import defaultdict

        qreg = QuantumRegister(self.no_spins)
        creg = ClassicalRegister(self.no_spins)
        circuit = QuantumCircuit(qreg, creg)

        if initial_config != None:
            assert(len(initial_config)==self.no_spins), "initial config array is not same length as number of spins"
            circuit.x(qreg[self.no_spins-1-np.argwhere(return_array==1).flatten()])  # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have no_spins-1-index

        angle_dict = defaultdict(tuple)
        for qubit in np.arange(self.no_spins):
            one_body_Ham = SparsePauliOp(["X", "Z"], [gamma, alpha*(1-gamma)*self.model_instance_one_body[qubit]]).simplify()
            angle_dict[qubit]= self.Euler_angle_decomposition(scipy.linalg.expm(-1.0j*time_delta*one_body_Ham.to_matrix()))   # always 2*2 so no problem of exponentiation, storage


        for _ in range(Trotter_repeat_length-1):
            for qubit in np.arange(self.no_spins):
                circuit.u(angle_dict[qubit][0],angle_dict[qubit][1],angle_dict[qubit][2], qreg[qubit])

            for qubit_tuple in list(itertools.combinations(np.arange(self.no_spins),r=2)):
                circuit = self.two_qubit_gate(circuit,
                                    2*self.model_instance_two_body[qubit_tuple[0],qubit_tuple[1]]*(1-gamma)*alpha*time_delta,
                                    qreg[qubit_tuple[0]], qreg[qubit_tuple[1]], mode="no_decomposition")
                                    # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have no_spins-1-index
            circuit.barrier()

        for qubit in np.arange(self.no_spins):
            circuit.u(angle_dict[qubit][0],angle_dict[qubit][1],angle_dict[qubit][2], qreg[qubit])

        return circuit


    def get_quantum_circuit_proposal_matrix(self, mode="Exact-no Trotter error/no Sampling error"):
        # This function creates the entire proposal distribution matrix Q(s'|s) from the quantum circuit
        alpha = self.computing_norm_ratio()
        time_array, time_delta_step = self.scalar_time_sampling(sampling_type="discrete")
        gamma_array, gamma_step = self.scalar_gamma_sampling(sampling_type="discrete")
        Proposal_mat = 0

        for gamma_val in gamma_array:
            Proposal_mat = np.multiply(gamma_step/(gamma_array[-1]-gamma_array[0]), Proposal_mat)    # averaging over gamma vals

            for time in time_array:
                if mode == "Exact-no Trotter error/no Sampling error":
                    self.Learner_Ham_maker_w_wo_mixer(gamma_val, alpha, type_of_Ham="with_mixer")
                    full_Ham_mat = self.Learner_Ham_w_wo_mixer.to_matrix()
                    U_t = scipy.linalg.expm(np.multiply(-1.0j*time, full_Ham_mat))

                if mode == "Trotter error/no Sampling error":
                    circuit = self.Trotter_circuit(Trotter_repeat_length=int(time/time_delta_step), alpha=alpha,
                                                 gamma=gamma_val,time_delta=time_delta_step, initial_config=None)
                    import qiskit.quantum_info as qi
                    U_t = qi.Operator(circuit)

                Proposal_mat +=  np.multiply(np.conjugate(U_t), U_t)

                # if(time in [time_array[-1], time_array[-2], time_array[-3]]):
                #     self.Learner_Ham_maker_with_mixer(gamma_val, alpha)
                #     full_Ham_mat = self.Learner_Ham_w_mixer.to_matrix()
                #     U_t_1 = scipy.linalg.expm(np.multiply(-1.0j*time, full_Ham_mat))
                #     print("rel norm diff with exact", time, gamma_val, np.linalg.norm(np.multiply(np.conjugate(U_t_1), U_t_1)- np.multiply(np.conjugate(U_t), U_t))/np.linalg.norm(np.multiply(np.conjugate(U_t_1), U_t_1)))

            Proposal_mat = np.multiply(time_delta_step/(time_array[-1]-time_array[0]), Proposal_mat) # averaging over t vals


        if scipy.linalg.issymmetric(Proposal_mat, atol=10**(-10)):
            print("Quantum Proposal matrix is symmetric with abs error 10^-10")

        return Proposal_mat


    def get_uniform_proposal_matrix(self):
        #Creates a uniform proposal matrix as np.array where the transition prob from s to s' i..e P(s'|s)=1/2^n for all s and s' pair
        Proposal_mat = np.multiply(1.0/2**(self.no_spins), np.ones((2**(self.no_spins), 2**(self.no_spins))))
        return Proposal_mat


    def get_local_proposal_matrix(self):
         #Creates a proposal matrix as np.array wherein a spin is chosen at random and flipped.
        #This is local proposal flipping responsible for generating configs with close by hamming dists
        list_op=[]
        for qubit_index in np.arange((self.no_spins)):
            list_op.append(("X", [qubit_index], 1.0))
        Sum_Pauli_x = SparsePauliOp.from_sparse_list(list_op, num_qubits=(self.no_spins)).simplify()
        Proposal_mat = np.multiply(1.0/self.no_spins, Sum_Pauli_x.to_matrix())
        return Proposal_mat


    def get_Haar_random_proposal_matrix(self):
        #Generate a Haar-random matrix using the QR decomposition of an arbitrary complex mattrix with entries from std Gaussian dist (0,1).
        Arb_complex_mat = np.random.normal(0,1.0,size=(2**self.no_spins, 2**self.no_spins)) + 1.0j*np.random.normal(0,1.0,size=(2**self.no_spins, 2**self.no_spins))
        Q, R = np.linalg.qr(Arb_complex_mat)
        New_lam_matrix = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(2**self.no_spins)])
        Proposal_mat = np.real(np.multiply(np.conjugate(np.dot(Q, New_lam_matrix)), np.dot(Q, New_lam_matrix)))
        return Proposal_mat


    def get_transition_matrix_from_proposal(self, Proposal_mat, acceptance_criteria='metropolis'):
        # This function gets the full transition matrix P(s'|s) = Q(s'|s) * Acceptance(s'|s) where Q(s'|s)
        #can be a quantum circuit proposal, a local flip proposal, uniform proposal or Haar random proposal
        import math
        E_rowstack = np.tile(self.Energy_array, (2**self.no_spins,1))  # E_rowstack[i,j] = E_j for all i
        E_diff = E_rowstack.T - E_rowstack # E_diff[i,j] = E_i - E_j (new E minus old E)

        uphill_moves = (E_diff >= 0) # includes isoenergetic moves
        downhill_moves = np.invert(uphill_moves)

        if acceptance_criteria =='metropolis':
            if self.beta > 0:
                A_s_sp = np.exp(-E_diff*self.beta, where=uphill_moves, out=np.ones_like(E_diff)) #only compute exp for uphill moves, downhill ones are filled with 1 anyway
            if self.beta == math.inf:
            # reject all uphill, accept all others (level and downhill)
                A_s_sp = np.where(uphill_moves, 0.0, 1.)

        Transition_mat = np.multiply(A_s_sp, Proposal_mat)  #note not np.dot but elem wise multiplication
        np.fill_diagonal(Transition_mat, 0)
        diag = np.ones(2**self.no_spins) - np.sum(Transition_mat, axis=0) # This step just fills the diag elems with 1-row_sum. This ensures that row_sum=1 for Transition mat
        Transition_mat = Transition_mat + np.diag(diag)
        return Transition_mat


    def generate_MCMC_trajectories(self, transition_matrix, starting_config:np.ndarray):
        #This function given a transition matrix (local, uniform, quantum, Haar) anything..can choose a configuration with the prob given along rows of transition matrix
        number_chosen = np.random.choice(np.arange(2**self.no_spins), p=transition_matrix[:,starting_config])
        return self.get_int_to_spinconfig(number_chosen, self.no_spins)


    def get_abs_spectral_gap_from_transition_mat(self, Transition_mat):
        #Returns the absolute spectral gap of transition_mat

        dist = np.sort(1-np.abs(scipy.linalg.eigvals(Transition_mat)))
        delta = np.min(dist[1:])

        return delta

    def get_int_to_spinconfig(self, number:int, n:int):
        # Let i be the input for number entry above. This number is between 0 to 2^(n)-1 where n is the positive integer power deciding the number of slots in the binary rep of i
        # The output of this function will be [1,-1]^n which will be achieved by converting i to binary first and replacing every 0 bit with -1
        #Thus if i=0 === [-1,-1,-1....-1_n]
        #        i=1 === [-1,-1,-1....1_n]   # least sig bit flipped to 1
        #      i=2^n -1 == [1,1,1....1_n]    # all bits flipped
        if number < 0 or number > (2**n-1):
            raise ValueError('number out of range')

        bin_list = list( np.binary_repr(number,n))
        return np.array([-1 if bit=='0' else 1 for bit in bin_list])


    def get_spinconfig_to_int(self, spinconfig:np.ndarray):
        #The inverse of int_to_spinconfig. Example: if spinconfig is int(np.array([-1,-1,1])) then output is 1. Given a config it first
        # undo the array as -1 is changed to 0 and 1 is left alone.
        bit_string = ''.join(['0' if spin==-1 else '1' for spin in spinconfig])
        return int(bit_string, 2)  #the base is binary hence 2


    def get_hamming_dist(self,spinconfig1:np.ndarray,spinconfig2:np.ndarray):
        #config1 and config2 will have to have same number of slots n
        #Computes the Hamming distance between the binary representations of configs given . They are first converted to integers
        int1 = self.get_spinconfig_to_int(spinconfig1)
        int2 = self.get_spinconfig_to_int(spinconfig2)
        diff_int = np.bitwise_xor(int1, int2)
        diff_bin = [int(bit) for bit in np.binary_repr(diff_int)]
        return np.sum(diff_bin)