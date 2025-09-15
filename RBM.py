import numpy as np
import time
from spin_utils import get_conn

class RBM:
    def __init__(self, N, M, X=[], seed=1, dtype='complex'):
        self.N = N
        self.M = M
        self.beta = 1
        np.random.seed(seed)

        if len(X) == 0: 
            if dtype=='complex': self.X = np.random.randn(2*(N+M+N*M)) / 10
            elif dtype=='real': self.X = np.random.randn(N+M+N*M) / 10
        else: 
            self.X = X

        if dtype=='complex':
            l = len(self.X)//2
            self.a=self.X[:N] + 1j*self.X[l:l+N]
            self.b=self.X[N:N+M] + 1j*self.X[l+N:l+N+M]
            self.w=np.reshape(self.X[N+M:l] + 1j*self.X[l+N+M:],(N,M))
        elif dtype=='real': 
            self.a=self.X[:N] 
            self.b=self.X[N:N+M] 
            self.w=np.reshape(self.X[N+M:],(N,M))
    
    def enum(self,N):
        N=self.N
        return self.key_to_spin(np.arange(2**N))
    
    def get_params(self):
        return self.a,self.b,self.w
    
    def key_to_spin(self, key):
        N = self.N
        s = np.zeros((len(key),N),dtype=int)
        for i in range(N):
            s[:,N-i-1] = 1 - 2*(key%2)
            key = key//2
        return s

    def spin_to_key(self, s):
        N=self.N
        key=np.zeros(len(s),dtype=int)

        for i in range(N):
            key+=(2**i)*(1-s[:,N-i-1])//2
        return key
    
    def key_to_spin_nv(self, key):
        s=bin(key)[2:]
        s=(self.N-len(s))*'0' + s
        s=np.array([1-2*int(x) for x in s])
        return s

    def spin_to_key_nv(self, s):
        key=0
        for i in range(len(s)):
            key+=(2**i)*(1-s[self.N-i-1])/2
        return int(key)

    def f(self,s):
        b_vec = np.broadcast_to(self.b, (len(s), self.M))
        return self.beta*(s@self.w + b_vec)

    def f_nv(self,s):
        return self.beta*(s@self.w+self.b)

    def prob_nv(self,s):
        return np.prod(np.abs(np.cosh(self.f_nv(s)))**2) * np.exp(-2*self.beta*(np.dot(s,np.real(self.a))))

    def prob(self,s):
        return np.prod((np.abs(np.cosh(self.f(s)))**2), axis=1) * np.exp(-2*self.beta*(s@np.real(self.a))) 
    
    
    def wavefunction(self, s):
        a,b,w = self.get_params()
        beta=self.beta

        log_psi = -beta*(s@a) + np.sum(np.log(np.cosh(beta*(s@w+b))), axis=-1) 

        return np.exp(log_psi)
    

    def sampler(self,s,algo='Metropolis_uniform'):
        N = self.N
        if algo=='Metropolis_uniform':
            p1 = self.prob_nv(s)

        s_new = np.random.choice([1,-1],size=N)
        p2 = self.prob_nv(s_new)

        accept = min(1.0,p2/p1)

        if np.random.rand()<accept:
            return s_new
        else: return s


    def sampling(self,sample_size=1000,burn=None,algo='Metropolis_uniform'):
        N=self.N

        if algo=='Exact':
            s = self.enum(N)
            prob_dist = self.prob(s)
            prob_dist = prob_dist / np.sum(prob_dist)
            samples = np.random.choice(np.arange(2**N), size=sample_size, p=prob_dist)
            #prob_mat, _ = np.histogram(samples, bins=np.arange(2**N+1))
            return samples #prob_mat/sample_size

        s=np.random.choice([1,-1],size=N)
        samples = []

        # prob_mat = np.zeros(2**N)

        if burn is None:
            burn = sample_size//10

        for k in range(burn):
            s = self.sampler(s,algo='Metropolis_uniform')

        for k in range(sample_size):
            s = self.sampler(s,'Metropolis_uniform')
            samples.append(s)
        #     prob_mat[self.spin_to_key_nv(s)]+=1
        # prob_mat = prob_mat / np.sum(prob_mat)

        # s=self.enum(self.N)
        # psi = self.wavefunction(s)
        # psi = psi / np.linalg.norm(psi)

        # plt.plot(prob_mat)
        # plt.plot(psi*psi.conj())
        # plt.show()

        return np.array(samples) #prob_mat
    

    def local_op(self,O,s):
        s2, val = get_conn(O,s)
        loc_op = np.zeros(len(val), dtype=complex)

        for i in range(s2.shape[1]):
            loc_op += val[:,i] * self.wavefunction(s2[:,i]) / self.wavefunction(s)
        return loc_op

    def Op_sampling(self,O,prob_mat=[],samples=[], method='sampling'):
        if method == 'exact' or len(prob_mat)!=0:
            s = self.enum(self.N)
            if method == 'exact': 
                rho_diag = self.prob(s) 
            else: 
                rho_diag = prob_mat

            E = np.sum(rho_diag * self.local_op(O,s)) / np.sum(rho_diag)
        
        elif len(samples)!=0:
            s=samples
            prob_mat = self.prob(s)

            E = np.sum(self.local_op(O,s)) / len(samples)

        return np.real(E)


    def swap_operator(self, u, v, partition):
        if partition == 0 or partition==self.N: return 1

        u2 = u.copy()
        v2 = v.copy()
        u2[:,:partition], v2[:,:partition] = v[:,:partition], u[:,:partition]

        return u2, v2


    def renyi_entropy_sampling(self, u, v, partition):
        num_samples = len(u)
        
        u2, v2 = self.swap_operator(u, v, partition)
        swap_loc = self.wavefunction(u2) * self.wavefunction(v2) / self.wavefunction(u) / self.wavefunction(v)

        return -np.log2(np.sum(swap_loc) / num_samples) 
      

    # def renyi_entropy(self, partition):
    #     N = self.N
    #     s = self.enum(N)
    #     psi = self.wavefunction(s)
    #     psi = psi / np.linalg.norm(psi)

    #     print(np.sum(np.abs(psi)**2))

    #     swap_expect = 0

    #     v = self.enum(N)

    #     for i in range(2**N):
    #         for j in range(2**N):
    #             v1 = v[i]
    #             v2 = v[j]
    #             swap_expect += np.abs(psi[i]*psi[j])**2 * self.swap_operator_loc(v1, v2, partition)

    #     return -np.log2(swap_expect) 


    def renyi_entropy_exact(self, partition):
        s = self.enum(self.N)
        psi = self.wavefunction(s)
        psi = psi / np.linalg.norm(psi)
       
        psi_reshaped = psi.reshape(2**partition, 2**(self.N - partition))
        
        rho_A = psi_reshaped.conj() @ psi_reshaped.T

        trace_rho_A2 = np.trace(rho_A @ rho_A)

        return -np.log2(np.real(trace_rho_A2))


    def derivative_operator(self,s1,s2,idx):
        beta = self.beta
        a,b,w = self.get_params()

        var, k, m = self.map_idx_to_var(idx)

        def f(s,p):
            return beta * (b[p] + s@w[:,p])

        if var=='ra':
            return -beta*(s1[:,k]+s2[:,k])
        elif var=='ia':
            return -1j*beta*(s1[:,k]-s2[:,k])

        elif var=='rb':
            return beta*(np.tanh(f(s1,k)) + np.tanh(f(s2,k).conj()))    
        elif var=='ib':
            return 1j*beta*(np.tanh(f(s1,k)) - np.tanh(f(s2,k).conj()))

        elif var=='rw':
            return beta*(np.tanh(f(s1,m))*s1[:,k] + np.tanh(f(s2,m).conj())*s2[:,k])
        elif var=='iw':
            return 1j*beta*(np.tanh(f(s1,m))*s1[:,k] - np.tanh(f(s2,m).conj())*s2[:,k])


    def map_idx_to_var(self,idx):
        N=self.N
        M=self.M

        l = len(self.X)//2

        if idx<N: return 'ra', idx, None
        elif idx<N+M: return 'rb', idx-N, None
        elif idx<N+M+N*M:
            indices = idx-(N+M)
            return 'rw', indices//M, indices%M

        elif idx<l+N: return 'ia', idx-l, None
        elif idx<l+N+M: return 'ib', idx-(l+N), None
        elif idx<l+N+M+N*M:
            indices = idx-(l+N+M)
            return 'iw', indices//M, indices%M


    def local_gradient(self,O,s,idx):
        s2, val = get_conn(O,s)
        loc_grad = np.zeros(len(val), dtype=complex)

        for i in range(s2.shape[1]):
            ratio = (self.wavefunction(s2[:,i]) / self.wavefunction(s))  
            grad_op = self.derivative_operator(s2[:,i],s,idx)  

            loc_grad += val[:,i] * grad_op * ratio

        return loc_grad


    def grad_Sampling(self,O,O_val=None,prob_mat=[],samples=[],method='sampling'):
        #Works only for pauli-string operators
        N=self.N

        grad = np.zeros(len(self.X),dtype=complex)

        if method == 'exact' or len(prob_mat)!=0:
            s = self.enum(N)
            if method == 'exact': 
                rho_diag = self.prob(s) 
            else: 
                rho_diag = prob_mat

            for idx in range(len(self.X)):
                derivative_op_diag = np.sum(rho_diag * self.derivative_operator(s,s,idx))
                local_grads = np.sum(rho_diag * self.local_gradient(O,s,idx))

                grad[idx] = (local_grads - O_val * derivative_op_diag)/np.sum(rho_diag)

        elif len(prob_mat) == 0:
            s = samples

            for idx in range(len(self.X)):
                derivative_op_diag = np.sum(self.derivative_operator(s,s,idx))
                local_grads = np.sum(self.local_gradient(O,s,idx))
                grad[idx] = (local_grads - O_val * derivative_op_diag)/len(s)

        return np.real(grad)


    def grad_renyi_entropy_sampling(self, SWAP_val, u, v, partition):
        u2, v2 = self.swap_operator(u, v, partition)
        swap_loc = self.wavefunction(u2) * self.wavefunction(v2) / self.wavefunction(u) / self.wavefunction(v)
        
        grad = np.zeros(len(self.X),dtype=complex)

        for idx in range(len(self.X)):
            grad_op = self.derivative_operator(u2,u,idx) + self.derivative_operator(v2,v,idx)   #Order of indices were changed
            local_grads = np.sum(swap_loc * grad_op)

            derivative_op_diag = np.sum(self.derivative_operator(u,u,idx) + self.derivative_operator(v,v,idx))
        
            grad[idx] = (local_grads - SWAP_val * derivative_op_diag)/len(u)

        return - grad / SWAP_val / np.log(2)
    
      
