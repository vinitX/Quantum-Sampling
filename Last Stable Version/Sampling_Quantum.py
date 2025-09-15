import os
import numpy as np
#import netket as nk
from CudaQ.get_conn import get_conn
import scipy as sp
import numpy.linalg as la
import scipy.linalg as spla
import time
import matplotlib.pyplot as plt


class RBM():
  def __init__(self,X,N,M,D,beta=1,vv=False):
    self.N=N
    self.M=M
    self.D=D
    self.beta=beta

    l = len(X)//2

    self.X=X
    self.a=X[:N]+1j*X[l:l+N]
    self.b=X[N:N+M]+1j*X[l+N:l+N+M]
    self.w=np.reshape(X[N+M:N+M+N*M]+1j*X[l+N+M:l+N+M+N*M],(N,M))
    self.u=np.reshape(X[N+M+N*M:N+M+N*M+N*D]+1j*X[l+N+M+N*M:l+N+M+N*M+N*D],(N,D))
    self.d=X[N+M+N*M+N*D:N+M+N*M+N*D+D]+1j*X[l+N+M+N*M+N*D:l+N+M+N*M+N*D+D]
    self.c=np.reshape(X[N+M+N*M+N*D+D:l]+1j*X[l+N+M+N*M+N*D+D:],(N,N))

    self.vv = vv
    if self.vv==False: c=np.zeros((N,N))

  def get_params(self):
    return self.a,self.b,self.w,self.u,self.d,self.c

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


  def enum(self,N):
    N=self.N
    return self.key_to_spin(np.arange(2**N))

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

  def f(self,s):
    beta = self.beta
    b_vec = np.broadcast_to(self.b, (len(s), self.M))
    return beta*(s@self.w + b_vec)

  def g(self,s):
    beta = self.beta
    d_vec = np.broadcast_to(self.d, (len(s), self.D))
    return beta*(2*s@np.real(self.u) + 2*np.real(d_vec))

  def f_nv(self,s):
    beta = self.beta
    return beta*(s@self.w+self.b)

  def g_nv(self,s):
    beta = self.beta
    return beta*(2*s@np.real(self.u)+2*np.real(self.d))


  def log_rho_diag_nv(self,s):
    a,b,w,u,d,c = self.get_params()
    return np.real(np.sum(np.log(np.cosh(self.f_nv(s))*np.cosh((self.f_nv(s)).conj())))) -2*self.beta*(np.dot(s,np.real(a)))  #-2*beta*(s.T@np.real(c)@s) + np.sum(np.log(np.cosh(self.g(s))))

  def log_rho_diag(self,s):
    a,b,w,u,d,c = self.get_params()
    return np.real(np.sum(np.log(np.cosh(self.f(s))*np.cosh((self.f(s)).conj())), axis=1)) -2*self.beta*(s@np.real(a))  #-2*beta*(s.T@np.real(c)@s) + np.sum(np.log(np.cosh(self.g(s))))

  def log_rho_diag_pred_nv(self,s,params):
    a,b,w,u,d,c = self.get_params()
    N=self.N
    beta=self.beta

    y_pred = params[0]
    y_pred +=  np.dot(s,params[1:1+N])
    y_pred += np.dot(np.reshape(np.outer(s,s),-1), params[1+N:])

    return y_pred

  def log_rho_diag_pred(self,s,params):
    a,b,w,u,d,c = self.get_params()
    N=self.N
    beta=self.beta

    y_pred = params[0] * np.ones(len(s))
    y_pred +=  s @ params[1:1+N]
    y_pred += np.reshape(np.einsum('ki,kj->kij',s,s,optimize='optimal'),(-1,N*N)) @ params[1+N:]

    return y_pred

  def build_surrogate(self):
    self.choose_config()
    self.approx_linear()
    self.approx_BFGS() #(init_params = np.zeros(2+2*M))


  def choose_config(self, k=None, num_rand=None):
    #k: Number of best configs.
    #num_rand: Number of random configs
    a,b,w,u,d,c = self.get_params()
    N=self.N
    M=self.M
    beta=self.beta

    if k is None:
      k = N**2//2
      num_rand = 3*N**2//2
    num_rand = min(num_rand,2**N)

    config = np.zeros((1+2*M, N),dtype=int)
    keys = np.zeros(1+2*M, dtype=int)
    phi = np.zeros(1+2*M)

    config[0] = -np.sign(np.real(a))
    config[1:1+M] = np.sign(np.real(w)).T
    config[1+M:1+2*M] = -np.sign(np.real(w)).T

    result_config = []
    result_keys = []

    phi = self.log_rho_diag(config)
    keys = self.spin_to_key(config)

    #print(keys)
    #Keeping only unique configurations
    keys, idx = np.unique(keys, return_index = True)
    config = config[idx]
    phi = phi[idx]
    #print(keys)

    #Sorting configs w.r.t \phi(v) values
    idx = np.argsort(phi)
    phi = phi[idx]
    config = config[idx]
    keys = keys[idx]

    for i in range(k):
      if len(config) == 0: break
      curr_config = config[-1]
      result_config.append(curr_config)
      result_keys.append(self.spin_to_key_nv(curr_config))

      config = np.delete(config,-1,axis=0)
      keys = np.delete(keys,-1,axis=0)
      phi = np.delete(phi,-1,axis=0)

      for j in range(N):
        s_new = np.copy(curr_config)
        s_new[j] = -s_new[j]     #Single-site perturbation

        phi_j = self.log_rho_diag_nv(s_new)
        key_j = self.spin_to_key_nv(s_new)

        if key_j in keys or key_j in result_keys: continue

        #Merging
        ii = np.searchsorted(phi, phi_j)
        if ii==0: continue

        phi = np.insert(phi, ii, phi_j)
        keys = np.insert(keys, ii, key_j)
        config = np.insert(config, ii, s_new, axis=0)

      #Truncating.
      phi = phi[-k+i:]
      keys = keys[-k+i:]
      config = config[-k+i:]

      #print(self.log_rho_diag(curr_config))
      #print(phi)

    #Appending Random Cofigs.
    random_keys = np.random.randint(0, 2**N, size=num_rand)
    #random_keys = np.random.choice(np.arange(2**N),size=num_rand,replace=False)
    result_keys = np.unique(np.concatenate((result_keys, random_keys)))
    #print(len(result_keys))

    result_config = self.key_to_spin(result_keys)
    self.config = np.array(result_config)

    global config_list_vec
    config_list_vec = np.array(result_keys)

    y_actual = self.log_rho_diag(self.config)
    self.log_rho_max = np.max(y_actual)


  def approx_linear(self, eps = 1e-4):
    config = self.config
    a,b,w,u,d,c = self.get_params()
    N=self.N
    M=self.M
    beta=self.beta

    y_actual = self.log_rho_diag(config)
    A = np.zeros((len(config),1+N+N**2))

    A[:,0] = 1
    A[:,1:N+1] = config
    A[:,N+1:] = np.reshape(np.einsum('ki,kj->kij',config,config,optimize='optimal'),(-1,N*N))

    W = np.exp(y_actual - self.log_rho_max)
    W = W / np.sum(W)
    W = W + eps
    W = np.diag(np.sqrt(W))

    params = la.lstsq(W@A, W@y_actual, rcond=None)
    params = params[0]

    self.poly = params
    #print(params)

  def approx_BFGS(self, init_params=None):
    config = self.config
    a,b,w,u,d,c = self.get_params()
    N=self.N
    M=self.M
    beta=self.beta

    err_hist = []
    def func(params):
      y_pred= self.log_rho_diag_pred(config, params)
      y_actual = self.log_rho_diag(config)

      err = np.exp(y_pred-self.log_rho_max) - np.exp(y_actual-self.log_rho_max)
      err = np.sqrt(np.mean(err**2))
      err_hist.append(err)
      return err

    if init_params is None:
      init_params = self.poly

    #print("Linear Fit Error:  ", func(init_params))
    res = sp.optimize.minimize(func, init_params, method='BFGS')
    #print("BFGS Fit Error:  ", res.fun)

    #plt.plot(err_hist)
    #plt.show()
    if res.fun < func(init_params):
      self.poly = res.x


  def reduced_density_matrix(self,s1,s2):
    a,b,w,u,d,c = self.get_params()
    N=self.N
    M=self.M
    D=self.D
    beta=self.beta

    s1_vec = np.zeros(np.shape(s2))
    for k in range(np.shape(s2)[1]):
      s1_vec[:,k] = s1

    def gamma(s1, s2):
      b_vec = np.broadcast_to(b, (np.shape(s2)[:-1] + (M,)))
      return np.cosh(beta*(s1@w+b_vec)) * np.cosh(beta*(s2@w+b_vec)).conj()

    def pi(s1,s2):
      d_vec = np.broadcast_to(d, (np.shape(s2)[:-1] + (D,)))
      return np.cosh(beta*(s1@u+s2@u.conj()+2*np.real(d_vec)))

    log_rho = -beta*(s1_vec@a+s2@a.conj()) + np.sum(np.log(gamma(s1_vec,s2)), axis=-1) + np.sum(np.log(pi(s1_vec,s2)), axis=-1)

    if self.vv == True:
      log_rho -= beta*(np.einsum('ki,ij,kj->k',s1,c,s2,optimize='optimal') + np.einsum('ki,ij,kj->k',s2,c.conj(),s2,optimize='optimal'))
    return np.exp(log_rho - self.log_rho_max)



  def reduced_density_matrix_nv(self,s1,s2):
    a,b,w,u,d,c = self.get_params()
    N=self.N
    M=self.M
    D=self.D
    beta=self.beta

    s1_vec = np.broadcast_to(s1, np.shape(s2))

    def gamma(s1, s2):
      b_vec = np.broadcast_to(b, (len(s2), M))
      return np.cosh(beta*(s1@w+b_vec)) * np.cosh(beta*(s2@w+b_vec)).conj()

    def pi(s1,s2):
      d_vec = np.broadcast_to(d, (len(s2), D))
      return np.cosh(beta*(s1@u+s2@u.conj()+2*np.real(d_vec)))

    log_rho = -beta*(s1@a+s2@a.conj()) + np.sum(np.log(gamma(s1,s2)), axis=1) + np.sum(np.log(pi(s1,s2)), axis=1)

    if self.vv == True:
      log_rho -= beta*(np.einsum('ki,ij,kj->k',s1,c,s2,optimize='optimal') + np.einsum('ki,ij,kj->k',s2,c.conj(),s2,optimize='optimal'))

    return np.exp(log_rho - self.log_rho_max)


  def reduced_density_matrix_old(self,s1,s2):
    a,b,w,u,d,c = self.get_params()
    N=self.N
    beta=self.beta

    def gamma(s):
      return np.cosh(beta*(s@w+b))

    def pi(s1,s2):
      return np.cosh(beta*(s1@u+s2@u.conj()+2*np.real(d)))

    log_rho=np.zeros(len(s2),dtype=complex)
    for i in range(len(s2)):
      log_rho[i] = -beta*(np.dot(s1,a)+np.dot(s2[i],a.conj()))  +  -beta*(s1.T@c@s1+s2[i].T@c.conj()@s2[i]) + np.sum(np.log(gamma(s1) * gamma(s2[i]).conj())) + np.sum(np.log(pi(s1,s2[i])))
      #rho[i]=np.exp(-beta*(np.dot(s1,a)+np.dot(s2[i],a.conj()))) * np.exp(-beta*(s1.T@c@s1+s2[i].T@c.conj()@s2[i])) * gamma(s1) * gamma(s2[i]).conj() * pi(s1,s2[i])
    return np.exp(log_rho - self.log_rho_max)


  def prob(self,s):
    return np.exp(self.log_rho_diag_pred(s,self.poly) - self.log_rho_max)

  def prob_nv(self,s):
    return np.exp(self.log_rho_diag_pred_nv(s,self.poly) - self.log_rho_max)

  def kernel(self,s):
    return np.exp(self.log_rho_diag(s) - self.log_rho_diag_pred(s,self.poly))


  def median_filter(self,x,k=7):
    x_new = []
    x_iter = []
    for i in range(len(x)-k+1):
      x_new.append(np.median(x[i:i+k]))
      x_iter.append(i+np.argsort(x[i:i+k])[k//2])
    return np.array(x_new), np.array(x_iter)


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


  def local_energy(self,H,s,H_terms=None):
    N = self.N
    H_terms = 2*N  

    #tm = time.time()
    s2 = np.zeros((len(s),H_terms,N))
    val = np.zeros((len(s),H_terms),dtype=complex)
    for i in range(len(s)):
      temp_spin, temp_val = get_conn(H,s[i])
      s2[i,:len(temp_spin)] = temp_spin
      val[i,:len(temp_val)] = temp_val
      #s2[i], val[i] = get_conn(H,s[i])
    #print("\t\t\t\t\t\t get_conn() run-time: ",time.time()-tm)

    rho_conn = self.reduced_density_matrix(s,s2)

    return np.real(np.sum(val*rho_conn,axis=1)/self.reduced_density_matrix(s,np.reshape(s,(len(s),1,N)))[:,0])


  def sampling(self,H,sample_size=1000,burn=None,algo='Metropolis_uniform',exact_dist=None):
    N=self.N

    if algo=='Exact' and 2**N < sample_size:
      s = self.enum(N)
      prob_dist = self.prob(s)
      prob_dist = prob_dist / np.sum(prob_dist)
      samples = np.random.choice(np.arange(2**N), size=sample_size, p=prob_dist)
      prob_mat, _ = np.histogram(samples, bins=np.arange(2**N+1))
      return prob_mat/sample_size


    s=np.random.choice([1,-1],size=N)

    if burn is None:
      burn = sample_size//10

    #tm=time.time()
    for k in range(burn):
      s = self.sampler(s,algo='Metropolis_uniform')
    #print("\n#Burn Complete \n\n")
    #print("\t\t\t\t\t\t Burn Time: ", time.time()-tm)

    #tm=time.time()
    if 2**N < sample_size:
      prob_mat = np.zeros(2**N)

      for k in range(sample_size):
        s = self.sampler(s,'Metropolis_uniform')
        prob_mat[self.spin_to_key_nv(s)]+=1
      prob_mat = prob_mat / np.sum(prob_mat)
      return prob_mat

    else: 
      prob_dict = {}

      for k in range(sample_size):
        s = self.sampler(s,'Metropolis_uniform')
        key = self.spin_to_key_nv(s)
        if key in prob_dict: prob_dict[key]+=1
        else: prob_dict[key]=1
      
      for key in prob_dict.keys():
        prob_dict[key] = prob_dict[key] / sample_size


      return prob_dict
    #print("\t\t\t\t\t\t Metropolis Sampling: ", time.time()-tm)




  def Energy_exact(self,H):
    s = self.enum(self.N)
    rho_diag = self.prob(s) * self.kernel(s)
    E = np.sum(rho_diag * self.local_energy(H,s)) / np.sum(rho_diag)
    return np.real(E)
  
  def Energy_sampling(self,H,prob_dict={},prob_mat=[]):
    if len(prob_dict) > 0:
      keys = np.array(list(prob_dict.keys()))
      s = self.key_to_spin(keys)
      prob_mat = np.array(list(prob_dict.values()))
    elif len(prob_mat) > 0:
      s = self.enum(self.N)
    else: 
      print("Provide either a dictionary or a vector of probabilities")
      return 0

    #tm=time.time()    
    local_energies = self.local_energy(H,s)
    kernels = self.kernel(s)
    #print("\t\t\t\t\t\t Local Energy / Kernel Computation: ", time.time()-tm)

    E = np.real(np.sum(local_energies*kernels*prob_mat)/np.sum(kernels*prob_mat))
    return E


  def derivative_operator(self,s1,s2,idx):
    beta = self.beta
    a,b,w,u,d,c = self.get_params()

    s1_vec = np.zeros(np.shape(s2))
    for k in range(np.shape(s2)[1]):
      s1_vec[:,k] = s1

    var, k, m = self.map_idx_to_var(idx)

    def f(s,p):
      return beta * (b[p] + s@w[:,p])

    def g(p):
      return beta*(2*np.real(d[p])+s1_vec@u[:,p]+s2@u[:,p].conj())

    if var=='ra':
      return -beta*(s1_vec[:,:,k]+s2[:,:,k])
    elif var=='ia':
      return -1j*beta*(s1_vec[:,:,k]-s2[:,:,k])

    elif var=='rc':
      if k==m: return np.zeros(np.shape(s2)[:-1]) #No self-interaction
      if self.vv == False: return np.zeros(np.shape(s2)[:-1])
      return -beta*(s1_vec[:,:,k]*s1_vec[:,:,m]+s2[:,:,k]*s2[:,:,m])
    elif var=='ic':
      if k==m: return np.zeros(np.shape(s2)[:-1]) #No self-interaction
      if self.vv == False: return np.zeros(np.shape(s2)[:-1])
      return -1j*beta*(s1_vec[:,:,k]*s1_vec[:,:,m]-s2[:,:,k]*s2[:,:,m])

    elif var=='rb':
      return beta*(np.tanh(f(s1_vec,k)) + np.tanh(f(s2,k).conj()))
    elif var=='ib':
      return 1j*beta*(np.tanh(f(s1_vec,k)) - np.tanh(f(s2,k).conj()))

    elif var=='rw':
      return beta*(np.tanh(f(s1_vec,m))*s1_vec[:,:,k] + np.tanh(f(s2,m).conj())*s2[:,:,k])
    elif var=='iw':
      return 1j*beta*(np.tanh(f(s1_vec,m))*s1_vec[:,:,k] - np.tanh(f(s2,m).conj())*s2[:,:,k])

    elif var=='rd':
      return 2*beta*np.tanh(g(k))
    elif var=='id':
      return 0

    elif var=='ru':
      return beta*(s1_vec[:,:,k]+s2[:,:,k])*np.tanh(g(m))
    elif var=='iu':
      return 1j*beta*(s1_vec[:,:,k]-s2[:,:,k])*np.tanh(g(m))

  def map_idx_to_var(self,idx):
    N=self.N
    M=self.M
    D=self.D

    l = len(self.X)//2

    if idx<N: return 'ra', idx, None
    elif idx<N+M: return 'rb', idx-N, None
    elif idx<N+M+N*M:
      indices = idx-(N+M)
      return 'rw', indices//M, indices%M
    elif idx<N+M+N*M+N*D:
      indices = idx-(N+M+N*M)
      return 'ru', indices//D, indices%D
    elif idx<N+M+N*M+N*D+D: return 'rd', idx-(N+M+N*M+N*D), None
    elif idx<l:
      indices = idx-(N+M+N*M+N*D+D)
      return 'rc', indices//N, indices%N

    elif idx<l+N: return 'ia', idx-l, None
    elif idx<l+N+M: return 'ib', idx-(l+N), None
    elif idx<l+N+M+N*M:
      indices = idx-(l+N+M)
      return 'iw', indices//M, indices%M
    elif idx<l+N+M+N*M+N*D:
      indices = idx-(l+N+M+N*M)
      return 'iu', indices//D, indices%D
    elif idx<l+N+M+N*M+N*D+D: return 'id', idx-(l+N+M+N*M+N*D), None
    elif idx<2*l:
      indices = idx-(l+N+M+N*M+N*D+D)
      return 'ic', indices//N, indices%N

  def local_gradient(self,H,s,idx,H_terms=None):
    N = self.N
    H_terms = 2*N 

    #tm = time.time()
    s2 = np.zeros((len(s),H_terms,N))
    val = np.zeros((len(s),H_terms),dtype=complex)
    for i in range(len(s)):
      temp_spin, temp_val = get_conn(H,s[i])
      s2[i,:len(temp_spin)] = temp_spin
      val[i,:len(temp_val)] = temp_val
      #s2[i], val[i] = get_conn(H,s[i])
    #global total_get_conn_time
    #total_get_conn_time += (time.time()-tm)
    #print("\t\t\t\t\t\t get_conn() run-time: ",time.time()-tm)

    rho_conn = self.reduced_density_matrix(s,s2)
    grad_conn = self.derivative_operator(s,s2,idx)

    s_vec = np.reshape(s, (len(s),1,N))
    rho_diag = self.reduced_density_matrix(s,s_vec)[:,0]

    return np.sum(val * grad_conn * rho_conn, axis=1) / rho_diag

  def grad_exact(self,H):
    N=self.N
    s = self.enum(N)

    rho_diag = self.prob(s) * self.kernel(s)
    E = np.sum(rho_diag * self.local_energy(H,s)) / np.sum(rho_diag)

    s_vec = np.reshape(s, (len(s),1,N))

    derivative_op_diag = np.zeros(len(self.X),dtype=complex)
    grad = np.zeros(len(self.X),dtype=complex)

    #tm=time.time()
    for idx in range(len(self.X)):
      if self.vv == False:
        var, _, _ = self.map_idx_to_var(idx)
        if var == 'rc' or var == 'ic': continue

      derivative_op_diag[idx] = np.sum(rho_diag * self.derivative_operator(s,s_vec,idx)[:,0])
      grad[idx] = np.sum(rho_diag * self.local_gradient(H,s,idx))

    derivative_op_diag = derivative_op_diag / np.sum(rho_diag)

    #print("\t\t\t\t\t\t Exact Gradient Computation: ", time.time()-tm)

    return grad/np.sum(rho_diag) - E * derivative_op_diag   #np.real()



  def grad_Sampling(self,H,prob_dict={},prob_mat=[],Energy=None):
    if len(prob_dict) > 0:
      keys = np.array(list(prob_dict.keys()))
      s = self.key_to_spin(keys)
      prob_mat = np.array(list(prob_dict.values()))
    elif len(prob_mat) > 0:
      s = self.enum(self.N)
    else: 
      print("Provide either a dictionary or a vector of probabilities")
      return 0

    N=self.N

    derivative_op_diag = np.zeros(len(self.X),dtype=complex)
    local_grads = np.zeros(len(self.X),dtype=complex)
    grad = np.zeros(len(self.X),dtype=complex)

    #tm=time.time()
    local_energies = self.local_energy(H,s)
    kernels = self.kernel(s)

    rho_diag = kernels*prob_mat
    Energy = np.real(np.sum(local_energies*rho_diag)/np.sum(rho_diag))

    s_vec = np.reshape(s, (len(s),1,N))

    for idx in range(len(self.X)):
      if self.vv == False:
        var, _, _ = self.map_idx_to_var(idx)
        if var == 'rc' or var == 'ic': continue

      derivative_op_diag[idx] = np.sum(rho_diag * self.derivative_operator(s,s_vec,idx)[:,0])
      local_grads[idx] = np.sum(rho_diag * self.local_gradient(H,s,idx))

    grad = (local_grads - Energy * derivative_op_diag)/np.sum(rho_diag)

    #print("\t\t\t\t\t\t Local Gradient Computation: ", time.time()-tm)
    return grad