import numpy as np

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


def prob_Ising(s, N, poly, log_rho_max=1):
    y_pred = poly[0] * np.ones(len(s))
    y_pred +=  s @ poly[1:1+N]
    y_pred += np.reshape(np.einsum('ki,kj->kij',s,s,optimize='optimal'),(-1,N*N)) @ poly[1+N:]

    return np.exp(y_pred - log_rho_max)


def prob_Ising_nv(s, N, poly, log_rho_max=1):
    y_pred = poly[0]
    y_pred +=  np.dot(s,poly[1:1+N])
    y_pred += np.dot(np.reshape(np.outer(s,s),-1), poly[1+N:])

    return np.exp(y_pred - log_rho_max)


def sampler(s, prob_func, algo='Metropolis_uniform'):
    N=len(s)
    if algo=='Metropolis_uniform':
        s_new = np.random.choice([1,-1],size=N)

        p1 = prob_func(s)
        p2 = prob_func(s_new)

        accept = min(1.0,p2/p1)
        if np.random.rand()<accept:
            return s_new
        else: return s


def Sampling(N,prob_func,sample_size=1000,burn=None,algo='Metropolis_uniform'):
    if algo=='Exact' and 2**N < sample_size:
      s = enum(N)
      prob_dist = prob_func(s)
      prob_dist = prob_dist / np.sum(prob_dist)
      #samples = np.random.choice(np.arange(2**N), size=sample_size, p=prob_dist)
      #prob_mat, _ = np.histogram(samples, bins=np.arange(2**N+1))
      return prob_mat #/sample_size

    s=np.random.choice([1,-1],size=N)

    if burn is None:
      burn = sample_size//10

    #tm=time.time()
    for _ in range(burn):
      s = sampler(s, prob_func, algo='Metropolis_uniform')
    #print("\n#Burn Complete \n\n")
    #print("\t\t\t\t\t\t Burn Time: ", time.time()-tm)

    #tm=time.time()
    if 2**N < sample_size:
      prob_mat = np.zeros(2**N)

      for _ in range(sample_size):
        s = sampler(s,prob_func, algo='Metropolis_uniform')
        prob_mat[spin_to_key_nv(s)]+=1
      prob_mat = prob_mat / np.sum(prob_mat)
      return prob_mat

    else: 
      prob_dict = {}

      for _ in range(sample_size):
        s = sampler(s,prob_func, algo='Metropolis_uniform')
        key = spin_to_key_nv(s)
        if key in prob_dict: prob_dict[key]+=1
        else: prob_dict[key]=1
      
      for key in prob_dict.keys():
        prob_dict[key] = prob_dict[key] / sample_size

      return prob_dict
    #print("\t\t\t\t\t\t Metropolis Sampling: ", time.time()-tm)
