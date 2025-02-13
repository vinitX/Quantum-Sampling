import cudaq
import numpy as np

@cudaq.kernel
def two_qubit_gate(angle:float, qubit_1: cudaq.qubit, qubit_2: cudaq.qubit):  # mode: str = "CNOT_decomposition"  [cudaq doesn't support string type]
    # This function provides circuit description of RZZ(theta) - This is the 2-qubit gate used for H_Z
    x.ctrl(qubit_1, qubit_2)
    rz(angle, qubit_2)
    x.ctrl(qubit_1, qubit_2)

    # if mode == "no_decomposition":   #RZZ is not supported by CudaQ
    #     rzz(angle, qubit_1, qubit_2)

    # if mode == "RZX_decomposition":  #RZX is not supported by CudaQ
    #     h(qubit_2)
    #     rzx(angle, qubit_1, qubit_2)   # can be implemented natively by pulse-stretching
    #     h(qubit_2)


@cudaq.kernel
def Trotter_circuit(N: int, k:int, alpha:float,
                    gamma:float, time_delta: float, theta:np.ndarray, phi:np.ndarray, lam:np.ndarray, J:np.ndarray, initial_config:np.ndarray):  #list[int]
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
    # k : Trotter repeat length

    qreg=cudaq.qvector(N)

    # assert(len(initial_config)==N), "initial config array is not same length as number of spins"
    for i in range(N):
        if int(initial_config[i]) == 1: x(qreg[i])                
        #circuit.x(qreg[N-1-np.argwhere(return_array==1).flatten()])  # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have N-1-index
    
    for _ in range(k-1):
        for qubit in range(N):
            u3(theta[qubit], phi[qubit], lam[qubit], qreg[qubit])

        for i in range(N):
            for j in range(i + 1, N): 
                angle = 2*J[(N-1-i)*N + N-1-j]*(1-gamma)*alpha*time_delta
                two_qubit_gate(angle, qreg[i], qreg[j])

    for qubit in range(N):
        u3(theta[qubit], phi[qubit], lam[qubit], qreg[qubit])




#Builder mode

def two_qubit_gate_builder(kernel, angle:float, qubit_1: cudaq.qubit, qubit_2: cudaq.qubit):  # mode: str = "CNOT_decomposition"  [cudaq doesn't support string type]
    # This function provides circuit description of RZZ(theta) - This is the 2-qubit gate used for H_Z
    kernel.cx(qubit_1, qubit_2)
    kernel.rz(angle, qubit_2)
    kernel.cx(qubit_1, qubit_2)

    # if mode == "no_decomposition":   #RZZ is not supported by CudaQ
    #     rzz(angle, qubit_1, qubit_2)

    # if mode == "RZX_decomposition":  #RZX is not supported by CudaQ
    #     h(qubit_2)
    #     rzx(angle, qubit_1, qubit_2)   # can be implemented natively by pulse-stretching
    #     h(qubit_2)


def Trotter_circuit_builder(N: int, k:int, alpha:float,
                    gamma:float, time_delta: float, theta:np.ndarray, phi:np.ndarray, lam:np.ndarray, J:np.ndarray, initial_config: np.ndarray, shots_count: int):  #list[int]
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
    # k : Trotter_repeat_length

    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(N)

    assert(len(initial_config)==N), "initial config array is not same length as number of spins"
    for i in range(N):
        if initial_config[i] == 1: kernel.x(qreg[i])
                
        #circuit.x(qreg[N-1-np.argwhere(return_array==1).flatten()])  # Qiskit follows endian order with least sig bit as qubit[0] on top which is why we have no_spins-1-index

    for _ in range(k-1):
        for qubit in range(N):
            kernel.u3(theta[qubit], phi[qubit], lam[qubit], qreg[qubit])

        for i in range(N):
            for j in range(i + 1, N): 
                angle = 2*J[(N-1-i)*N + N-1-j]*(1-gamma)*alpha*time_delta
                two_qubit_gate_builder(kernel, angle, qreg[i], qreg[j])

        #circuit.barrier()

    for qubit in range(N):
        kernel.u3(theta[qubit], phi[qubit], lam[qubit], qreg[qubit])

    kernel.mz(qreg)
    return cudaq.sample(kernel, shots_count=shots_count)

