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
def Trotter_circuit(N: int, k:int, angles_ry:np.ndarray, angles_u3:np.ndarray, angles_2q:np.ndarray):  #list[int]
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
    # k : Trotter repeat length

    qreg=cudaq.qvector(N)

    for i in range(N):
        ry(angles_ry[i], qreg[i])

    for _ in range(k-1):
        for i in range(N):
            u3(angles_u3[i*3], angles_u3[i*3+1], angles_u3[i*3+2], qreg[i])

        for i in range(N):
            for j in range(i + 1, N): 
                two_qubit_gate(angles_2q[i*N+j], qreg[i], qreg[j])

    for i in range(N):
        u3(angles_u3[i*3], angles_u3[i*3+1], angles_u3[i*3+2], qreg[i])







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



    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
    # k : Trotter repeat length


def Trotter_circuit_builder(N: int, k:int, angles_ry:np.ndarray, angles_u3:np.ndarray, angles_2q:np.ndarray, shots_count: int):  #list[int]
    # This is the actual Trotter circuit. Here the circuit construction for Trotterized version of time evolution happens
    # k : Trotter_repeat_length

    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(N)

    for i in range(N):
        kernel.ry(angles_ry[i], qreg[i])

    for _ in range(k-1):
        for i in range(N):
            theta, phi, lam = angles_u3[i*3: i*3+3]
            kernel.u3(theta, phi, lam, qreg[i])

        for i in range(N):
            for j in range(i + 1, N): 
                two_qubit_gate_builder(kernel, angles_2q[i*N+j], qreg[i], qreg[j])

    for i in range(N):
        theta, phi, lam = angles_u3[i*3: i*3+3]
        kernel.u3(theta, phi, lam, qreg[i])

    kernel.mz(qreg)
    return cudaq.sample(kernel, shots_count=shots_count)

