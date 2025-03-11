#!/usr/bin/env python3 

import argparse, json
import cudaq
import numpy as np
import hamiltonian

#cudaq.set_target('nvidia', option = 'fp64')

####################################
# Calssical pre-processing

############################
# Parser
#############################
# Create the parser
parser=argparse.ArgumentParser()

# Add arguments

parser.add_argument('--obi-file', help='file of the electron one-body integrals', type=str)
parser.add_argument('--tbi-file', help='file of the electron two-body integrals', type=str)
parser.add_argument('--json-file', help='json file', type=str)
parser.add_argument('--cutoff', help='Cutoff of the fermionic hamiltonian coefficient', default= 1e-15,type=float)
parser.add_argument('--verbose', help="Verbose printout", action='store_true', default=False)

args=parser.parse_args()
verbose = args.verbose

##################################################
# Input
#################################################
with open(args.json_file) as f:
    json_data=json.load(f)
f.close()

nelectrons: int=json_data['num_electrons']
norbitals: int=json_data['num_orbitals']
constant: float=json_data['core_energy']

# Total number of qubits
qubits_num: int= 2 * norbitals

print('Total number of qubits: ', qubits_num)

####################################################
## Generate the qubit Hamiltonian for cudaq
####################################################

with open(args.obi_file, mode='rb') as f:
    obi = np.fromfile(f, dtype=complex)

f.close()

with open(args.tbi_file, mode='rb') as f:
    tbi=np.fromfile(f, dtype=complex)

f.close()


# Reshape the one and two electron integrals
h1e=np.zeros((qubits_num,qubits_num), dtype=complex)
h2e=np.zeros((qubits_num,qubits_num,qubits_num,qubits_num), dtype=complex)

count=0
for p in range(qubits_num):
    for q in range(qubits_num):
        h1e[p,q]=obi[count]
        count+=1

count=0
for p in range(qubits_num):
    for q in range(qubits_num):
        for r in range(qubits_num):
            for s in range(qubits_num):
                h2e[p,q,r,s]=tbi[count]
                count+=1

# Compute the spin hamiltonian.
tolerance=args.cutoff
spin_ham=hamiltonian.jordan_wigner_fermion(h1e,h2e,constant,tolerance)

if verbose: print('Total number of pauli hamiltonian terms: ',spin_ham.get_term_count(), flush=True)

#####################################
