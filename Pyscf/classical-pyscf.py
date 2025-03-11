#!/usr/bin/env python3 

import argparse, json
from functools import reduce

try:
    import numpy as np
except:
    print('Numpy should be installed.')
    raise

try:
    from pyscf import gto,scf,cc,fci,ao2mo,mp,mcscf
except:
    print('PySCF should be installed.')
    raise

#############################
# Parser
#############################
# Create the parser
parser=argparse.ArgumentParser()

# Add arguments
parser.add_argument('--xyz', help="xyz file", type=str)
parser.add_argument('--c', help="charge of the system", type=int)
parser.add_argument('--out-file-name', help='base file name for output data.', type=str)
parser.add_argument('--s', help="no. of unpaired electrons (2 * s)", type=int)
parser.add_argument('--UR', help="Allow using unrestricted spin for calculations", action='store_true', default=False)
parser.add_argument('--basis', help="The basis set", type=str)
parser.add_argument('--nele-cas', help="number of electrons in the active space", default=None, type=int)
parser.add_argument('--norb-cas', help="number of spatial orbitals in the active space", default=None, type=int)
parser.add_argument('--memory', help="Max. memory to use in MB", default=4000.0, type=float)
parser.add_argument('--cycles', help="Max. no. of iterations", default=100, type=int)
parser.add_argument('--symmetry', help="Invoke point group symmetry for molecular calculations", action='store_true', default=False)
parser.add_argument('--initguess', help="Initial guess of the scf calculation", default='minao', type=str)
parser.add_argument('--MP2', help="Run MP2",action='store_true', default=False)
parser.add_argument('--natorb', help="Compute energies using natural orbitals calculated from RMP2", action='store_true', default=False)
parser.add_argument('--casci', help="Run CASCI", action='store_true', default=False)
parser.add_argument('--ccsd', help="Run CCSD",action='store_true', default=False)
parser.add_argument('--casscf', help="Run CASSCF", action='store_true', default=False)
parser.add_argument('--integrals-natorb', help="Compute 1e and 2e integrals using natural orbitals calculated from RMP2", action='store_true', default=False)
parser.add_argument('--integrals-casscf', help="Compute 1e and 2e integrals using CASSCF orbitals", action='store_true', default=False)
parser.add_argument('--verbose', help="Verbose printout", action='store_true', default=False)

#Parse the argument
args=parser.parse_args()
filename=args.out_file_name
verbose = args.verbose 

if (args.nele_cas is None) and (args.norb_cas is not None):
     raise ValueError("WARN: nele_cas is None and norb_cas is not None. nele_cas and norb_cas should be either both None\
                      or have values")

if (args.nele_cas is not None) and (args.norb_cas is None):
    raise ValueError("WARN: nele_cas is not None and norb_cas is None. nele_cas and norb_cas should be either both None\
                      or have values")

########################################################################
# To add (coming soon)

if args.UR and args.nele_cas is None:
    raise ValueError("WARN: Unrestricted spin calculation for the full space is not supported yet on Cudaq.\
                      Only active space is currently supported for the unrestricted spin calculations.")


############################################################
def generate_molecular_spin_ham_r(h1e,h2e,ecore):

    # This function generates the molecular spin Hamiltonian
    # H= E_core+sum_{pq}  h_{pq} a_p^dagger a_q + 
    #                          0.5 * h_{pqrs} a_p^dagger a_q^dagger a_r a_s
    # h1e: one body integrals h_{pq} 
    # h2e: two body integrals h_{pqrs}
    # ecore: constant (nuclear repulsion or core energy in the active space Hamiltonian)

    # Total number of qubits equals the number of spin molecular orbitals
    nqubits= 2*h1e.shape[0]

    # Initialization
    one_body_coeff=np.zeros((nqubits,nqubits))
    two_body_coeff=np.zeros((nqubits,nqubits,nqubits,nqubits))


    for p in range(nqubits//2):
        for q in range(nqubits//2):

            # p & q have the same spin <a|a>= <b|b>=1
            # <a|b>=<b|a>=0 (orthogonal)
            one_body_coeff[2*p,2*q]=h1e[p,q]
            one_body_coeff[2*p+1,2*q+1]=h1e[p,q]

            for r in range(nqubits//2):
                for s in range(nqubits//2):

                    # Same spin (aaaa, bbbbb) <a|a><a|a>, <b|b><b|b>
                    two_body_coeff[2*p,2*q,2*r,2*s]=0.5*h2e[p,q,r,s]
                    two_body_coeff[2*p+1,2*q+1,2*r+1,2*s+1]=0.5*h2e[p,q,r,s]

                    # Mixed spin(abab, baba) <a|a><b|b>, <b|b><a|a>
                    #<a|b>= 0 (orthogoanl)
                    two_body_coeff[2*p,2*q+1,2*r+1,2*s]=0.5*h2e[p,q,r,s]
                    two_body_coeff[2*p+1,2*q,2*r,2*s+1]=0.5*h2e[p,q,r,s]

    return one_body_coeff,two_body_coeff,ecore

###########################################
def generate_molecular_spin_ham_ur(h1e,h2e,h2e_prime,ecore):

    # This function generates the molecular spin Hamiltonian
    # H= E_core+sum_{pq}  h_{pq} a_p^dagger a_q + 
    #                          0.5 * h_{pqrs} a_p^dagger a_q^dagger a_r a_s
    # h1e: one body integrals h_{pq} 
    # h2e: two body integrals h_{pqrs}
    # ecore: constant (nuclear repulsion or core energy in the active space Hamiltonian)

    # Total number of qubits equals the number of spin molecular orbitals
    nqubits= 2*h1e[0].shape[0]
    
    # Initialization
    one_body_coeff=np.zeros((nqubits,nqubits))
    two_body_coeff=np.zeros((nqubits,nqubits,nqubits,nqubits))


    for p in range(nqubits//2):
        for q in range(nqubits//2):

            # p & q have the same spin <a|a>, <b,b>
            one_body_coeff[2*p,2*q]=h1e[0,p,q]
            one_body_coeff[2*p+1,2*q+1]=h1e[1,p,q]
    
            for r in range(nqubits//2):
                for s in range(nqubits//2):

                    # Same spin (aaaa, bbbbb) <a|a><a|a>, <b|b><b|b>
                    two_body_coeff[2*p,2*q,2*r,2*s]=0.5*h2e[0,p,q,r,s]
                    two_body_coeff[2*p+1,2*q+1,2*r+1,2*s+1]=0.5*h2e[2,p,q,r,s]

                    # Mixed spin(abba, baab) <a|a><b|b>, <b|b><a|a>
                    two_body_coeff[2*p+1,2*q,2*r,2*s+1]=0.5*h2e_prime[p,q,r,s]
                    two_body_coeff[2*p,2*q+1,2*r+1,2*s]=0.5*h2e[1,p,q,r,s]

    return one_body_coeff,two_body_coeff,ecore

#############################
## Beginning of simulation
#############################

################################
# Initialize the molecule
################################
mol=gto.M(
    atom=args.xyz,
    spin=args.s,
    charge=args.c,
    basis=args.basis,
    max_memory=args.memory,
    symmetry=args.symmetry,
    output=filename+'-pyscf.log',
    verbose=4
)

##################################
# Mean field (HF)
##################################

if args.UR:
    myhf=scf.UHF(mol)
    myhf.max_cycle=args.cycles
    myhf.chkfile=filename+'-pyscf.chk'
    myhf.init_guess=args.initguess
    myhf.kernel()

    norb = myhf.mo_coeff[0].shape[1]
    if verbose: print('[pyscf] Total number of alpha molecular orbitals = ', norb)
    norb = myhf.mo_coeff[1].shape[1]
    if verbose: print('[pyscf] Total number of beta molecular orbitals = ', norb)

else:
    myhf=scf.RHF(mol)
    myhf.max_cycle=args.cycles
    myhf.chkfile=filename+'-pyscf.chk'
    myhf.init_guess=args.initguess
    myhf.kernel()

    norb = myhf.mo_coeff.shape[1]
    if verbose: print('[pyscf] Total number of orbitals = ', norb)

nelec = mol.nelectron
if verbose: print('[pyscf] Total number of electrons = ', nelec)
if verbose: print('[pyscf] HF energy = ', myhf.e_tot)

##########################
# MP2
##########################
if args.MP2:

    if args.UR:
        mymp=mp.UMP2(myhf)
        mp_ecorr, mp_t2=mymp.kernel()
        if verbose: print('[pyscf] UR-MP2 energy= ', mymp.e_tot)

        if args.integrals_natorb or args.natorb:
        # Compute natural orbitals
            dma,dmb=mymp.make_rdm1()
            noon_a, U_a= np.linalg.eigh(dma)
            noon_b, U_b= np.linalg.eigh(dmb)
            noon_a=np.flip(noon_a)
            noon_b=np.flip(noon_b)

            if verbose: print('Natural orbital (alpha orbitals) occupation number from UR-MP2: ')
            if verbose: print(noon_a)
            if verbose: print('Natural orbital (beta orbitals) occupation number from UR-MP2: ')
            if verbose: print(noon_b)

            natorbs=np.zeros(np.shape(myhf.mo_coeff))
            natorbs[0,:,:]=np.dot(myhf.mo_coeff[0], U_a)
            natorbs[0,:,:]=np.fliplr(natorbs[0,:,:])
            natorbs[1,:,:]=np.dot(myhf.mo_coeff[1], U_b)
            natorbs[1,:,:]=np.fliplr(natorbs[1,:,:])

    else:
        if args.s!=0:
            raise ValueError("WARN: ROMP2 is unvailable in pyscf.")
        else:
            mymp=mp.MP2(myhf)
            mp_ecorr, mp_t2=mymp.kernel()
            if verbose: print('[pyscf] R-MP2 energy= ', mymp.e_tot)

            if args.integrals_natorb or args.natorb:
            # Compute natural orbitals
                noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
                if verbose: print('Natural orbital occupation number from R-MP2: ')
                if verbose: print(noons)

#######################################
# CASCI if active space is defined
# FCI if the active space is None
######################################
if args.casci:

    if args.UR:
        if args.natorb:
            mycasci = mcscf.UCASCI(myhf, args.norb_cas, args.nele_cas)
            mycasci.kernel(natorbs)
            if verbose: print('[pyscf] UR-CASCI energy using natural orbitals= ', mycasci.e_tot)
        else:
            mycasci_mo = mcscf.UCASCI(myhf, args.norb_cas, args.nele_cas)
            mycasci_mo.kernel()
            if verbose: print('[pyscf] UR-CASCI energy using molecular orbitals= ', mycasci_mo.e_tot)


    else:
        if args.nele_cas is None:
            myfci=fci.FCI(myhf)
            result= myfci.kernel()
            if verbose: print('[pyscf] FCI energy = ', result[0])

        else:
            if args.natorb and (args.s==0):
                mycasci = mcscf.CASCI(myhf, args.norb_cas, args.nele_cas)
                mycasci.kernel(natorbs)
                if verbose: print('[pyscf] R-CASCI energy using natural orbitals= ', mycasci.e_tot)

            elif args.natorb and (args.s!=0):
                raise ValueError("WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf.")

            else:
                mycasci_mo = mcscf.CASCI(myhf, args.norb_cas, args.nele_cas)
                mycasci_mo.kernel()
                if verbose: print('[pyscf] R-CASCI energy using molecular orbitals= ', mycasci_mo.e_tot)

########################
# CCSD
########################
if args.ccsd:

    if args.UR:

        mc = mcscf.UCASCI(myhf, args.norb_cas, args.nele_cas)
        frozen=[]
        frozen=[y for y in range(0,mc.ncore[0])]
        frozen+=[y for y in range(mc.ncore[0]+mc.ncas, len(myhf.mo_coeff[0]))]
    
        if args.natorb:
            mycc=cc.UCCSD(myhf,frozen=frozen, mo_coeff=natorbs)
            mycc.max_cycle=args.cycles
            mycc.kernel()
            if verbose: print('[pyscf] UR-CCSD energy of the active space using natural orbitals= ', mycc.e_tot)

        else:
            mycc=cc.UCCSD(myhf,frozen=frozen)
            mycc.max_cycle=args.cycles
            mycc.kernel()
            if verbose: print('[pyscf] UR-CCSD energy of the active space using molecular orbitals= ', mycc.e_tot)

    else:
        if args.nele_cas is None:
            mycc=cc.CCSD(myhf)
            mycc.max_cycle=args.cycles
            mycc.kernel()
            if verbose: print('[pyscf] Total R-CCSD energy = ', mycc.e_tot)

        else:
            mc = mcscf.CASCI(myhf, args.norb_cas, args.nele_cas)
            frozen=[]
            frozen+=[y for y in range(0,mc.ncore)]
            frozen+=[y for y in range(mc.ncore+args.norb_cas, len(myhf.mo_coeff))]
            if args.natorb and (args.s==0):
                mycc=cc.CCSD(myhf,frozen=frozen, mo_coeff=natorbs)
                mycc.max_cycle=args.cycles
                mycc.kernel()
                if verbose: print('[pyscf] R-CCSD energy of the active space using natural orbitals= ', mycc.e_tot)

            elif args.natorb and (args.s!=0):
                raise ValueError("WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf.")

            else:
                mycc=cc.CCSD(myhf,frozen=frozen)
                mycc.max_cycle=args.cycles
                mycc.kernel()
                if verbose: print('[pyscf] R-CCSD energy of the active space using molecular orbitals= ', mycc.e_tot)

#########################
# CASSCF
#########################
if args.casscf:
    if args.nele_cas is None:
        raise ValueError("WARN: You should define the active space.")

    if args.UR:
        if args.natorb:
            mycas = mcscf.UCASSCF(myhf,args.norb_cas,args.nele_cas)
            mycas.max_cycle_macro=args.cycles
            mycas.kernel(natorbs)
            if verbose: print('[pyscf] UR-CASSCF energy using natural orbitals= ', mycas.e_tot)
        else:
            mycas = mcscf.UCASSCF(myhf,args.norb_cas,args.nele_cas)
            mycas.max_cycle_macro=args.cycles
            mycas.kernel()
            if verbose: print('[pyscf] UR-CASSCF energy using molecular orbitals= ', mycas.e_tot)

    else:

        if args.natorb and (args.s==0):
            mycas = mcscf.CASSCF(myhf, args.norb_cas, args.nele_cas)
            mycas.max_cycle_macro=args.cycles
            mycas.kernel(natorbs)
            if verbose: print('[pyscf] R-CASSCF energy using natural orbitals= ', mycas.e_tot)

        elif args.natorb and (args.s!=0):
            raise ValueError("WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf.")

        else:
            mycas = mcscf.CASSCF(myhf, args.norb_cas, args.nele_cas)
            mycas.max_cycle_macro=args.cycles
            mycas.kernel()
            if verbose: print('[pyscf] R-CASSCF energy using molecular orbitals= ', mycas.e_tot)

###################################
# CASCI: FCI of the active space
##################################
if args.casci and args.casscf:

    if args.UR:
        h1e_cas, ecore = mycas.get_h1eff()
        h2e_cas = mycas.get_h2eff()
    
        e_fci, fcivec = fci.direct_uhf.kernel(h1e_cas,h2e_cas,args.norb_cas, args.nele_cas, ecore=ecore)
        if verbose: print('[pyscf] UR-CASCI energy using the casscf orbitals= ', e_fci)

    else:
        if args.natorb and (args.s!=0):
            raise ValueError("WARN: Natural orbitals cannot be computed. ROMP2 is unavailable in pyscf.")
        else:
            h1e_cas, ecore = mycas.get_h1eff()
            h2e_cas = mycas.get_h2eff()

            e_fci, fcivec = fci.direct_spin1.kernel(h1e_cas,h2e_cas,args.norb_cas, args.nele_cas, ecore=ecore)
            if verbose: print('[pyscf] R-CASCI energy using the casscf orbitals= ', e_fci)
    
###################################################################################
# Computation of one- and two- electron integrals for the active space Hamiltonian
###################################################################################
            
if args.UR:
    if args.integrals_natorb:
        mc = mcscf.UCASCI(myhf, args.norb_cas, args.nele_cas)
        h1e, ecore = mc.get_h1eff(natorbs)
        h1e_cas=np.array(h1e)
        h2e=mc.get_h2eff(natorbs)  
        h2e_cas=np.array(h2e)  
        h2e_cas[0] = np.asarray(h2e_cas[0].transpose(0,2,3,1), order='C')
        h2e_cas[1] = np.asarray(h2e_cas[1].transpose(0,2,3,1), order='C')
        h2e_cas[2] = np.asarray(h2e_cas[2].transpose(0,2,3,1), order='C')
        h2e_cas_prime=np.asarray(h2e_cas[1].transpose(2,0,1,3), order='C')
    
    elif args.integrals_casscf:
        if args.casscf:
            h1e, ecore = mycas.get_h1eff()
            h1e_cas=np.array(h1e)
            h2e = mycas.get_h2eff()
            h2e_cas=np.array(h2e)
            h2e_cas[0] = np.asarray(h2e_cas[0].transpose(0,2,3,1), order='C')
            h2e_cas[1] = np.asarray(h2e_cas[1].transpose(0,2,3,1), order='C')
            h2e_cas[2] = np.asarray(h2e_cas[2].transpose(0,2,3,1), order='C')
            h2e_cas_prime=np.asarray(h2e_cas[1].transpose(2,0,1,3), order='C')
        else: 
            raise ValueError("WARN: You need to run casscf. Use --casscf.")

    else:
        mc = mcscf.UCASCI(myhf, args.norb_cas, args.nele_cas)
        h1e, ecore = mc.get_h1eff(myhf.mo_coeff)
        h1e_cas=np.array(h1e)
        h2e=mc.get_h2eff(myhf.mo_coeff)
        h2e_cas=np.array(h2e)
        h2e_cas[0] = np.asarray(h2e_cas[0].transpose(0,2,3,1), order='C')
        h2e_cas[1] = np.asarray(h2e_cas[1].transpose(0,2,3,1), order='C')
        h2e_cas[2] = np.asarray(h2e_cas[2].transpose(0,2,3,1), order='C')
        h2e_cas_prime=np.asarray(h2e_cas[1].transpose(2,0,1,3), order='C')


    # Compute the molecular spin electronic Hamiltonian from the 
    # molecular electron integrals 
    obi,tbi,core_energy = generate_molecular_spin_ham_ur(h1e_cas,h2e_cas,h2e_cas_prime,ecore)

else:

    if args.nele_cas is None:
        # Compute the 1e integral in atomic orbital then convert to HF basis
        h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        ## Ways to convert from ao to mo
        #h1e=np.einsum('pi,pq,qj->ij', myhf.mo_coeff, h1e_ao, myhf.mo_coeff)
        h1e=reduce(np.dot, (myhf.mo_coeff.T, h1e_ao, myhf.mo_coeff))
        #h1e=reduce(np.dot, (myhf.mo_coeff.conj().T, h1e_ao, myhf.mo_coeff))

        # Compute the 2e integrals then convert to HF basis
        h2e_ao = mol.intor("int2e_sph", aosym='1')
        h2e=ao2mo.incore.full(h2e_ao, myhf.mo_coeff)

        # Reorder the chemist notation (pq|rs) ERI h_prqs to h_pqrs
        # a_p^dagger a_r a_q^dagger a_s --> a_p^dagger a_q^dagger a_r a_s 
        h2e=h2e.transpose(0,2,3,1)

        nuclear_repulsion = myhf.energy_nuc()

        # Compute the molecular spin electronic Hamiltonian from the 
        # molecular electron integrals 
        obi,tbi,e_nn = generate_molecular_spin_ham_r(h1e,h2e,nuclear_repulsion)

        # Dump obi and tbi to binary file. 
        obi.astype(complex).tofile(f'{filename}_one_body.dat')
        tbi.astype(complex).tofile(f'{filename}_two_body.dat')

        metadata = {'num_electrons':nelec, 'num_orbitals':norb, 'core_energy':nuclear_repulsion, 'hf_energy':myhf.e_tot}
        with open(f'{filename}_metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        
    else:

        if args.integrals_natorb:
            if args.s!=0:
                raise ValueError("WARN: ROMP2 is unvailable in pyscf.")
            else:
                mc = mcscf.CASCI(myhf, args.norb_cas, args.nele_cas)
                h1e_cas, ecore = mc.get_h1eff(natorbs)
                h2e_cas=mc.get_h2eff(natorbs)
                h2e_cas = ao2mo.restore('1', h2e_cas, args.norb_cas)
                h2e_cas = np.asarray(h2e_cas.transpose(0,2,3,1), order='C')

        elif args.integrals_casscf:
            if args.casscf:
                h1e_cas, ecore = mycas.get_h1eff()
                h2e_cas = mycas.get_h2eff()
                h2e_cas = ao2mo.restore('1', h2e_cas, args.norb_cas)
                h2e_cas = np.asarray(h2e_cas.transpose(0,2,3,1), order='C')
            else: 
                raise ValueError("WARN: You need to run casscf. Use --casscf.")

        else:
            mc = mcscf.CASCI(myhf, args.norb_cas, args.nele_cas)
            h1e_cas, ecore = mc.get_h1eff(myhf.mo_coeff)
            h2e_cas=mc.get_h2eff(myhf.mo_coeff)
            h2e_cas = ao2mo.restore('1', h2e_cas, args.norb_cas)
            h2e_cas = np.asarray(h2e_cas.transpose(0,2,3,1), order='C')
            

        # Compute the molecular spin electronic Hamiltonian from the 
        # molecular electron integrals 
        obi,tbi,core_energy = generate_molecular_spin_ham_r(h1e_cas,h2e_cas,ecore)

        # Dump obi and tbi to binary file. 
        obi.astype(complex).tofile(f'{filename}_one_body.dat')
        tbi.astype(complex).tofile(f'{filename}_two_body.dat')

        metadata = {'num_electrons':args.nele_cas, 'num_orbitals':args.norb_cas, 'core_energy':ecore, 'hf_energy':myhf.e_tot}
        with open(f'{filename}_metadata.json', 'w') as f:
            json.dump(metadata, f)

