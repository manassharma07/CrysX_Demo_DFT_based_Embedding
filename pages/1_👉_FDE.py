import streamlit as st 
from pyscf import gto, dft, df, lib, scf
# from pyscf.dft import rks
import numpy as np
import py3Dmol
import streamlit.components.v1 as components
import os
import pandas as pd
import scipy

# Some global variables (Yes unfortunately I had to use them)
isSupermolecularBasis = False
kedf = '521'
embPot = 0
isFrozen = True
isXCchange = False
isNucAdensB = True
isNucAauxB = False
isFDEconverged = False
conv_crit = 1e-7

def dump_scf_summary(mf):
    summary = mf.scf_summary
    
    # st.text('**** SCF Summary ****')
    st.write('Total Energy =                    ', mf.e_tot)
    st.write('Nuclear Repulsion Energy =        ', summary['nuc'])
    st.write('One-electron Energy =             ', summary['e1'])
    # st.write('Two-electron Energy =             %24.15f', summary['e2'])
    st.write('Two-electron Coulomb Energy =     ', summary['coul'])
    st.write('DFT Exchange-Correlation Energy = ', summary['exc'])

def nucAux(molA, molB, molTotal, dmatB):
    global isSupermolecularBasis, basisSetA
    #auxmolB = df.addons.make_auxmol(molB, auxbasis='unc-weigend')
    auxmolB = df.addons.make_auxmol(molB, auxbasis='weigend')
    #auxmolB.cart=True
    #molB.cart=True
    # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
    # indices of AO basis and P is the auxiliary basis
    ints_3c2e_BB = df.incore.aux_e2(molB, auxmolB, intor='int3c2e')
    ints_2c2e_BB = auxmolB.intor('int2c2e')
    

    nao = molB.nao
    naux = auxmolB.nao
    print(naux)

    # Compute the DF coefficients (df_coef) of aux basis of B
    df_coef = scipy.linalg.solve(ints_2c2e_BB, ints_3c2e_BB.reshape(nao*nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)
    
    df_coeff = np.einsum('ijk,jk', df_coef, dmatB)
    #print(df_coeff.shape)
    
    #----Create a molecule with a ghost atom with a basis function with exp=0
    #----Then add the atoms of A as they are needed for nuclei of A
    #----These atoms can be assigned any basis since it doesn't matter, but we assign them
    #----just the basis of A so that when we use molA.nbas or something we get the correct number.
    #----Next we assign atoms of B as ghosts to mol, so that they don't contribute any nuclear charges.
    #----These are assigned the auciliary basis of B for which we already have the fitting coeffs.
    #----Finally we just perform the integral of the 0 exponent BF with all the auxiliary BFs
    mol = gto.Mole()
    mol.atom = '''Ghost 0 0 0; '''
    mol.basis = {'Ghost': gto.basis.parse('''
    Ghost    S
         0.0              1.0
    ''')}
    mol.atom = mol.atom + molA.atom + ' ;'
    for i in range(molB.natm):
        coordBohr = molB._atom[i][1]
        coordAngs = [x*0.52917721092 for x in coordBohr]
        mol.atom = mol.atom + ' Ghost'+str(i)+' '+ str(coordAngs)[1:-1]+' ; '
         
    for i in range(molA.natm):
        mol.basis[molA._atom[i][0]] = basisSetA
    for i in range(molB.natm):
        atomName = molB._atom[i][0]
        if isSupermolecularBasis:
            #Strip the 'ghost-' prefix before the atom names 
            if 'GHOST-' in atomName:
                atomName = atomName.replace('GHOST-','')
        
        #mol.basis['Ghost'+str(i)] = gto.basis.load('weigend', molB._atom[i][0])
        #mol.basis['Ghost'+str(i)] = gto.uncontract(gto.basis.load('weigend', atomName))
        mol.basis['Ghost'+str(i)] = gto.basis.load('weigend', atomName)
    #print(mol._basis)
    #print(mol._bas)
    #print(mol.atom)    
    #print(mol.basis)

    #In case the number of electrons in molB is not even, 
    #doing mol.build() gives an error about spin not being 0 and blah blah..
    #Although, we aren't doing spin unrestricted calculations
    # at the moment, here we should check if the nelectrons is even
    # or not and accordingly set the spin of the mol before building it.
    # Since, this mol object is just an auxiliary object to enable the  
    # calculation of nucAauxB energy, so doing changes to spin won't affect the result.
    mol.charge = molB.charge
    #if(molB.nelectrons%2!=0):
    #mol.spin=1
    
    mol.build()
    #mol._libcint_ctr_coeff(0)[:] = 3.544925214180127
    mol._libcint_ctr_coeff(0)[:] = 3.544907701810305
    #for i in range(mol.nbas):
    #    print(mol._libcint_ctr_coeff(i))

    #print(mol._basis)
    #print(mol._bas)

    #----Finally we just perform the integral of the 0 exponent BF with all the auxiliary BFs
    V_nucA_auxB_mat = mol.intor('int1e_nuc', shls_slice=(0,1,molA.nbas+1,mol.nbas))#[0,molA.nao_nr()+1:mol.nao_nr()+1]
    #print(V_nucA_auxB_mat.shape)
    #print(df_coeff.shape)
    #print(mol.natm)
    #print(mol._atom)
    #print(mol.ao_labels())

    
    #for i in range(mol.natm):
    #    print(mol.atom_charge(i))
    
    energy = np.einsum('i,ji',df_coeff,V_nucA_auxB_mat)
    #energy = np.dot(df_coeff,V_nucA_auxB_mat)
    #print(mol.intor('int1e_nuc')[0:5,0:5])
    
    return energy

def coulombMatrixEmbDF(molTotal, molA, molB, dmatB):
    auxmolB = df.addons.make_auxmol(molB, auxbasis='weigend')
    # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
    # indices of AO basis and P is the auxiliary basis
    ints_3c2e_BB = df.incore.aux_e2(molB, auxmolB, intor='int3c2e')
    ints_2c2e_BB = auxmolB.intor('int2c2e')
    
    nao = molB.nao
    naux = auxmolB.nao

    # Compute the DF coefficients (df_coef)
    df_coef = scipy.linalg.solve(ints_2c2e_BB, ints_3c2e_BB.reshape(nao*nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)

    ints_3c2e_AB = df.incore.aux_e2(molA, auxmolB, intor='int3c2e')
    df_coeff = np.einsum('ijk,jk', df_coef, dmatB)
    Jab2 = np.einsum('ijk,k',ints_3c2e_AB,df_coeff)
    
    return Jab2

def coulombMatrixEmb(molTotal, molA, molB, dmatB):
    #Two electron integrals for A+B total system
    TwoE = molTotal.intor('int2e', shls_slice=(0,molA.nbas,0,molA.nbas,molA.nbas,molA.nbas+molB.nbas,molA.nbas,molA.nbas+molB.nbas))
    #Construct the Coulomb matrix of A due to B
    Jab = np.einsum('ijkl,lk',TwoE,dmatB)
    return Jab

def coulombMatrixEmbSuper(molTotal, mfTotal, dmatB):
    Jab, temp = scf.hf.get_jk(molTotal, dmatB, hermi=1, vhfopt=None, with_j=True, with_k=False, omega=None)
    return Jab

def calculateEnergy(dmat, mat):
    energy = np.einsum('pq,qp',dmat,mat)
    return energy

def calculateSemiDFCoeff(mol):
    # Define the auxiliary fitting basis for 3-center integrals. Use the function
    # make_auxmol to construct the auxiliary Mole object (auxmol) which will be
    # used to generate integrals.
    auxbasis = 'weigend'
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
    # indices of AO basis and P is the auxiliary basis
    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
    ints_2c2e = auxmol.intor('int2c2e')

    nao = mol.nao
    naux = auxmol.nao

    # Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)    
    return df_coef, ints_3c2e

def energy_nuc(molA, molB):
    '''Compute nuclear repulsion energy (AU) or static Coulomb energy
    Returns
        float
    '''
    chargesA = molA.atom_charges()
    chargesB = molB.atom_charges()
    coordsA = molA.atom_coords()
    coordsB = molB.atom_coords()
    e = 0
    for j in range(len(molB._atm)):
        q2 = chargesB[j]
        r2 = coordsB[j]
        for i in range(molA.natm):
            q1 = chargesA[i]
            r1 = coordsA[i]
            r = np.linalg.norm(r1-r2)
            if q1!=0 and q2!=0:
                e += q1 * q2 / r
    return e

def coulombMatrixDF(mol, dmat, df_coef=None, ints_3c2e=None):
    if (df_coef is None):
        # Define the auxiliary fitting basis for 3-center integrals. Use the function
        # make_auxmol to construct the auxiliary Mole object (auxmol) which will be
        # used to generate integrals.
        auxbasis = 'weigend'
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
        # indices of AO basis and P is the auxiliary basis
        if (ints_3c2e is None):
            ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
        ints_2c2e = auxmol.intor('int2c2e')

        nao = mol.nao
        naux = auxmol.nao

        # Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
        df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)

    

    df_coeff = np.einsum('ijk,jk', df_coef, dmat)
    Jaa = np.einsum('ijk,k',ints_3c2e,df_coeff)
    #df_eri = lib.einsum('ijP,Pkl->ijkl', ints_3c2e, df_coef)
    #TwoE = df_eri
    #Jaa = np.einsum('ijkl,kl',TwoE, dmat)
    return Jaa

def nucDensB(molA, molB, molTotal, dmatB):
    global isSupermolecularBasis
    if isSupermolecularBasis:
        Vnuctot_Mat = molTotal.intor('int1e_nuc')
    else:
        Vnuctot_Mat = molTotal.intor('int1e_nuc')[molA.nao_nr():, molA.nao_nr():]
    VnucB_mat = molB.intor('int1e_nuc')
    V_eff = Vnuctot_Mat-VnucB_mat
    energy = np.einsum('pq,qp',dmatB,V_eff)
    return energy

def scf1(molB, molA, mfA, mfB, molTotal, mfTotal, dmatB, excB, Jab, Vab, max_cycle, max_memory, dmA=None):
    global kedf #KE functional str
    global embPot
    global isXCchange
    global isFrozen
    global isNucAdensB, isNucAauxB
    global isFDEconverged
    global conv_crit
    
    #INPUT:
    #Jab: Electron(A)-Electron(B) Coulomb Matrix due to B in basis of A
    #Vab: Electron(A)-Nuclear(B) Coulomb Matrix due to B in basis of A
    #dmatB: Density Matrix of B
    #excB:  Exchange-Correlation Energy of B (Needed to subtract from the total Exc of A+B )
    #mfA: RKS object of Mol A (Cluster)
    #mfTotal: RKS object of total System
    #---------------------------------------------------------
    #LOCAL:
    #scf_conv:    is SCF is converged
    #S_AA :       Overlap matrix of A in basis of A
    #---------------------------------------------------------
    scf_conv = False
    ni = mfTotal._numint
    
    
    ekeF_B = 0.0
    if not(kedf=='electro'):
        #KE Func Potential for density of B
        # 'ekeF_B' is needed to calculate the correct energy of A
        n_B, ekeF_B, KEF_B = ni.nr_rks(molB, mfB.grids, kedf, dmatB)
        # n_B, ekeF_B, KEF_B = ni.nr_rks(molTotal, mfTotal.grids, kedf, dmatB)
    #Overlap Matrix of A in basis of A
    S_AA = mfA.get_ovlp(molA)
    #KE + Vnuc matrix of A in basis of A
    H_core_A = mfA.get_hcore(molA)
    #Embedding Potential Matrix (Coulombic)
    EmbdmatPot = Vab + Jab  
    #Debugging
    #EmbdmatPot = EmbdmatPot-EmbdmatPot
    #Initial Density Matrix
    if dmA is not None:
        dmatNew = dmA
    else:
        dmatNew = mfA.get_init_guess(molA, 'minao')
       

    if(isDF):
        temp = calculateSemiDFCoeff(molA)
        df_coef = temp[0]
        int3c2e = temp[1]
    else:
        #TwoEA = molA.intor('int2e')
        J_AA, temp = scf.hf.get_jk(molA, dmatNew, hermi=1, vhfopt=None, with_j=True, with_k=False, omega=None)
        
    
    #Debugging
    #dmatNew = dmA
    e_tot_old = 0.0
    #Non-additive KE
    ekeF_A = 0.0
    ekeF_AandB = 0.0
    nadd_eKE = 0.0
    nAdd_KE_Pot = 0
    cycle = 1
    mfA.diis = scf.CDIIS()
    
    if isinstance(mfA.diis, lib.diis.DIIS):
        diis = mfA.diis

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mfTotal.xc, spin=molA.spin)
    #Fock exchange
    VK = 0
    #Exchange energy
    eK = 0.0
    #Debug
    #isXCchange=True
    #xcName = 'PBE'
        

    while not scf_conv and cycle <= max(1, max_cycle):
        if not(cycle==1):
            if not(isDF):
                #Jdiff = np.einsum('ijkl,lk',TwoEA, dmatNew-dmatA)
                Jdiff, temp = scf.hf.get_jk(molA, dmatNew-dmatA, hermi=1, vhfopt=None, with_j=True, with_k=False, omega=None)
        dmatA = dmatNew

        #Electron-Electron Coulomb Matrix of A in basis of A
        if(isDF):
            J_AA = coulombMatrixDF(molA, dmatA, df_coef, int3c2e)
        else:
            if (cycle ==1):
                J_AA = J_AA
                
            else:
                J_AA = J_AA + Jdiff
        #The XC embedding potential requires the density of A as well,
        #therefore, needs to be updated at each scf iteration
        
        #Construct the total density matrix by having dmatA and dmatB as diagonal blocks of dmatTotal
        if isSupermolecularBasis:
            dmatTotal = dmatA + dmatB
        else:
            dmatTotal = scipy.linalg.block_diag(dmatA, dmatB)
            #print(dmatTotal)
        #-----------------------------------
        #Fock exchange
        #-----------------------------------
        if hyb>1e-10:
            temp, VK = scf.hf.get_jk(molA, dmatA, hermi=1, vhfopt=None, with_j=False, with_k=True, omega=None)
            #print(VK.shape)
            #print(hyb)
            #print(alpha)
            VK *= hyb
            VK = -VK * .5
            eK = np.einsum('ij,ji', dmatA, VK).real * .5 
        #---------------XC STUFF-----------------------------------
        #Calculate EXC(A+B) and XC_Mat(A+B) and nElec(A+B)
        #Original:
        n_AandB1, exc_AandB, XC_AandB = ni.nr_rks(molTotal, mfTotal.grids, mfTotal.xc, dmatTotal, max_memory=max_memory)
        
        #But we only need the matrix elements that belong to the basis functions of A
        XC_AandB_basA = XC_AandB[0:molA.nao_nr(),0:molA.nao_nr()] 
        exc_A1 = 0
        #The following is not exactly nadd energy as exc_A1 is not yet calculated and needs to be subtracted too. 
        #For now we just subtract excB from it and exc_A1=0 so that we don't have to change a lot of equations where energy is being calculated.
        #Also, exc_A1 is not yet calculated as it is not needed if the same XCF is being used for both cluster and embedding potential.
        #exc_A1 gets cancelled out in the FDE equation for energy of cluster, so it saves time.
        nadd_eXC = exc_AandB - excB - exc_A1

        #----------------------------------------------------------

        #---------------KE FUNC STUFF-----------------------------------
        if not(kedf=='electro'):
            #KE Func Potential for total density
            #Original:
            n_AandB2, ekeF_AandB, KEF_AandB = ni.nr_rks(molTotal, mfTotal.grids, kedf, dmatTotal, max_memory=max_memory)
            #Efficient:
            #n_AandB2, ekeF_AandB, KEF_AandB = ni.nr_rks2(molTotal, mfTotal.grids, KEfunc, dmatTotal, molA, max_memory=max_memory)
            #But we only need the matrix elements that belong to the basis functions of A
            KEF_AandB_basA = KEF_AandB[0:molA.nao_nr(),0:molA.nao_nr()]
            #KE Func Potential for density of A
            n_A, ekeF_A, KEF_A = ni.nr_rks(molA, mfA.grids, kedf, dmatA, max_memory=max_memory)
            #Non additive KE potential
            nAdd_KE_Pot = KEF_AandB_basA - KEF_A
            #Non additive KE
            nadd_eKE = ekeF_AandB - ekeF_A - ekeF_B
        #---------------------------------------------------------------

        #Debug
        #n_AandB, exc_AandB, XC_AandB = ni.nr_rks(molA, mfA.grids, mfA.xc, dmatA, max_memory=max_memory)
        #XC_AandB_basA = XC_AandB
        #print(exc_AandB)
        #----------------------------------------------------------------
        #Construct Fock (KSCED) Matrix
        #----------------------------------------------------------------
        if not(kedf=='electro'): #If non-additive KE is needed
            Fo = H_core_A + EmbdmatPot + J_AA + XC_AandB_basA + nAdd_KE_Pot + VK
        else: #If electrostatic embedding is requested
            Fo = H_core_A + EmbdmatPot + J_AA + XC_AandB_basA + VK
        
        diis_start_cycle = mfA.diis_start_cycle
        if cycle >= diis_start_cycle:
            F = diis.update(S_AA, dmatA, Fo)
        #Solve the Fock matrix eigenvalue equation
        mo_energy, mo_coeff = mfA.eig(F, S_AA)
        #mo_energy, mo_coeff = eig(F, S_AA)

            

        #Get occupations
        mo_occ = mfA.get_occ(mo_energy, mo_coeff)
        #print(mo_occ)
        #Construct the new density matrix
        dmatNew = mfA.make_rdm1(mo_coeff, mo_occ)
        #----------------------------------------------------------------
        #saveDensityCube(molTotal, dmatTotal, cycle)
        #Total Energy
        #----------------------------------------------------------------
        e_tot_New = calculateEnergy(dmatA, J_AA)*0.5 + calculateEnergy(dmatA, H_core_A+EmbdmatPot) + exc_A1 + nadd_eXC + nadd_eKE + eK
        print(e_tot_New, flush=True)
        #----------------------------------------------------------------
        if abs(e_tot_New-e_tot_old)<=conv_crit:
            scf_conv = True
        else:
            e_tot_old = e_tot_New
            cycle = cycle + 1        

    isFDEconverged = scf_conv
    #After SCF, perform an extra diagonalization in case of level-shifting to remove it
    mo_energy, mo_coeff = mfA.eig(Fo, S_AA) #Use original Fock matrix Fo
    mo_occ = mfA.get_occ(mo_energy, mo_coeff)
    dmatNew = mfA.make_rdm1(mo_coeff, mo_occ)
    #Total energy = Electron-Electron Coulomb Energy + KE + Nuc-Ele. Energy(only A) + Coulomb Embedding Energy(Vab + Jab) 
    #               +  Exc(A+B)  + NaddKE((A+B_-A)
    #Debugging
    #dmatNew = dmatA
    e_tot_Final = calculateEnergy(dmatNew, J_AA)*0.5 + calculateEnergy(dmatNew, H_core_A+EmbdmatPot) + exc_A1 + nadd_eXC + nadd_eKE + eK
    print('SCF converged', flush=True)
    print('Add the static energies due to the environment:', flush=True)
    if(isDF):
        #print(e_tot_Final + mfA.energy_nuc() + energy_nuc(molA, molB)  + nucAux(molA, molB, molTotal, dmatB) - excB) 
        e_tot_Final = e_tot_Final + mfA.energy_nuc() + energy_nuc(molA, molB)  
    else:
        #print(e_tot_Final + mfA.energy_nuc() + energy_nuc(molA, molB)  + nucDensB(molA, molB, molTotal, dmatB) - excB)
        e_tot_Final = e_tot_Final + mfA.energy_nuc() + energy_nuc(molA, molB)  
    
    if isDF:
        if isNucAdensB:
            e_tot_Final = e_tot_Final + nucDensB(molA, molB, molTotal, dmatB) 
        else:
            e_tot_Final = e_tot_Final + nucAux(molA, molB, molTotal, dmatB)[0] 
    else:
         e_tot_Final = e_tot_Final + nucDensB(molA, molB, molTotal, dmatB)

    #Debug
    #if isDF:
        #print(calculateEnergy(dmatNew, Vab))
        #print(nucAux(molB, molA, molTotal, dmatNew)[0])
        #e_tot_Final = e_tot_Final - calculateEnergy(dmatNew, Vab) + nucAux(molB, molA, molTotal, dmatNew)[0]
        #e_tot_Final = e_tot_Final - calculateEnergy(dmatNew, molA.intor('int1e_nuc')) + nucAauxA(molA, dmatNew)[0] 
        #print(calculateEnergy(dmatNew, Vab) - nucAux(molB, molA, molTotal, dmatNew)[0])
        #print(calculateEnergy(dmatNew, molA.intor('int1e_nuc')) - nucAauxA(molA, dmatNew)[0])
    print('----------------------------------')
    print('|Converged Energy: ',e_tot_Final, flush=True) 
    print('----------------------------------')
    
    ke_cluster = calculateEnergy(dmatNew, molA.intor('int1e_kin'))
    e_nn_cluster = mfA.energy_nuc()
    e_nucA_nucB = energy_nuc(molA, molB)
    e_coul_nucA_densA = calculateEnergy(dmatNew, molA.intor('int1e_nuc'))
    e_coul_densA_densA = calculateEnergy(dmatNew, J_AA)*0.5
    e_nucA_densB = nucDensB(molA, molB, molTotal, dmatB)
    e_coul_densA_nucB_densB = calculateEnergy(dmatNew, EmbdmatPot)
    #Debug
    #print(abs(dmatNew - dmA).max())
    print('Cycles:=\t                                   '+str(cycle), flush=True)
    print('KE cluster=\t                                '+str(ke_cluster), flush=True)
    #print('Enuc cluster ',calculateEnergy(dmatNew, molA.intor('int1e_nuc')))
    #print('Enuc total ',calculateEnergy(dmatNew, molTotal.intor('int1e_nuc')))
    #print('Enuc total - Enuc cluster ',calculateEnergy(dmatNew, molTotal.intor('int1e_nuc')))
    print('Enuc-nuc total=\t                            '+str(mfTotal.energy_nuc()), flush=True)
    print('Enuc-nuc cluster=\t                          '+str(e_nn_cluster), flush=True)
    print('Enuc-nuc total - Enuc-nuc cluster=\t         '+str(e_nucA_nucB), flush=True)
    print('XC_Energy env=\t                             '+str(excB), flush=True)
    print('XC_Energy Total=\t                           '+str(exc_AandB), flush=True)
    print('KEDF_Energy env=\t                           '+str(ekeF_B), flush=True)
    print('KEDF_Energy clus=\t                          '+str(ekeF_A), flush=True)
    print('KEDF_Energy Total=\t                         '+str(ekeF_AandB), flush=True)
    print('KEDF_Energy nadd=\t                          '+str(nadd_eKE), flush=True)
    print('Electrostatic Embedding_Energy Total=\t      '+str(e_coul_densA_nucB_densB), flush=True)
    if isDF:
        if isNucAdensB:
            print('nucAdensB energy=\t                          '+str(e_nucA_densB), flush=True)
            #Debug
            #print('(Debug)nucAauxB energy=\t                          '+str(nucAux(molA, molB, molTotal, dmatB)[0]))
            #print('(Debug)nucAauxB energy=\t                          '+str(nucAux2(molA, molB, molTotal, dmatBprime)[0]))
        else:
            print('nucAauxB energy=\t                          '+str(nucAux(molA, molB, molTotal, dmatB)[0]), flush=True)
            #Debug
            print('(Debug)nucAdensB energy=\t                          '+str(e_nucA_densB), flush=True)
    else:
        print('nucAdensB energy=\t                          '+str(e_nucA_densB), flush=True)
    print('Couenrgy: Enuc+ J clu=\t                     '+str(e_coul_densA_densA + e_coul_nucA_densA), flush=True)
    print('Exchange energy=\t                            '+str(eK), flush=True)

    #print('Embedding Potential')
    # for i in range(molA.nao_nr()):
    #     for j in range(i+1):
    #         print(EmbdmatPot[i,j])
    # for i in range(molB.nao_nr()):
    #     for j in range(i+1):
    #         print(dmatB[i,j])

    # dump_scf(molA, 'fde.chk', e_tot_Final, mo_energy, mo_coeff, mo_occ, overwrite_mol=True)
    
    #XC Func Potential for density of A
    #This is needed if the same XCF is used for the cluster and embedding potential
    #As in this case we skip its calculation during the SCF stage. 
    #But it is still needed to get the correct embedding potential.
    n_A, eXC_A, XC_A2 = ni.nr_rks(molA, mfA.grids, mfA.xc, dmatNew, max_memory=max_memory)
    # n_A, eXC_A, XC_A2 = ni.nr_rks(molTotal, mfTotal.grids, mfA.xc, dmatNew, max_memory=max_memory)
    print('XC_Energy Cluster                            '+str(eXC_A), flush=True)   #XC energy of A using xcName (env)
    nadd_eXC = exc_AandB - eXC_A - excB
    print('XC_Energy nadd                            '+str(nadd_eXC), flush=True)

    #Construct the Embedding potential
    embPot = EmbdmatPot + nAdd_KE_Pot + XC_AandB_basA - XC_A2
    
    if isDF:
        if isNucAdensB:
            E_int = nadd_eXC + nadd_eKE + e_coul_densA_nucB_densB + e_nucA_nucB + e_nucA_densB
        else:
            E_int = nadd_eXC + nadd_eKE + e_coul_densA_nucB_densB + e_nucA_nucB + nucAux(molA, molB, molTotal, dmatB)[0]
    else:
        E_int = nadd_eXC + nadd_eKE + e_coul_densA_nucB_densB + e_nucA_nucB + e_nucA_densB
    print('Embedding energy from DFT embeddding potential and DFT cluster density', E_int, flush=True)

    pot_matrices = {'nadd_KE_pot':nAdd_KE_Pot, 'nadd_XC_pot':XC_AandB_basA - XC_A2, 'coul_A':J_AA}
    energies = {'ke_cluster':ke_cluster, 'e_nn_cluster':e_nn_cluster,'e_nucA_nucB':e_nucA_nucB,'nadd_eXC':nadd_eXC,
                'nadd_eKE':nadd_eKE,'e_nucA_densB':e_nucA_densB,'e_coul_densA_nucB_densB':e_coul_densA_nucB_densB,
                'exc_cluster':eXC_A,'e_coul_nucA_densA':e_coul_nucA_densA, 'e_coul_densA_densA':e_coul_densA_densA} 
    mo_info = {'mo_occ':mo_occ, 'mo_energy':mo_energy, 'mo_coeff':mo_coeff}
    return e_tot_Final, E_int, dmatNew, pot_matrices, energies, mo_info



if os.path.exists('viz.html'):
    os.remove('viz.html')

# Set page config
st.set_page_config(page_title='Frozen Density Embedding', layout='wide', page_icon="🧊",
menu_items={
         'About': "This is an online demo of frozen density embedding. It lets you perform your own embedding calculations for small systems with less than 50 basis functions."
     })

# Sidebar stuff
st.sidebar.write('# About')
st.sidebar.write('### Made By [Manas Sharma](https://manas.bragitoff.com)')
st.sidebar.write('### *Powered by*')
st.sidebar.write('* [PySCF](https://pyscf.org/) for Molecular Integrals and DFT Calculations')
st.sidebar.write('* [Py3Dmol](https://3dmol.csb.pitt.edu/) for Chemical System Visualizations')
st.sidebar.write('* [Streamlit](https://streamlit.io/) for making of the Web App')
st.sidebar.write('## Brought to you by [CrysX](https://www.bragitoff.com/crysx/)')
# st.sidebar.write('## Cite us:')
# st.sidebar.write('[Sharma, M. & Mishra, D. (2019). J. Appl. Cryst. 52, 1449-1454.](http://scripts.iucr.org/cgi-bin/paper?S1600576719013682)')


# Main app
st.write('# CrysX-DEMO: Frozen Density Embedding (FDE)')
st.write('This is an online demo of frozen density embedding (FDE) using Gaussian basis functions. You can perform FDE calculations on the already available small test systems or use your own. NOTE: Calculations can only be performed for systems with less than 50 basis functions due to limited compute resources on the server where the web app is freely hosted.')
st.write('FDE utilizes an embedding potential of the following form')
st.latex(r'v_{\mathrm{emb}}\left[\rho^{\mathrm{A}}, \rho^{\mathrm{B}}, v_{\mathrm{nuc}}^{\mathrm{B}}\right](\boldsymbol{r})=v_{\mathrm{nuc}}^{\mathrm{B}}(\boldsymbol{r}) + \int \frac{\rho^{\mathrm{B}}\left(\boldsymbol{r}^{\prime}\right)}{\left|\boldsymbol{r}-\boldsymbol{r}^{\prime}\right|} d \boldsymbol{r}^{\prime} + \frac{\delta E_{\mathrm{xc}}^{\mathrm{nadd}}\left[\rho^{\mathrm{A}}, \rho^{\mathrm{B}}\right]}{\delta \rho^{\mathrm{A}}(\boldsymbol{r})} + \frac{\delta T_{\mathrm{s}}^{\mathrm{nadd}}\left[\rho^{\mathrm{A}}, \rho^{\mathrm{B}}\right]}{\delta \rho^{\mathrm{A}}(\boldsymbol{r})}')
# DATA for test systems
hf_dimer_xyz = '''
4

F          1.32374       -0.09023       -0.00001
H          1.74044        0.73339        0.00001
F         -1.45720        0.01926       -0.00001
H         -0.53931       -0.09466        0.00015
'''
h2o_dimer_xyz = '''
6

O          1.53175        0.00592       -0.12088
H          0.57597       -0.00525        0.02497
H          1.90625       -0.03756        0.76322
O         -1.39623       -0.00499        0.10677
H         -1.78937       -0.74228       -0.37101
H         -1.77704        0.77764       -0.30426
'''
nh3_dimer_xyz = '''
8

N          1.57523        0.00008       -0.04261
H          2.13111        0.81395       -0.28661
H          1.49645       -0.00294        0.97026
H          2.13172       -0.81189       -0.29145
N         -1.68825        0.00008        0.10485
H         -2.12640       -0.81268       -0.31731
H         -2.12744        0.81184       -0.31816
H         -0.71430        0.00054       -0.19241
'''
ch4_dimer_xyz = ''''''
benzene_fulvene_dimer_xyz = '''
24

C         -0.65914       -1.21034        3.98683
C          0.73798       -1.21034        4.02059
C         -1.35771       -0.00006        3.96990
C          1.43653       -0.00004        4.03741
C         -0.65915        1.21024        3.98685
C          0.73797        1.21024        4.02061
H         -1.20447       -2.15520        3.97369
H          1.28332       -2.15517        4.03382
H         -2.44839       -0.00006        3.94342
H          2.52722       -0.00004        4.06369
H         -1.20448        2.15509        3.97373
H          1.28330        2.15508        4.03386
C          0.64550       -0.00003        0.03038
C         -0.23458       -1.17916       -0.00274
C         -0.23446        1.17919       -0.00272
C         -1.51856       -0.73620       -0.05059
C         -1.51848        0.73637       -0.05058
C          1.99323       -0.00010        0.08182
H          0.11302       -2.20914        0.01010
H          0.11325        2.20913        0.01013
H         -2.41412       -1.35392       -0.08389
H         -2.41398        1.35418       -0.08387
H          2.56084        0.93137        0.10366
H          2.56074       -0.93163        0.10364
'''
ethane_xyz = '''
8

C          0.00000        0.00000        0.76510
H          0.00000       -1.02220        1.16660
H         -0.88530        0.51110        1.16660
H          0.88530        0.51110        1.16660
C          0.00000        0.00000       -0.76510
H          0.88530       -0.51110       -1.16660
H          0.00000        1.02220       -1.16660
H         -0.88530       -0.51110       -1.16660
'''

dict_name_to_xyz = {'HF dimer': hf_dimer_xyz,'H2O dimer': h2o_dimer_xyz,'NH3 dimer': nh3_dimer_xyz,'Benzene-Fulvene': benzene_fulvene_dimer_xyz,'Ethane': ethane_xyz}

input_test_system = st.selectbox('Select a test system',
     ( 'HF dimer', 'H2O dimer', 'NH3 dimer', 'Benzene-Fulvene', 'Ethane'))

selected_xyz_str = dict_name_to_xyz[input_test_system]

st.write('#### Alternatively you can provide the XYZ file contents of your own structure here')

input_geom_str = st.text_area(label='XYZ file of the given/selected system', value = selected_xyz_str, placeholder = 'Put your text here', height=250)
# Get rid of trailing empty lines
input_geom_str = input_geom_str.rstrip()
# Get rid of leading empty lines
input_geom_str = input_geom_str.lstrip()

basis_set_tot = st.selectbox('Select a basis set',
     ( 'STO-3G', 'def2-SVP', 'def2-TZVP', '6-31G', '6-311G'))

# Create mol object for PySCF
molTot = gto.M()
molTot.unit='A'
molTot.atom = input_geom_str.split("\n",2)[2]
molTot.basis = basis_set_tot
molTot.build()

show_parsed_coords = st.checkbox('Debug: Show parsed coordinates and other information about the system')
if show_parsed_coords:
    # Create a dataframe for atomic coordinates
    # Create an array of Atomic symbols
    molTot_atoms = []
    for i in range(molTot.natm):
        molTot_atoms.append(molTot.atom_pure_symbol(i))
    # Create a dataframe object of atomic coordinates
    pd_mol_tot = pd.DataFrame(data=molTot.atom_coords(unit='ANG'), columns=['x','y','z'])  # 1st row as the column names
    # Insert the atomic symbols array as a column into the dataframe object
    # pd_mol_tot['Atom'] = molTot_atoms
    pd_mol_tot.insert(loc=0, column='Atom', value = molTot_atoms)
    # Increment the index by 1 so that the atomic indices start from 1
    pd_mol_tot.index += 1 
    # print the dataframe object
    st.write(pd_mol_tot)
    # st.write(molTot._atom)
    # for i in range(molTot.natm):
    #     st.write('%s %s  charge %f  xyz %s' % (molTot.atom_symbol(i),
    #                                     molTot.atom_pure_symbol(i),
    #                                     molTot.atom_charge(i),
    #                                     molTot.atom_coord(i)))
    st.write('*Charge*: '+str(molTot.charge))
    st.write('*Spin*: '+str(molTot.spin))
    st.write('*No. of basis functions*: '+str(molTot.nao))

# Error checks:
if molTot.charge!=0:
    st.error('The charge on the total system is '+str(molTot.charge)+'which is not currently supported. Please use a neutral system.', icon="🚨")
    st.stop()

st.write('#### Visualization')
### VISUALIZATION ####
style = st.selectbox('Visualization style',['ball-stick','line','cross','stick','sphere'])
col1, col2 = st.columns(2)
spin = col1.checkbox('Spin', value = False)
showLabels = col2.checkbox('Show Labels', value = True)
# style='stick'
# style='cartoon'
# style='sphere'
view = py3Dmol.view(width=500, height=300)
view.addModel(input_geom_str, 'xyz')
if style=='ball-stick': # my own custom style
    view.setStyle({'sphere':{'colorscheme':'Jmol','scale':0.3},
                       'stick':{'colorscheme':'Jmol', 'radius':0.}})
else:
    view.setStyle({style:{'colorscheme':'Jmol'}})
# Label addition template
if showLabels:
    atmidx=1
    for atom in molTot.atom_coords(unit='ANG'):
        view.addLabel(str(atmidx), {'position': {'x':atom[0], 'y':atom[1], 'z':atom[2]}, 
            'backgroundColor': 'white', 'backgroundOpacity': 0.5,'fontSize':18,'fontColor':'black',
                'fontOpacity':1,'borderThickness':0.0,'inFront':'true','showBackground':'false'})
        atmidx = atmidx + 1
view.zoomTo()
view.spin(spin)
view.show()
view.render()
# view.png()
t = view.js()
f = open('viz.html', 'w')
f.write(t.startjs)
f.write(t.endjs)
f.close()

HtmlFile = open("viz.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
components.html(source_code, height = 300, width=900)
HtmlFile.close()


dict_name_to_partition_indx = {'HF dimer': 2,'H2O dimer': 3,'NH3 dimer': 4,'Benzene-Fulvene': 6,'Ethane': 4}
st.write('#### Select the atoms that should be used as the subsystem A (active subsystem)')
partition_indx = st.slider('The first n selected atoms would be used as the subsystem A and the remaining as the subsystem B (environment)', 0, molTot.natm, molTot.natm//2)
isSupermolecularBasis =  st.checkbox('Use a supermolecular basis for the subsystems')

# Set charge for the subsystems
col1, col2 = st.columns(2)
chargeA = col1.number_input('Charge for Subsystem A', step=1)
chargeB = col2.number_input('Charge for Subsystem B', step=1)


# Create mol objects for the subsystems
molA = gto.M()
molA.unit='A'
temp = input_geom_str.split("\n",2)[2]
if isSupermolecularBasis:
    temp_list = temp.split('\n')
    for i in range(partition_indx, len(temp_list)):
        temp_list[i] = 'ghost-'+temp_list[i]
    # st.write(temp_list)
    molA.atom = temp_list
else:
    molA.atom = temp.split('\n')[0:partition_indx]
molA.basis = basis_set_tot
molA.charge = chargeA
molA.build()

# Create mol objects for the subsystems
molB = gto.M()
molB.unit='A'
temp = input_geom_str.split("\n",2)[2]
if isSupermolecularBasis:
    temp_list = temp.split('\n')
    for i in range(0, partition_indx):
        temp_list[i] = 'ghost-'+temp_list[i]
    # st.write(temp_list)
    molB.atom = temp_list
else:
    molB.atom = temp.split('\n')[partition_indx:]
molB.basis = basis_set_tot
molB.charge = chargeB
molB.build()

show_subsystem_information = st.checkbox('Debug: Show subsystem information')
if show_subsystem_information:
    col1, col2 = st.columns(2)
    col1.write('##### Subsystem A')
    col2.write('##### Subsystem B')
    # Create a dataframe for atomic coordinates
    # Create an array of Atomic symbols
    molA_atoms = []
    for i in range(molA.natm):
        molA_atoms.append(molA.atom_pure_symbol(i))
    # Create a dataframe object of atomic coordinates
    pd_mol_A = pd.DataFrame(data=molA.atom_coords(unit='ANG'), columns=['x','y','z'])  # 1st row as the column names
    # Insert the atomic symbols array as a column into the dataframe object
    # pd_mol_tot['Atom'] = molTot_atoms
    pd_mol_A.insert(loc=0, column='Atom', value = molA_atoms)
    # Increment the index by 1 so that the atomic indices start from 1
    pd_mol_A.index += 1 
    # print the dataframe object
    col1.write(pd_mol_A)
    col1.write('*Charge*: '+str(molA.charge))
    col1.write('*Spin*: '+str(molA.spin))
    col1.write('*No. of basis functions*: '+str(molA.nao))

    # Create a dataframe for atomic coordinates
    # Create an array of Atomic symbols
    molB_atoms = []
    for i in range(molB.natm):
        molB_atoms.append(molB.atom_pure_symbol(i))
    # Create a dataframe object of atomic coordinates
    pd_mol_B = pd.DataFrame(data=molB.atom_coords(unit='ANG'), columns=['x','y','z'])  # 1st row as the column names
    # Insert the atomic symbols array as a column into the dataframe object
    # pd_mol_tot['Atom'] = molTot_atoms
    pd_mol_B.insert(loc=0, column='Atom', value = molB_atoms)
    # Increment the index by 1 so that the atomic indices start from 1
    pd_mol_B.index += 1 
    # print the dataframe object
    col2.write(pd_mol_B)
    col2.write('*Charge*: '+str(molB.charge))
    col2.write('*Spin*: '+str(molB.spin))
    col2.write('*No. of basis functions*: '+str(molB.nao))
    massA = molA.atom_mass_list()
    coordsA = molA.atom_coords()
    comA = np.einsum('i,ij->j', massA, coordsA)/massA.sum()
    massB = molB.atom_mass_list()
    coordsB = molB.atom_coords()
    comB = np.einsum('i,ij->j', massB, coordsB)/massB.sum()
    bohr2angs = 0.529177249 
    st.write('*Distance between the two subsystems*:  '+str(np.linalg.norm(comA-comB)*bohr2angs)+'  Angstroms')

# st.snow()

st.write('#### Set other calculation parameters')
xc = st.selectbox('Select an exchange-correlation functional',
     ( 'PBE', 'BLYP', 'BP86', 'PW91', 'SVWN','REVPBE'))
kedf_select = st.selectbox('Select a kinetic energy density functional',
     ( 'GGA_K_LC94', 'LDA_K_TF', 'GGA_K_APBE', 'GGA_K_REVAPBE','GGA_K_TW2', 'electro'))
dict_kedf = {'GGA_K_LC94':'521','LDA_K_TF':'50', 'electro':'electro', 'GGA_K_APBE':'185', 'GGA_K_REVAPBE':'55', 'GGA_K_TW2':'188'}
kedf = dict_kedf[kedf_select]
isDF =  st.checkbox('Use density fitting (Resolution of Identity)')
exponent = st.slider('Convergence criteria', 5, 8, 7)
conv_crit = 10**(-exponent)

# show_scf_summary =  st.checkbox('Show SCF Summary', key='scf_summary')

col1, col2, col3 = st.columns([1,1,1])
if col2.button('Run FDE calculation'):
    if molTot.nao>=50:
        st.error('The no. of basis functions of the total system is '+str(molTot.nao)+' which is too much for a free online tool. Please use a smaller basis set or choose a smaller system.', icon="🚨")
        st.stop()

    with st.spinner('Running a DFT calculation on the total system for reference...'):
        # st.write('#### DFT on Total System for Reference')
        if isDF:
            mfTot = dft.RKS(molTot).density_fit(auxbasis='weigend')
        else:
            mfTot = dft.RKS(molTot)
        mfTot.xc = xc
        mfTot.conv_tol = conv_crit
        energyTot = mfTot.kernel()
        dmTot = mfTot.make_rdm1(mfTot.mo_coeff, mfTot.mo_occ)
        if mfTot.converged:
            st.success('##### Reference DFT energy of the total system =   **'+ str(energyTot)+'**'+'  a.u.', icon = '✅')
            with st.expander("See SCF Summary"):
                dump_scf_summary(mfTot)
                mfTot.dump_scf_summary()
                # mfTot.analyze(verbose=5, ncol=10, digits=9)
            with st.expander('See the orbital info and density matrix of total system'):
                st.write('#### MO Energy Info')
                tmp_list = []
                for i,c in enumerate(mfTot.mo_occ):
                    # st.write('**MO** #'+str(i+1)+ '    **energy** = '+ str(mfTot.mo_energy[i])+ '     **occ** = '+str(c))
                    tmp_list.append([str(i+1), mfTot.mo_energy[i], c])
                df_mo_Tot = pd.DataFrame(tmp_list, columns=['MO #','Energy (Ha)','Occupation'])
                st.write(df_mo_Tot)
                st.write('#### Density matrix of total system')
                st.write(dmTot)
        else:
            st.error('DFT calculation for the total system did not converge.', icon = '🚨')
            st.stop()
    
    with st.spinner('Running a DFT calculation on the environment (subsystem B)...'):
        st.info('#### Step 1: Calculate the density of isolated subsystem B')
        if isDF:
            mfB = dft.RKS(molB).density_fit(auxbasis='weigend')
        else:
            mfB = dft.RKS(molB)
        mfB.xc = xc
        mfB.conv_tol = conv_crit
        energyB = mfB.kernel()
        dmB = mfB.make_rdm1(mfB.mo_coeff, mfB.mo_occ)
        if mfB.converged:
            st.success('##### DFT energy of the isolated environment (subsystem B) =   **'+ str(energyB)+'**'+'  a.u.', icon = '✅')
            with st.expander("See SCF Summary"):
                dump_scf_summary(mfB)
                mfB.dump_scf_summary()
            with st.expander('See the orbital info and density matrix of subsystem B'):
                st.write('#### MO Energy Info')
                tmp_list = []
                for i,c in enumerate(mfB.mo_occ):
                    # st.write('**MO** #'+str(i+1)+ '    **energy** = '+ str(mfB.mo_energy[i])+ '     **occ** = '+str(c))
                    tmp_list.append([str(i+1), mfB.mo_energy[i], c])
                df_mo_B = pd.DataFrame(tmp_list, columns=['MO #','Energy (Ha)','Occupation'])
                st.write(df_mo_B)
                st.write('#### Density matrix of Subsystem B')
                st.write(dmB)
                
        else:
            st.error('DFT calculation for the environment (subsystem B) did not converge.', icon = '🚨')
            st.stop()
    
    with st.spinner('##### Running an FDE calculation on the active subsystem (subsystem A) using the frozen density of isolated B...'):
        st.info('#### Step 2: Solving the Kohn Sham Constrained Electron Density (KSCED) equations')
        st.latex(r'\left[-\frac{\nabla^{2}}{2}+v_{\mathrm{eff}}^{\mathrm{KS}}\left[\rho^{\mathrm{A}}\right](\boldsymbol{r})+v_{\mathrm{emb}}\left[\rho^{\mathrm{A}}, \rho^{\mathrm{B}}\right](\boldsymbol{r})\right] \phi_{i}^{(\mathrm{A})}(\boldsymbol{r})=\epsilon_{i} \phi_{i}^{(\mathrm{A})}(\boldsymbol{r}) \quad ; \quad i=1, \ldots, N_{\mathrm{A}} / 2')
        excB = mfB.scf_summary['exc']
        
        if isDF:
            mfA = dft.RKS(molA).density_fit(auxbasis='weigend')
        else:
            mfA = dft.RKS(molA)
        mfA.xc = xc
        #calculate the Coulomb matrix of A due to B
        if (isDF):
            Jab = coulombMatrixEmbDF(molTot, molA, molB, dmB)
        else:
            if isSupermolecularBasis:
                Jab = coulombMatrixEmbSuper(molTot, mfTot, dmB)
            else:    
                Jab = coulombMatrixEmb(molTot, molA, molB, dmB)
        #Calcuate nuclear matrix due to the whole system in the basis of A
        Vnuctot = molTot.intor('int1e_nuc')[0:molA.nao_nr(), 0:molA.nao_nr()]
        #Nuclear matrix of A due to B
        Vab = Vnuctot - molA.intor('int1e_nuc') 
        
        energyA_FDE, E_intAB, dmA_fde, pot_matrices, energies, mo_info  = scf1(molB, molA, mfA, mfB, molTot, mfTot, dmB, excB, Jab, Vab, 40, 2000)
        if isFDEconverged:
            st.success('##### FDE energy of the embedded active subsystem (subsystem A) =   **'+ str(energyA_FDE)+'**'+'  a.u.', icon = '✅')
            st.write('The above energy also includes the interaction energy (E_int) =  '+str(E_intAB)+'  a.u.')
            st.write('Energy of subsystem A (E_A) without the interaction energy =  '+str(energyA_FDE-E_intAB)+'  a.u.')
            with st.expander("See SCF Summary"):
                # st.text('**** SCF Summary ****')
                st.write('##### Energies of subsystem A')
                st.write('Kinetic energy of A =  '+str(energies['ke_cluster'])+'  a.u.')
                st.write('Electron_A-Electron_A energy =  '+str(energies['e_coul_densA_densA'])+'  a.u.')
                st.write('Nuclear_A-Electron_A energy =  '+str(energies['e_coul_nucA_densA'])+'  a.u.')
                st.write('Exchange correlation energy of A =  '+str(energies['exc_cluster'])+'  a.u.')
                st.write('Nuclear_A-Nuclear_A energy =  '+str(energies['e_nn_cluster'])+'  a.u.')
                st.write('##### Energies due to interaction with subsystem B')
                st.write('Electron_A-Nuclear_B + Electron_A-Electron_B energy =  '+str(energies['e_coul_densA_nucB_densB'])+'  a.u.')
                st.write('Non-additive kinetic energy =  '+str(energies['nadd_eKE'])+'  a.u.')
                st.write('Non-additive exchange-correlation energy =  '+str(energies['nadd_eXC'])+'  a.u.')
                st.write('NuclearA-NuclearB energy =  '+str(energies['e_nucA_nucB'])+'  a.u.')
                st.write('NuclearA-ElectronB energy = '+str(energies['e_nucA_densB'])+'  a.u.')
            with st.expander('See the potential matrices'):
                st.write('#### Embedding potential matrix')
                st.latex(r'V^\mathrm{emb}_{\mu \nu} = \left<\mu \right | v_{\mathrm{emb}} \left | \nu \right>')
                st.write(embPot)
                st.write('#### Nuclear potential matrix due to the nuclei of subsystem B in the basis of A')
                st.latex(r'V^{\mathrm{nuc};\mathrm{B}}_{\mu \nu} = \left<\mu \right | v_{\mathrm{nuc}}^\mathrm{B} \left | \nu \right>')
                st.write(Vab)
                st.write('#### Coulomb potential matrix due to the electrons of subsystem B in the basis of A')
                st.latex(r'J^\mathrm{B}_{\mu \nu} = \left<\mu \right | \int \frac{\rho^{\mathrm{B}}\left(\boldsymbol{r}^{\prime}\right)}{\left|\boldsymbol{r}-\boldsymbol{r}^{\prime}\right|} d \boldsymbol{r}^{\prime} \left | \nu \right>')
                st.write(Jab)
                st.write('#### Non-additive exchange-correlation potential matrix')
                st.latex(r'X^\mathrm{nadd}_{\mu \nu} = \left<\mu \right | \frac{\delta E_{\mathrm{xc}}^{\mathrm{nadd}}\left[\rho^{\mathrm{A}}, \rho^{\mathrm{B}}\right]}{\delta \rho^{\mathrm{A}}(\boldsymbol{r})} \left | \nu \right>')
                st.write(pot_matrices['nadd_XC_pot'])
                st.write('#### Non-additive kinetic potential matrix')
                st.latex(r'T^\mathrm{nadd}_{\mu \nu} = \left<\mu \right | \frac{\delta T_{\mathrm{s}}^{\mathrm{nadd}}\left[\rho^{\mathrm{A}}, \rho^{\mathrm{B}}\right]}{\delta \rho^{\mathrm{A}}(\boldsymbol{r})} \left | \nu \right>')
                st.write(pot_matrices['nadd_KE_pot'])
            with st.expander('See the orbital info and density matrix'):
                st.write('#### MO Energy Info')
                tmp_list = []
                for i,c in enumerate(mo_info['mo_occ']):
                    tmp_list.append([str(i+1), mo_info['mo_energy'][i], c])
                df_mo_A = pd.DataFrame(tmp_list, columns=['MO #','Energy (Ha)','Occupation'])
                st.write(df_mo_A)
                st.write('#### Density matrix of embedded Subsystem A')
                st.write(dmA_fde)

        else:
            st.error('FDE calculation for the active subsystem did not converge.', icon = '🚨')
            st.stop()
    st.write('The energy of the total system from FDE')
    st.latex('E_\mathrm{Tot} = E_\mathrm{A}+E_\mathrm{B}+E_\mathrm{int}')
    energyTot_FDE = energyA_FDE + energyB
    st.success('##### Energy of the total system from FDE =   **'+ str(energyTot_FDE)+'**'+'  a.u.', icon = '✅')
    
    
    
    st.write('Error with respect to a regular KS-DFT calculation on the total system')
    st.latex(r'\Delta E = E^\mathrm{tot}_\mathrm{DFT} - E^\mathrm{tot}_\mathrm{FDE}')
    st.info('##### *Error (E_DFT - E_FDE)* = '+str(energyTot-energyTot_FDE)+'  a.u.')

    # if isSupermolecularBasis:
    #     if isDF:
    #         mfTot = dft.RKS(molTot).density_fit(auxbasis='weigend')
    #     else:
    #         mfTot = dft.RKS(molTot)
    #     mfTot.verbose = 4
    #     mfTot.xc = xc
    #     mfTot.conv_tol = conv_crit
    #     # energyTot = mfTot.kernel()
    #     mfTot.max_cycle=0
    #     energyTot_FDE2=mfTot.kernel(dmA_fde+dmB)
    #     st.write('check:',energyTot_FDE2)