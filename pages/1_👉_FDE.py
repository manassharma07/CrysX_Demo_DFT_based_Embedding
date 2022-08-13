import streamlit as st 
from pyscf import gto, dft 
import numpy as np
import py3Dmol
import streamlit.components.v1 as components
import os
import pandas as pd

def nucDensB(molA, molB, molTotal, dmatB):
    if isSupermolecular:
        Vnuctot_Mat = molTotal.intor('int1e_nuc')
    else:
        Vnuctot_Mat = molTotal.intor('int1e_nuc')[molA.nao_nr():, molA.nao_nr():]
    VnucB_mat = molB.intor('int1e_nuc')
    V_eff = Vnuctot_Mat-VnucB_mat
    energy = np.einsum('pq,qp',dmatB,V_eff)
    return energy

def get_hcore(molA, molB, dmB):
    '''nuc_A + nuc_B + t_A + vj_B + Pb
    '''
    h = molA.intor_symmetric('int1e_kin')

    h+= molA.intor_symmetric('int1e_nuc')
    h+= molA.intor_symmetric('int1e_nuc')

    return h

if os.path.exists('viz.html'):
    os.remove('viz.html')

# Set page config
st.set_page_config(page_title='Frozen Density Embedding', layout='wide', page_icon="🧊",
menu_items={
         'About': "# This is an online demo of various DFT based embedding techniques. It lets you perform your own embedding calculations for small systems with less than 50 basis functions."
     })

# Sidebar stuff
st.sidebar.write('# About')
st.sidebar.write('### Made By [Manas Sharma](https://www.bragitoff.com/about/)')
st.sidebar.write('### *Powered by*')
st.sidebar.write('* [PySCF](https://pyscf.org/) for Quantum Chemistry Calculations')
st.sidebar.write('* [Py3Dmol](https://3dmol.csb.pitt.edu/) for Chemical System Visualizations')
st.sidebar.write('* [Streamlit](https://streamlit.io/) for making of the Web App')
st.sidebar.write('## Brought to you by [CrysX](https://www.bragitoff.com/crysx/)')
# st.sidebar.write('## Cite us:')
# st.sidebar.write('[Sharma, M. & Mishra, D. (2019). J. Appl. Cryst. 52, 1449-1454.](http://scripts.iucr.org/cgi-bin/paper?S1600576719013682)')


# Main app
st.write('# CrysX-DEMO: Frozen Density Embedding (FDE)')
st.write('This is an online demo of frozen density embedding (FDE). You can perform FDE calculations on the already available small test systems or use your own. NOTE: Calculations can only be performed for systems with less than 50 basis functions due to limited compute resources on the server where the web app is freely hosted.')
st.write('FDE utilizes an embedding potential of the following form')
st.latex(r'v_{\mathrm{emb}}\left[\rho^{\mathrm{act}}, \rho^{\mathrm{env}}, v_{\mathrm{nuc}}^{\mathrm{env}}\right](\boldsymbol{r})=v_{\mathrm{nuc}}^{\mathrm{env}}(\boldsymbol{r})+\int \frac{\rho^{\mathrm{env}}\left(\boldsymbol{r}^{\prime}\right)}{\left|\boldsymbol{r}-\boldsymbol{r}^{\prime}\right|} d \boldsymbol{r}^{\prime}+\frac{\delta E_{\mathrm{xc}}^{\mathrm{nadd}}\left[\rho^{\mathrm{act}}, \rho^{\mathrm{env}}\right]}{\delta \rho^{\mathrm{act}}(\boldsymbol{r})}+v_{T}\left[\rho^{\mathrm{act}}, \rho^{\mathrm{env}}\right](\boldsymbol{r})')
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

st.write('#### Alternatively you can provide the XYZ file of your own structure here.')

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
    st.write('##### Charge: '+str(molTot.charge))
    st.write('##### Spin: '+str(molTot.spin))
    st.write('##### No. of basis functions: '+str(molTot.nao))

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


# Set charge for the subsystems
col1, col2 = st.columns(2)
chargeA = col1.number_input('Charge for Subsystem A', step=1)
chargeB = col2.number_input('Charge for Subsystem B', step=1)


# Create mol objects for the subsystems
molA = gto.M()
molA.unit='A'
temp = input_geom_str.split("\n",2)[2]
molA.atom = temp.split('\n')[0:partition_indx]
molA.basis = basis_set_tot
molA.charge = chargeA
molA.build()

# Create mol objects for the subsystems
molB = gto.M()
molB.unit='A'
temp = input_geom_str.split("\n",2)[2]
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

# st.snow()

st.write('#### Set other calculation parameters')
xc = st.selectbox('Select an exchange-correlation functional',
     ( 'PBE', 'BLYP', 'BP86', 'PW91', 'SVWN','REVPBE'))
kedf = st.selectbox('Select a kinetic energy density functional',
     ( 'GGA_K_LC94', 'LDA_K_TF', 'GGA_K_APBE', 'GGA_K_REVAPBE','GGA_K_TW2'))
isDF =  st.checkbox('Use density fitting (Resolution of Identity)')
isSupermolecularBasis =  st.checkbox('Use a supermolecular basis for the subsystems')

col1, col2, col3 = st.columns([1,1,1])
if col2.button('Run FDE calculation'):
    if molTot.nao>=50:
        st.error('The no. of basis functions of the total system is '+str(molTot.nao)+' which is too much for a free online tool. Please use a smaller basis set or choose a smaller system.', icon="🚨")
        st.stop()
    
    with st.spinner('Running a DFT calculation on the environment (subsystem B)...'):
        mfB = dft.RKS(molB)
        mfB.xc = xc
        energyB = mfB.kernel()
        if mfB.converged:
            st.success('DFT energy of the environment (subsystem B) =   '+ str(energyB), icon = '✅')
        else:
            st.error('Calculation did not converge.', icon = '🚨')

    with st.spinner('Running a DFT calculation on the total system...'):
        mf = dft.RKS(molTot)
        mf.xc = xc
        energyTot = mf.kernel()
        st.success('Reference DFT energy of the total system =   '+ str(energyTot), icon = '✅')

