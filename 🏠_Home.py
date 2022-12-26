import streamlit as st
import platform

# Set page config
st.set_page_config(page_title='CrysX-DEMO: DFT based Embedding', layout='wide', page_icon="🧊",
menu_items={
         'About': "# This is an online demo of various DFT based embedding techniques. It lets you perform your own embedding calculations for small systems with less than 50 basis functions."
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
st.write('# CrysX-DEMO: DFT based Embedding')
st.write('#### This is an online interactive demo of various density functional theory (DFT) based embedding techniques.')
st.write(' It lets you perform your own embedding calculations for small systems with less than 50 basis functions.')


intro_text = '''
The purpose of this interactive web application is to compliment my thesis, posters and manuscripts, to give the readers a way to run their own calculations. I hope that this would serve as a reasonably fast method of making you (the readers) familiar with DFT based embedding.

Embedding techniques are usually needed to perform calculations on large systems where treating the entire system using a very expensive higher level of theory is not feasible. Therefore, the total system is divided into two or more subsystems, and the subsystem corresponding to the region of interest is treated at a higher level of theory like wavefunction theory (WFT) methods while the remaining subsystems can be treated using a lower level of theory like LDA-DFT. 

'''

st.write(intro_text)
st.image('embedding_pic.png')
st.write('DFT provides a natural way to partition the system based on the electronic density. Then the influence of the environment subsystem onto the active subsystem can be considered using an embedding potential which is a functional of both the active and environment subsystem densities.')
st.image('embedding_pot_pic.png')
st.write('DFT based embedding refers to the fact that the embedding potential is constructed at the DFT level, using which one may perform DFT-in-DFT or WFT-in-DFT calculations for either ground-state or excited state properties.')



# Check if the app is being run on streamlit cloud
#https://discuss.streamlit.io/t/check-if-run-local-or-at-share-streamlit-io/11841/3
isDemo = True
if platform.processor():
    isDemo = False
    
if isDemo:
    st.info('This web app is being run on streamlit cloud with limited resources. Therefore, you can only run calculations on systems with less than 50 basis functions.', icon = '🚨')
