# CrysX - DEMO of DFT-based Embedding
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crysx-dfet-trial.streamlit.app/)

This GitHub repo contains all the source code of a streamlit web application demonstrating the DFT-based embedding methods. 

The [streamlit web app](https://crysx-dfet-trial.streamlit.app/) provides an online interactive demo of various density functional theory (DFT) based embedding techniques like Frozen Density Embedding and Projection-based Embedding.

![alt text](https://github.com/manassharma07/CrysX_Demo_DFT_based_Embedding/blob/main/Thumbnail_FDE_PbE_CrysX?raw=true)

The app lets you perform your own embedding calculations for small systems with less than 50 basis functions.

The purpose of this interactive web application is to complement my thesis, posters, and manuscripts, to give the readers a way to run their own calculations. I hope this would serve as a reasonably fast method of making you (the readers) familiar with DFT based embedding.

The online web app contains the following features:

* [Frozen Density Embedding (FDE)](https://crysx-dfet-trial.streamlit.app/FDE) calculation with and without [Freeze-and-Thaw (FaT)](https://crysx-dfet-trial.streamlit.app/FDE_+_FaT)
* [Projection-based Embedding (PbE)](https://crysx-dfet-trial.streamlit.app/PbE) calculation with and without [Freeze-and-Thaw (FaT)](https://crysx-dfet-trial.streamlit.app/PbE_+_FaT)

Furthermore, users have the ability to
1. Choose from various basis sets, exchange-correlation, and kinetic energy density functionals.
2. Choose from a template system or provide your own atomic coordinates.
3. Use a supermolecular or monomolecular basis.
4. Use density fitting (three-centered two-electron integrals)  or the traditional four-centered two-electron integrals.
5. Visualize the system.
6. See the potential matrices, density matrices, molecular orbital energies, and a lot more information.

Therefore, although it is a demo app, it can be used as a benchmark for testing your own codes and implementations of DFT based embedding techniques.

## Limitations
The same basis set is used for the entire system. You cannot choose a different basis set for the subsystems or a particular set of atoms.
The same exchange-correlation functional is used for the subsystems as well as the non-additive exchange-correlation potential term.

## Web App
The web app is available at [https://crysx-dfet-trial.streamlit.app/](https://crysx-dfet-trial.streamlit.app/).

Hint: Use the sidebar menu on the left to navigate between FDE, FDE+FaT, PbE, PBE+FaT. The menu may be hidden on mobile devices so you will need to expand it first by clicking on the arrow icon in the upper-left corner.
