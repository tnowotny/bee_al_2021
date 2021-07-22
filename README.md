# bee_al_2021
This repo contains a model of the bee antennal lobe (AL) intended to investigate odor responses in the antennal lobe. Results of using this model are published in (REFERENCE).
The model uses PyGeNN, the Python interface for the GeNN simulator (https://github.com/genn-team/genn).

To use this software you will need PyGeNN installed (see http://genn-team.github.io/genn/documentation/4/html/d8/d99/Installation.html#pygenn for instructions).
There are 5 main python scripts that allow running different investigations of the model.
experiment1.py:
---
This script shoudl be used first as it allows generating new odors and Hill exponents. Thes are saved into binary python data files which can be reused in subsequent runs and in the other scripts.
   Apart from generating odors and Hill exponents this script runs a simulation where each odor is presented for 3s at 25 different concentrations from 10^{-7} to 10^{-1}. Trials are separated by 9s of simulation without active inputs. The results of teh simulation are saved as spike times and spike IDs. All data is automativally saved into a directory named <date>-runs, where <date> is the current data when the script is invoked.
   Results from running thsi script can be used to generate figures using the Jupyter notebooks "figure2.ipynb" and "figure3.ipynb".
  
experiment2.py
---
  
experiment3.py
---
  
