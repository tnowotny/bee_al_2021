# bee_al_2021
This repo contains a model of the bee antennal lobe (AL) intended to investigate odour responses in the antennal lobe. The results of using this model are published in (UNDER REVIEW).
The model uses PyGeNN, the Python interface for the GeNN simulator (https://github.com/genn-team/genn).

To use this software you will need PyGeNN installed (see http://genn-team.github.io/genn/documentation/4/html/d8/d99/Installation.html#pygenn for instructions).
There are 3 main python scripts that allow running different investigations of the model. The Jupyter notebooks are for analysing the data and making figures, including all figures in the publication. 

experiment1.py:
---
This script should be used first as it allows generating new odours and Hill exponents. These are saved into binary Python data files which can be reused in subsequent runs and in the other scripts.
   Apart from generating odours and Hill exponents, this script runs a simulation where each odour is presented for 3s at 25 different concentrations from 10^{-7} to 10^{-1}. Trials are separated by 9s of simulation without active inputs. The results of the simulation are saved as spike times and spike IDs. All data is automatically saved into a directory named <date>-runs, where <date> is the current date when the script is invoked.
   Results from running this script can be used to generate figures using the Jupyter notebooks "figure2.ipynb" and "figure3.ipynb".
  
experiment2.py
---
In this computational experiment, two odours are presented simultaneously for 3 seconds, followed by 9 seconds of clean air. We scan through 25 concentrations (as above) for both odours and test all combinations. The results are again saved as spike times and spike IDs of the projection neurons. The experiment is run with the strength of inhibition LN->PN, LN->LN and the connectivity scheme defined on the command line. The remaining two command line parameters are the ID of the two odours to use. 

experiment3.py
---
In this computational experiment, we investigate the effect of decreasing response with higher concentration.  
We generate N_odour-1 odours randomly with the following properties: 
1. Each odour has a Gaussian profile of glomerulus binding (kp1) with sigma drawn from Gauss(mu_sig,sig_sig). 
2. The Gaussian odour profile is over a random permutation of the glomeruli.
3. The overall sensitivity to an odour (amplitude of the Gaussian profile for k1p) is varied by 10^eta, where eta is a Gaussian random variable 
3. The activation kp2 is homogeneous across glomeruli and is given by zeta, an 
   essentially Gaussian random variable

We then add one odour, "geosmin", which has high sensitivity, a broad profile, but low activation kp2
Then, all odours are presented at 25 concentrations for 3s each trial, with 3-second pauses.
The overall strength of inhibition is scaled by a command line argument "ino".

Jupyter Notebooks and Figures
---
More detail on the Figures in the published article will be added upon publication.
