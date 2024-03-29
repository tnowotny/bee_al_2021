import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from ALsim import ALsim
import sim
import sys
import os
from ALsimParameters import std_paras
import random

"""
Run and experiment to investigate the effect of decreasing response with higher concentration. 
We generate N_odour-1 odours randomly with the following properties: 
1. Each odour has a a Gaussian profile of glomerulus binding (kp1) with sigma drawn from 
   Gauss(mu_sig,sig_sig). 
2. The Gaussian odour profile is over a random permutation of the glomeruli.
3. The overall sensitivity to an odour (amplitude of the Gaussian profile for k1p) is varied by
   10^eta, where eta is a Gaussian random variable 
3. The activation kp2 is homogeneous across glomeruli and is given by zeta, an 
   essentially Gaussian random variable
We then add one odour, "geosmin", which has high sensitivity, broad profile, but low activation kp2
Then, all odours are presented at 25 concentrations for 3s each trial, with 3 second pauses.
The overall strength of inhibition is scaled by a command line argument "ino".
"""

paras= std_paras()
paras["write_to_disk"]= True
if len(sys.argv) < 3:
    print("usage: python experiment1.py <ino> <connect_I: hom/corr0/corr1/cov0/cov1")
    exit()

ino= float(sys.argv[1])
connect_I= sys.argv[2]

if ino == -100:
    paras["lns_pns_g"]= 0
    paras["lns_lns_g"]= 0
else:
    paras["lns_pns_g"]*= np.power(10,ino)
    paras["lns_lns_g"]*= np.power(10,ino)
    
# write results into a dir with current date in the name
timestr = time.strftime("%Y-%m-%d")
paras["dirname"]= timestr+"-runs"

path = os.path.isdir(paras["dirname"])
if not path:
    print("making dir "+paras["dirname"])
    os.makedirs(paras["dirname"])

paras["use_spk_rec"]= True

# Control what to record
paras["rec_state"]= [
]

paras["rec_spikes"]= [
    "PNs",
    ]

label= "run"
paras["label"]= label+"_"+connect_I+"_"+str(ino)

# Assume a uniform distribution of Hill coefficients inspired by Rospars'
# work on receptors tiling the space of possible sigmoid responses
hill_new= True

if hill_new:
    hill_exp= np.random.uniform(0.95, 1.05, paras["n_glo"])
    np.save(paras["dirname"]+"/"+label+"_hill",hill_exp)
else:
    hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")

# Generate odors or load previously generated odors from file
odor_new= True
paras["N_odour"]= 100

if odor_new:
    odors= []
    # create a permutation that will be applied to both IAA and geosmin
    the_shuffle= np.arange(paras["n_glo"])
    random.shuffle(the_shuffle)
    # Add a "IAA" odour 
    od= gauss_odor(paras["n_glo"], paras["n_glo"]//2, paras["IAA_sigma"], paras["IAA_A"], paras["odor_clip"], paras["IAA_act"], 0.0, 1e-10, 1.0)
    od[:,0]= od[the_shuffle,0]
    odors.append(np.copy(od))
    # Add "Geosmin" that is particularly early binding, broad, and low activating
    od= gauss_odor(paras["n_glo"], paras["n_glo"]//2+paras["geo_shift"], paras["geo_sigma"], paras["geo_A"], paras["odor_clip"], paras["geo_act"], 0.0, 1e-10, 1.0)
    od[:,0]= od[the_shuffle,0]
    odors.append(np.copy(od))
    # add any remaining random odours
    for i in range(paras["N_odour"]-2):
        sigma= 0
        while sigma < paras["min_sig"]:
            sigma= np.random.normal(paras["mu_sig"],paras["sig_sig"])
        A= -100.0
        while A < paras["min_A"] or A > paras["max_A"]:
            A= np.random.normal(paras["mean_A"], paras["sig_A"])
        od= gauss_odor(paras["n_glo"], 0, sigma, A, paras["odor_clip"], paras["mean_act"], paras["sig_act"],paras["min_act"],paras["max_act"])
        random.shuffle(od[:,0])
        odors.append(np.copy(od))
    odors= np.array(odors)
    np.save(paras["dirname"]+"/"+label+"_odors",odors)
else:
    odors= np.load(paras["dirname"]+"/"+label+"_odors.npy")
    oNo= odors.shape[0]

# define the inhibitory connectivity pattern in the antennal lobe
correl= choose_inh_connectivity(paras,connect_I)
                
# Now, let's make a protocol where each odor is presented for 3 secs with
# 3 second breaks and at each of 25 concentration values
paras["trial_time"]= 12000.0
protocol= []
t_off= 3000.0
base= np.power(10,0.25)

for i in range(paras["N_odour"]):
    for c in range(25):
        sub_prot= {
            "t": t_off,
            "odor": i,
            "ochn": str(0),
            "concentration": 1e-7*np.power(base,c),
        }
        protocol.append(sub_prot)
        sub_prot= {
            "t": t_off+3000.0,
            "odor": i,
            "ochn": str(0),
            "concentration": 0.0,
        }
        protocol.append(sub_prot)
        t_off+= paras["trial_time"];

paras["t_total"]= t_off
print("We are running for a total simulated time of {}ms".format(t_off))

state_bufs, spike_t, spike_ID, ORN_cnts= ALsim(odors, hill_exp, paras, protocol, lns_gsyn= correl)
