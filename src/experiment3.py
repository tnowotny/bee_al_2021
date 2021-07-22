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

if len(sys.argv) < 2:
    print("usage: python experiment1.py <connect_I: hom/corr0/corr1/cov0/cov1")
    exit()

connect_I= sys.argv[1]

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
#    ("ORs", "ra"),
#    ("ORNs", "V"),
#    ("ORNs", "a"),
]

paras["rec_spikes"]= [
    "ORNs",
]

label= "run"
paras["label"]= label+"_"+connect_I

# In this experiment run without PNs, LNs
paras["n"]["PNs"]= 0
paras["n"]["LNs"]= 0

# Assume a uniform distribution of Hill coefficients inspired by Rospars'
# work on receptors tiling the space of possible sigmoid responses

hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")

# Generate odors 
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
odors= np.array(odors)

# Now, let's make a protocol where each odor is presented for 3 secs with
# 3 second breaks and at each of 25 concentration values
paras["trial_time"]= 12000.0
protocol= []
t_off= 3000.0
base= np.power(10,0.25)

for c1 in [ 0, 1e-3, 1e-1 ]:
    for c2 in [ 0, 1e-6, 1e-5, 1e-4, 1e-3 ]:
        sub_prot= {
            "t": t_off,
            "odor": 0,
            "ochn": "0",
            "concentration": c1
        }
        protocol.append(sub_prot)        
        if c2 != 0:
            sub_prot= {
                "t": t_off,
                "odor": 1,
                "ochn": "1",
                "concentration": c2
            }
            protocol.append(sub_prot)
        sub_prot= {
            "t": t_off+3000.0,
            "odor": 0,
            "ochn": "0",
            "concentration": 0.0
        }
        protocol.append(sub_prot)
        if c2 != 0:
            sub_prot= {
                "t": t_off+3000.0,
                "odor": 1,
                "ochn": "1",
                "concentration": 0.0
            }
            protocol.append(sub_prot)
        t_off+= paras["trial_time"]
        
paras["t_total"]= t_off
print("We are running for a total simulated time of {}ms".format(t_off))

state_bufs, spike_t, spike_ID, ORN_cnts= ALsim(odors, hill_exp, paras, protocol)

avgNo= int(paras["trial_time"]/(paras["spk_rec_steps"]*sim.dt))
d= np.zeros(ORN_cnts.shape[0]//avgNo)
for i in range(ORN_cnts.shape[0]//avgNo):
        d[i]= np.sum(ORN_cnts[i*avgNo:(i+1)*avgNo])

np.save(dirname+"_ONR_cnts",d)
