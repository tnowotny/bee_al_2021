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
Generate different versions of "geosmin" odors varying sensitivity (A), activation (act) and breadth (sigma) systematically.
Then, all odours are presented at 25 concentrations for 3s each trial, with 9 second pauses.
The overall strength of inhibition is scaled by a command line argument "ino".
"""

paras= std_paras()
paras["write_to_disk"]= True
paras["geoshift"]= 38
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

label= "scan_geo_breadth"
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

odors= []
# Add "Geosmin" odors of different breadth and activation (let's not shuffle
# glomeruli for reproducibility/ definitiveness
abase= 2.0
bbase= np.power(10.0, 0.2)
for A in range(3):
    for a in range(-1,2):
        for b in range(-5,6):
            od= gauss_odor(paras["n_glo"], paras["n_glo"]//2+paras["geo_shift"], np.power(bbase,b)*paras["geo_sigma"], A+paras["geo_A"], paras["odor_clip"], np.power(abase,a)*paras["geo_act"], 0.0, 1e-10, 1.0)
            od[:,0]= od[the_shuffle,0]
            odors.append(np.copy(od))
paras["N_odour"]= len(odors)
odors= np.array(odors)
np.save(paras["dirname"]+"/"+label+"scan_breadth_odors",odors)

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
