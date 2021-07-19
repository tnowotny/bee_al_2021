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
import scipy.optimize as opt

"""
Run an optimization method to align the AL simulation with experimental observations
"""

paras= std_paras()
paras["write_to_disk"]= False
paras["dt"]= sim.dt

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
paras["progress_display"]= False

# Control what to record
paras["rec_state"]= [
#    ("ORs", "ra"),
#    ("ORNs", "V"),
#    ("ORNs", "a"),
#    ("PNs", "V")
]

paras["rec_spikes"]= [
    "ORNs",
    "PNs",
#    "LNs"
    ]

label= "opt"
paras["label"]= label+"_"+connect_I+"_"

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
paras["N_odour"]= 2

paras["IAA_sigma"]= 5
paras["IAA_A"]= paras["max_A"]-3
paras["IAA_act"]= paras["max_act"]*2
paras["geo_sigma"]= 10
paras["geo_A"]= paras["max_A"]+1.0
paras["geo_act"]= paras["min_act"]/3.0
paras["geo_shift"]= 38

# define the inhibitory connectivity pattern in the antennal lobe
correl= choose_inh_connectivity(paras,connect_I)
                
# Now, let's make a protocol where each odor is presented for 3 secs with
# 3 second breaks and at each of 25 concentration values
protocol= []
t_off= 3000.0

for c1 in [ 0, 1e-3, 1e-1 ]:
    for c2 in [ 0, 1e-6, 1e-5, 1e-4, 1e-3 ]:
        if c1 != 0:
            sub_prot= {
                "t": t_off,
                "odor": 0,
                "ochn": "0",
                "concentration": c1
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
                "t": t_off,
                "odor": 1,
                "ochn": "1",
                "concentration": c2
            }
            protocol.append(sub_prot)
            sub_prot= {
                "t": t_off+3000.0,
                "odor": 1,
                "ochn": "1",
                "concentration": 0.0
            }
            protocol.append(sub_prot)
        t_off+= 6000.0;

paras["t_total"]= t_off
print("We are running for a total simulated time of {}ms".format(t_off))

x0= np.array([
    paras["orns_pns_ini"]["g"],    # 0
    paras["orns_lns_ini"]["g"],    # 1
    paras["pns_lns_ini"]["g"],     # 2
    paras["lns_pns_g"],            # 3
    paras["lns_lns_g"],            # 4
    paras["IAA_sigma"],            # 5
    paras["IAA_A"],                # 6
#    paras["IAA_act"],              # 7
    paras["geo_sigma"],            # 8
    paras["geo_A"],                # 9
#    paras["geo_act"],              # 10
    paras["geo_shift"]             # 11
    ])
    
bounds= [
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (2.0, 6.0),
    (-3, 5),
#    (0.01, 0.1),
    (5.0, 11.0),
    (-3, 5),
#    (1e-5, 0.1),
    (0.0, 80)
    ]

def evaluate(x, *args) -> float:
    assert len(args) == 3
    paras= args[1]
    odors= []
    od= gauss_odor(paras["n_glo"], paras["n_glo"]//2, x[5], x[6], paras["odor_clip"], paras["IAA_act"], 0.0, 1e-10, 0.1)
    odors.append(np.copy(od))
    # Add "Geosmin" that is particularly early binding, broad, and low activating
    od= gauss_odor(paras["n_glo"], paras["n_glo"]//2+x[9], x[7], x[8], paras["odor_clip"], paras["geo_act"], 0.0, 1e-10, 0.1)
    odors.append(np.copy(od))
    odors= np.array(odors)
    paras["orns_pns_ini"]["g"]= x[0]
    paras["orns_lns_ini"]["g"]= x[1]
    paras["pns_lns_ini"]["g"]= x[2]
    paras["lns_pns_g"]= x[3]
    paras["lns_lns_g"]= x[4]
   
    state_bufs, spike_t, spike_ID, ORN_cnts= ALsim(odors, args[0], paras, protocol, lns_gsyn= args[2])
    err= []
    # first reproduce EAG data
    d= np.zeros(ORN_cnts.shape[0]//3)
    for i in range(ORN_cnts.shape[0]//3):
        d[i]= np.sum(ORN_cnts[i*3:(i+1)*3])
    r= d[1:]/d[:-1]
    # experimentally observed ratios
    ratio= [
        1.0,
        2.0,
        7.0/2.0,
        13.0/7.0,
        11.5/13.0,
        10.7/11.5,
        11.5/10.7,
        16.0/11.5,
        22.0/16.0,
        98.0/22.0,
        1.0,
        95.0/98.0,
        98.0/95.0,
        105.0/98.0
        ]
    err.append(np.linalg.norm(r-ratio))
    
    # now some constraints on pure odors' PN activity
    st= spike_t["PNs"]
    id= spike_ID["PNs"]
    jchoice= [ 1, 4, 5, 10, 14 ]
    sno= np.zeros(len(jchoice))
    hst= 3000.0 # half sample time
    cnt= 0
    lbound= (paras["n_glo"]//2-3*paras["IAA_sigma"])*paras["n"]["PNs"]
    rbound= (paras["n_glo"]//2+3*paras["IAA_sigma"])*paras["n"]["PNs"]
    li= 0
    for j in jchoice:
        left= hst+2*j*hst
        right= left+hst
        while li < len(st) and st[li] < left:
            li+= 1
        ri= li
        while ri < len(st) and st[ri] < right:
            ri+= 1
        if j == 14:
            # here we are counting spikes in "non-IAA" glomeruli
            the_id= id[li:ri]
            sno[cnt]= len(the_id[the_id < lbound])+len(the_id[the_id > rbound])
        else:
            sno[cnt]= ri-li
        cnt+= 1

    err.append(np.maximum(1000-sno[0],0)) # at least 1000 spikes for IAA at 10^-6
    err.append(sno[1])
    err.append(np.maximum(2000-sno[3],0)) # at least 3000 spikes for IAA at 10^-1
    err.append(np.maximum(sno[2]-sno[3],0))
    err.append(sno[4])
    with open("progress.txt","a") as f:
        f.write("x is {}\n".format(x))
        f.write(" and err is {}.\n".format(err))
        f.close()
    wght= np.array([
        1.0,
        0.05,
        0.05,
        0.01,
        0.02,
        0.05
        ])
    return np.dot(err,wght)

#print(evaluate(x0,hill_exp,paras,correl))
opt.minimize(evaluate, x0, args=(hill_exp,paras,correl), method="Nelder-Mead", bounds= bounds, options={"maxiter": 1000, "disp": True})
