import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from exp1_plots import exp1_plots
from ALsim import ALsim
import sim
import sys
import os
from ALsimParameters import std_paras
import random

"""
experiment to investigate the effect of decreasing response with higher concentration. In this version we generate N_odour odours randomly with the following properties: 
1. Each odour has a a Gaussian profile of glomerulus activation with sigma drawn from Gauss(mu_sig,sig_sig). 
2. the Gaussian odour profile is over a random permutation of the glomeruli.
"""

if len(sys.argv) < 5:
    print("usage: python example_traces.py <run#> <connect_I: corr0/corr1/cov0/cov1> <odor 1> <odor 2>" )
    exit()

ino= float(sys.argv[1])
connect_I= sys.argv[2]
o1= int(sys.argv[3])
o2= int(sys.argv[4])

paras= std_paras()

if ino == -100:
    paras["lns_pns_g"]= 0
    paras["lns_lns_g"]= 0
else:
    paras["lns_pns_g"]*= np.power(np.sqrt(10),ino)
    paras["lns_lns_g"]*= np.power(np.sqrt(10),ino)

# write results into a dir with current date in the name
timestr = time.strftime("%Y-%m-%d")
paras["dirname"]= timestr+"-runs"

path = os.path.isdir(paras["dirname"])
if not path:
    print("making dir "+paras["dirname"])
    os.makedirs(paras["dirname"])

paras["plotting"]= True
paras["plotdisplay"]= False
paras["use_spk_rec"]= True

paras["rec_state"]= [
    ("ORs", "ra"),
    ("ORNs", "V"),
#    ("ORNs", "a"),
    ("PNs", "V"),
    ("LNs", "V")
]

paras["rec_spikes"]= [
    "ORNs",
    "PNs",
    "LNs"
    ]

paras["plot_raster"]= [
#    "ORNs",
    "PNs",
#    "LNs"
    ]

paras["plot_sdf"]= {
#    "ORNs": range(74,86,2),
    "PNs": list(range(74,87,2)),
#    "LNs": list(range(74,87,2))
    }

label= "test_new"
paras["label"]= label+"_"+str(ino)+"_"+str(o1)+"_"+str(o2)


# Load Hill coefficients and odors from file
hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")
odors= np.load(paras["dirname"]+"/"+label+"_odors.npy")
paras["N_odour"]= odors.shape[0]

HOMO_LN_GSYN= False
if connect_I == "corr0":
    correl= np.corrcoef(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
    correl= (correl+1.0)/20.0 # extra factor 10 in comparison to covariance ...
    for i in range(paras["n_glo"]):
        correl[i,i]= 0.0
    print("AL inhibition with correlation, no self-inhibition")
else:
    if connect_I == "corr1":
        correl= np.corrcoef(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
        correl= (correl+1.0)/20.0 # extra factor 10 in comparison to covariance ...
        print("AL inhibition with correlation and self-inhibition")
    else:
        if connect_I == "cov0":
            correl= np.cov(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
            correl= np.maximum(0.0, correl)
            for i in range(paras["n_glo"]):
                correl[i,i]= 0.0
            print("AL inhibition with covariance, no self-inhibition")
        else:
             if connect_I == "cov1":
                 correl= np.cov(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
                 correl= np.maximum(0.0, correl)
                 print("AL inhibition with covariance and self-inhibition")
             else:
                 correl= np.ones((paras["n_glo"],paras["n_glo"]))
                 HOMO_LN_GSYN= True
                 print("Homogeneous AL inhibition")
                 

# Now, let's make a protocol where each odor is presented for 3 secs with
# 3 second breaks and at two representative concentration values
paras["protocol"]= []
t_off= 3000.0

for o in [o1, o2]:
    for c in [ 1e-5, 1e-3 ]:
        sub_prot= {
            "t": t_off,
            "odor": o,
            "ochn": "0",
            "concentration": c
        }
        paras["protocol"].append(sub_prot)        
        sub_prot= {
            "t": t_off+3000.0,
            "odor": o,
            "ochn": "0",
            "concentration": 0.0,
        }
        paras["protocol"].append(sub_prot)
        t_off+= 6000.0;

paras["t_total"]= t_off+3000.0
print("We are running for a total simulated time of {}ms".format(t_off))

if __name__ == "__main__":
    state_bufs, spike_t, spike_ID= ALsim(odors, hill_exp, paras, lns_gsyn= correl)

    if paras["plotting"]:
        for i in range(4):
            ioff= int((i*6000.0+2900.0)/paras["dt"])
            toff= ioff*dt
            # plot raw data of the responses - bound receptor, ORN V, PN V, LN V
            figure, axes= plt.subplots(4)
            t= np.arange(t, t+3500, dt)
            npts= t.shape[0]
            # find the strongest glomerulus
            ra= state_bufs["ORs_ra"][ioff:ioff+npts,:]
            ra_sum= np.sum(ra, axis=1)
            idx= np.argmax(ra_sum)
            axes[0].plot(t, ra[:,idx])
            VORN= state_bufs["ORNs_V"][ioff:ioff+npts,:]
            axes[1].plot(t, VORN[:,idx])
            VPN= state_bufs["PNs_V"][ioff:ioff+npts,:]
            axes[2].plot(t, VPN[:,idx])
            VLN= state_bufs["LNs_V"][ioff:ioff+npts,:]
            axes[3].plot(t, VLN[:,idx])
        plt.show()
