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
from genome import set_x

connect_I= "hom"
dirname="opt-dir"
paras= std_paras()
paras["dirname"]= dirname
paras["write_to_disk"]= False
paras["dt"]= sim.dt
paras["use_spk_rec"]= True
paras["progress_display"]= True
paras["trial_time"]= 12000.0
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
    "LNs"
    ]

label= "opt"
paras["label"]= label+"_"+connect_I+"_"
# Assume a uniform distribution of Hill coefficients inspired by Rospars'
# work on receptors tiling the space of possible sigmoid responses
hill_new= True

if hill_new:
    hill_exp= np.random.uniform(0.95, 1.05, paras["n_glo"])
else:
    hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")


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
        if c2 != 0:
            sub_prot= {
                "t": t_off,
                "odor": 1,
                "ochn": "1",
                "concentration": c2
            }
            protocol.append(sub_prot)
        if c1 != 0:
            sub_prot= {
                "t": t_off+3000.0,
                "odor": 0,
                "ochn": "0",
                "concentration": 0
            }
            protocol.append(sub_prot)
        if c2 != 0:
            sub_prot= {
                "t": t_off+3000.0,
                "odor": 1,
                "ochn": "1",
                "concentration": 0
            }
            protocol.append(sub_prot)
        t_off+= paras["trial_time"];

paras["t_total"]= t_off
print("We are running for a total simulated time of {}ms".format(t_off))

x= np.array([
    0.008, 0.008, 0.001, 0.55e-4, 0.2e-4, 3.0e+00, 0.8e+00, 10, 4.4e+00, 30 ])
x[:5]= np.log(x[:5])/np.log(10)
print(x)

paras= set_x(paras,x)
# Generate odors or load previously generated odors from file
paras["geo_act"]= 0.003 #0.006
paras["N_odour"]= 2
odors= []
od= gauss_odor(paras["n_glo"], paras["n_glo"]//2, paras["IAA_sigma"], paras["IAA_A"], paras["odor_clip"], paras["IAA_act"], 0.0, 1e-10, 1.0)
odors.append(np.copy(od))
# Add "Geosmin" that is particularly early binding, broad, and low activating
od= gauss_odor(paras["n_glo"], paras["n_glo"]//2+paras["geo_shift"], paras["geo_sigma"], paras["geo_A"], paras["odor_clip"], paras["geo_act"], 0.0, 1e-10, 1.0)
odors.append(np.copy(od))
odors= np.array(odors)
   
state_bufs, spike_t, spike_ID, ORN_cnts= ALsim(odors, hill_exp, paras, protocol, lns_gsyn= correl)

plt.figure()
plt.plot(ORN_cnts)
avgNo= int(paras["trial_time"]/(paras["spk_rec_steps"]*sim.dt))
d= np.zeros(ORN_cnts.shape[0]//avgNo)
for i in range(ORN_cnts.shape[0]//avgNo):
        d[i]= np.sum(ORN_cnts[i*avgNo:(i+1)*avgNo])

plt.figure()
plt.bar(np.arange(d.shape[0]),d)
plt.savefig("bars.png")

def glo_avg(sdf: np.ndarray, n):
    nglo= sdf.shape[1]//n
    gsdf= np.zeros((sdf.shape[0],nglo))
    for i in range(nglo):
        gsdf[:,i]= np.mean(sdf[:,n*i:n*(i+1)],axis=1)
    return gsdf

def force_aspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


# now plot responses in PNs and LNs
plt.figure()
plt.scatter(spike_t["PNs"],spike_ID["PNs"],marker='.',s=0.2)
plt.scatter(spike_t["LNs"],spike_ID["LNs"]/5,marker='.',s=0.2)
#plt.savefig("spikeRaster.png",dpi=600)
plt.show()

for which, N in zip(["PNs","LNs"], [800, 4000]):
    st= spike_t[which]
    id= spike_ID[which]
    jchoice= [ 1, 4, 10, 11, 14 ]
    sno= np.zeros(len(jchoice))
    hst= 3000.0 # half sample time
    cnt= 0
    li= 0
    sigma_sdf= 100
    dt_sdf= 1
    lsdfs= []
    gsdfs= []
    for j in jchoice:
        left= hst+j*paras["trial_time"]
        right= left+hst
        while li < len(st) and st[li] < left:
            li+= 1
            ri= li
        while ri < len(st) and st[ri] < right:
            ri+= 1
        lsdfs.append(make_sdf(st[li:ri], id[li:ri], np.arange(0,N), left-3*sigma_sdf, right+3*sigma_sdf, dt_sdf, sigma_sdf))
        gsdfs.append(glo_avg(lsdfs[-1],5))

    plt.rc('font', size=7) #controls default text size
    mn= -10
    mx= 40
    fig, ax= plt.subplots(1,5)
    print(ax.shape)
    for i in range(5):
        ts= np.transpose(gsdfs[i][:,:])
        ax[i].imshow(ts,cmap="hot")
        force_aspect(ax[i],0.4)
        ax[i].set_yticklabels([])
        print(np.max(np.max(gsdfs[i])))
    #plt.savefig("maps"+which+".png")
    fig, ax= plt.subplots(1,5)
    for i in range(5):
        idx= np.argmax(np.mean(gsdfs[i][:,:],axis=0))
        ax[i].plot(gsdfs[i][:,idx])
    #plt.savefig("lines"+which+".png")
    plt.show()        
