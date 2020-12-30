import numpy as np
import matplotlib.pyplot as plt
from helper import *

cstr="1e-2"
file = open("ornSpkt_"+cstr+".bin", "rb")
ornSpkt = np.load(file)
file.close()
file = open("ornSpkID_"+cstr+".bin", "rb")
ornSpkID = np.load(file)
file.close()

t_total= 5000.0
sigma= 100.0
sdfs= make_sdf(ornSpkt, ornSpkID, np.arange(0,160), -3*sigma, t_total+3*sigma, 1.0, sigma)
plt.figure()
plt.imshow(sdfs, extent=[-3*sigma,t_total+3*sigma,0,160], aspect='auto')
plt.colorbar()
plt.savefig("ORNsdfmap_"+cstr+".png",dpi=300)

figure, axes= plt.subplots(4)
t_array= np.arange(-3*sigma, t_total+3*sigma, 1.0)
j= 0
for i in range(78,82):
    axes[j].plot(t_array, sdfs[i,:])
    j= j+1
plt.savefig("ornSDFtraces_"+cstr+".png",dpi=300)

plt.show()
