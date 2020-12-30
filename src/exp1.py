import numpy as np
import matplotlib.pyplot as plt

n_glo= 160
n_orn = 60
n_pn = 5
n_ln = 1

def gauss_odor(n_glo, m, sig):
    d= np.arange(0,n_glo)
    d= d-m
    d= np.minimum(np.abs(d),np.abs(d+n_glo))
    d= np.minimum(np.abs(d),np.abs(d-n_glo))
    od= np.exp(-np.power(d,2)/(2*np.power(sig,2)))
    return od
    

od= gauss_odor(n_glo, 80, 10)
odors= od
od= gauss_odor(n_glo, 90, 10)
odors= np.vstack((odors, od))

def set_odor_simple(ors, slot, odor, c, n):
    # Set the k's according to the "simple model" (not Ho Ka extension)
    # In this case the k's are independent of other present odors
    # ors - "neuron" population of ORs
    # slot - string: the odor slot to use "0", "1" or "2" 
    kp1cn= np.power(odor*c*100,n)
    km1= 0.05
    kp2= 0.5
    km2= 0.05

    vname= "kp1cn_"+slot
    ors.vars[vname].view[:]= kp1cn
    vname= "km1_"+slot
    ors.vars[vname].view[:]= km1
    vname= "kp2_"+slot
    ors.vars[vname].view[:]= kp2
    vname= "km2_"+slot
    ors.vars[vname].view[:]= km2
    # ors.vars["rb_"+slot].view[:]= np.zeros(odor.shape)
    # ors.vars["ra_"+slot].view[:]= np.zeros(odor.shape)
    
    
    
