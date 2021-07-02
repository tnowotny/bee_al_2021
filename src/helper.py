import numpy as np

"""
calculate SDFs from spike times. The returned SDF has time along axis 0
and neuron id along axis 1
"""

def make_sdf(sT, sID, allID, t0, tmax, dt, sigma):
    tleft= t0-3*sigma
    tright= tmax+3*sigma
    n= int((tright-tleft)/dt)
    sdfs= np.zeros((n,len(allID)))
    kwdt= 3*sigma
    i= 0
    x= np.arange(-kwdt,kwdt,dt)
    x= np.exp(-np.power(x,2)/(2*sigma*sigma))
    x= x/(sigma*np.sqrt(2.0*np.pi))*1000.0
    if sT is not None:
        for t, sid in zip(sT, sID):
            if (t > t0 and t < tmax): 
                left= int((t-tleft-kwdt)/dt)
                right= int((t-tleft+kwdt)/dt)
                if right <= n:
                    sdfs[left:right,sid]+=x
           
    return sdfs
                  

def set_odor_simple(ors, slot, odor, c, n):
    # Set the k's according to the "simple model" (not Ho Ka extension)
    # In this case the k's are independent of other present odors
    # ors - "neuron" population of ORs
    # slot - string: the odor slot to use "0", "1" or "2"
    # odor - array containing the relative activation of OR types (size num glo), normalized to sum 1
    # c - concentration in "dilution terms", so values 1e-7 ... 1e-1
    # n - Hill coefficient
    od= np.squeeze(odor)
    kp1cn= np.power(od[:,0]*c,n)
    # print(kp1cn)
    km1= 0.025
    kp2= od[:,1]
    km2= 0.025

    vname= "kp1cn_"+slot
    ors.vars[vname].view[:]= kp1cn
    vname= "km1_"+slot
    ors.vars[vname].view[:]= km1
    vname= "kp2_"+slot
    ors.vars[vname].view[:]= kp2
    vname= "km2_"+slot
    ors.vars[vname].view[:]= km2
    

def gauss_odor(n_glo: int, m: float, sig: float, A: float = 1.0, clip: float = 0.0, m_a: float = 0.025, sig_a: float = 0.0, min_a: float = 0.01, max_a: float = 0.05, hom_a: bool = True) -> np.array:
    # NOTE: We do not normalise by area. This is a) unrealistic (most activated
    # glomerulus markedly less activated in a broadly tuned odour than in a
    # narrowly tuned odour) and b) becomes impractical when trying to do "fair
    # comparison" systematically for odours that are wide/narrow
    # n_glo: Number of gomeruli
    # m: midpoint of Gaussian binding profile
    # sig: standard deviation of Gaussian binding profile
    # A: Amplitude of Gaussian binding profile
    # clip: cut-off threshold for Gaussian binding profile: Glomeruli with binding rateless than this are set to binding rate 0
    # m_a: mean value for activation rate
    # sig_a: standard deviation of activation rate
    # min_a: minimal activation rate allowed
    # max_a: maximal activation rate allowed
    # if hom_a is True, all glomeruli have the same activation rate but individual to the odor, otherwise individual rates
    odor= np.zeros((n_glo,2))
    d= np.arange(0,n_glo)
    d= d-m
    d= np.minimum(np.abs(d),np.abs(d+n_glo))
    d= np.minimum(np.abs(d),np.abs(d-n_glo))
    od= np.exp(-np.power(d,2)/(2*np.power(sig,2)))
    od*= np.power(10,A)
    od= np.maximum(od-clip, 0)
    od[od > 0]= od[od > 0]+clip    
    odor[:,0]= od
    if hom_a:
        a= 0.0
        while a < min_a or a > max_a:
            a= np.random.normal(m_a,sig_a)
        odor[:,1]= a
    else:
        for i in range(n_glo):
            a= 0.0
            while a < min_a or a > max_a:
                a= np.random.normal(m_a,sig_a)
            odor[i,1]= a
        
    return odor
