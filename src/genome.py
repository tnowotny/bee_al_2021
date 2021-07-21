import numpy as np


def get_x(paras):
    x= []
    lg10= np.log(10)
    x.append(np.log(paras["orns_pns_ini"]["g"])/lg10)
    x.append(np.log(paras["orns_lns_ini"]["g"])/lg10)
    x.append(np.log(paras["pns_lns_ini"]["g"])/lg10)
    x.append(np.log(paras["lns_pns_g"])/lg10)
    x.append(np.log(paras["lns_lns_g"])/lg10)
    x.append(paras["IAA_sigma"])
    x.append(paras["IAA_A"])
    x.append(paras["geo_sigma"])
    x.append(paras["geo_A"])
    x.append(paras["geo_shift"])
    return np.array(x)

def set_x(paras, x):
    paras["orns_pns_ini"]["g"]= np.power(10,x[0])
    paras["orns_lns_ini"]["g"]= np.power(10,x[1])
    paras["pns_lns_ini"]["g"]= np.power(10,x[2])
    paras["lns_pns_g"]= np.power(10,x[3])
    paras["lns_lns_g"]= np.power(10,x[4])
    paras["IAA_sigma"]= x[5]
    paras["IAA_A"]= x[6]
    paras["geo_sigma"]= x[7]
    paras["geo_A"]= x[8]
    paras["geo_shift"]= x[9]
    return paras
