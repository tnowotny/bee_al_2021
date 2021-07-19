from OR import or_params, or_ini
from neuron import orn_params, orn_ini, pn_params, pn_ini, ln_params, ln_ini
from synapse import n_orn_pn, orns_pns_ini, orns_pns_post_params, n_orn_ln, orns_lns_ini, orns_lns_post_params, pns_lns_ini, pns_lns_post_params, lns_pns_g, lns_pns_post_params, lns_lns_g, lns_lns_post_params
import sim
import subprocess

def std_paras():
    paras= dict()
    paras["or_params"]= or_params
    paras["or_ini"]= or_ini
    paras["orn_params"]= orn_params
    paras["orn_ini"]= orn_ini
    paras["pn_params"]= pn_params
    paras["pn_ini"]= pn_ini
    paras["ln_params"]= ln_params
    paras["ln_ini"]= ln_ini
    paras["n_orn_pn"]= n_orn_pn
    paras["orns_pns_ini"]= orns_pns_ini
    paras["orns_pns_post_params"]= orns_pns_post_params
    paras["n_orn_ln"]= n_orn_ln
    paras["orns_lns_ini"]= orns_lns_ini
    paras["orns_lns_post_params"]= orns_lns_post_params
    paras["pns_lns_ini"]= pns_lns_ini
    paras["pns_lns_post_params"]= pns_lns_post_params
    paras["lns_pns_g"]= lns_pns_g 
    paras["lns_pns_post_params"]= lns_pns_post_params
    paras["lns_lns_g"]= lns_lns_g
    paras["lns_lns_post_params"]= lns_lns_post_params
    paras["spk_rec_steps"]= 10000
    paras["n_glo"]= 160
    paras["n"]=  {
        "ORNs": 60,
        "PNs": 5,
        "LNs": 25
    }
    paras["N_odour"]= 100
    paras["mu_sig"]= 3
    paras["sig_sig"]= 0.5
    paras["min_sig"]= 1.5
    paras["odor_clip"]= 0.0
    paras["mean_act"]= 0.02
    paras["sig_act"]= 0.01
    paras["min_act"]= 0.01
    paras["max_act"]= 0.05
    paras["mean_A"]= 2.0
    paras["sig_A"]= 0.7
    paras["min_A"]= 0.0
    paras["max_A"]= 4.0
    result = subprocess.run(['git', 'log'], stdout=subprocess.PIPE)
    paras["git_version"]= result.stdout.decode('utf-8').split("\n")[0]
    paras["progress_display"]= True
    return paras
