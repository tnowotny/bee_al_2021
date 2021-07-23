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
    paras["sig_act"]= 0.02
    paras["min_act"]= 0.003
    paras["max_act"]= 0.2
    paras["mean_A"]= 1.5
    paras["sig_A"]= 0.5
    paras["min_A"]= 0.0
    paras["max_A"]= 4.0
    paras["IAA_sigma"]= 3
    paras["IAA_A"]= 0.8
    paras["IAA_act"]= 0.1
    paras["geo_sigma"]= 10
    paras["geo_A"]= 4.4
    paras["geo_act"]= 0.003
    paras["geo_shift"]= 30
    result = subprocess.run(['git', 'log'], stdout=subprocess.PIPE)
    paras["git_version"]= result.stdout.decode('utf-8').split("\n")[0]
    paras["progress_display"]= True
    paras["print_messages"]= True
    paras["write_to_disk"]= True
    paras["rec_ORN_counts"]= False
    return paras
