from pygenn.genn_model import create_custom_neuron_class
import numpy as np
import sim

adaptive_LIF = create_custom_neuron_class(
    "adaptive_LIF",
    param_names= ["C_mem", "V_reset", "V_thresh", "V_leak", "g_leak", "r_scale", "g_adapt", "V_adapt", "tau_adapt", "noise_A"],
    var_name_types= [("V", "scalar"), ("a", "scalar")],
    sim_code= """
    $(V)+= (-$(g_leak)*($(V)-$(V_leak)) - $(g_adapt)*$(a)*($(V)-$(V_adapt)) + $(r_scale)*$(Isyn)+$(noise_A)*$(gennrand_normal))*DT/$(C_mem);
    $(a)+= -$(a)*DT/$(tau_adapt);
    """,
    threshold_condition_code= """
    $(V) >= $(V_thresh)
    """,
    reset_code= """
    $(V)= $(V_reset);
    $(a)+= 0.5;
    """
    )

orn_params = {"C_mem": 1.0,
              "V_reset": -70.0,
              "V_thresh": -40.0,
              "V_leak": -60.0,
              "g_leak": 0.01,
              "r_scale": 5.0,
              "g_adapt": 0.005,
              "V_adapt": -70.0,
              "tau_adapt": 1000.0,
              "noise_A": 2.1/np.sqrt(sim.dt)
              }

orn_ini = {"V": -60.0,
           "a": 0.0
           }


pn_params = {"C_mem": 2.0,
             "V_reset": -70.0,
             "V_thresh": -40.0,
             "V_leak": -60.0,
             "g_leak": 0.01,
             "r_scale": 1.0,
             "g_adapt": 0.005,
             "V_adapt": -70.0,
             "tau_adapt": 1000.0,
             "noise_A": 1.4/np.sqrt(sim.dt)
             }

pn_ini = {"V": -60.0,
          "a": 0.0
          }


ln_params = {"C_mem": 1.0,
             "V_reset": -70.0,
             "V_thresh": -35.0,
             "V_leak": -60.0,
             "g_leak": 0.01,
             "r_scale": 1.0,
             "g_adapt": 0.005,
             "V_adapt": -70.0,
             "tau_adapt": 1000.0,
             "noise_A": 0.7/np.sqrt(sim.dt)
             }

ln_ini = {"V": -60.0,
           "a": 0.0
           }

