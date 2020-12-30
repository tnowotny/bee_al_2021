from pygenn.genn_model import create_custom_neuron_class


adaptive_LIF = create_custom_neuron_class(
    "adaptive_LIF",
    param_names= ["V_rest", "V_thresh", "g_leak", "r_scale", "g_adapt", "V_adapt", "tau_adapt", "noise_A"],
    var_name_types= [("V", "scalar"), ("a", "scalar")],
    sim_code= """
    $(V)+= (-$(g_leak)*($(V)-$(V_rest)) - $(g_adapt)*$(a)*($(V)-$(V_adapt)) + $(r_scale)*$(Isyn)+$(noise_A)*$(gennrand_normal))*DT;
    $(a)+= -$(a)*DT/$(tau_adapt);
    """,
    threshold_condition_code= """
    $(V) >= $(V_thresh)
    """,
    reset_code= """
    $(V)= $(V_rest);
    $(a)+= 0.5;
    """
    )

orn_params = {"V_rest": -60.0,
              "V_thresh": -20.0,
              "g_leak": 0.01,
              "r_scale": 20.0,
              "g_adapt": 0.2,
              "V_adapt": -60.0,
              "tau_adapt": 50.0,
              "noise_A": 1.0
              }

orn_ini = {"V": -60.0,
           "a": 0.0
           }


pn_params = {"V_rest": -60.0,
              "V_thresh": -20.0,
              "g_leak": 0.01,
              "r_scale": 20.0,
              "g_adapt": 0.2,
              "V_adapt": -60.0,
              "tau_adapt": 50.0,
              "noise_A": 1.0
              }

pn_ini = {"V": -60.0,
           "a": 0.0
           }


ln_params = {"V_rest": -60.0,
              "V_thresh": -20.0,
              "g_leak": 0.01,
              "r_scale": 20.0,
              "g_adapt": 0.2,
              "V_adapt": -60.0,
              "tau_adapt": 50.0,
              "noise_A": 1.0
              }

ln_ini = {"V": -60.0,
           "a": 0.0
           }
