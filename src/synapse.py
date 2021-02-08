from pygenn.genn_model import create_custom_weight_update_class, create_custom_postsynaptic_class, create_custom_sparse_connect_init_snippet_class, create_cmlf_class, create_custom_init_var_snippet_class, init_var
import numpy as np

""" 
weightupdate model to pass OR activation into ORNs
"""
pass_or = create_custom_weight_update_class(
    "pass_or",
    param_names=[],
    var_name_types=[],
    sim_code="",
    synapse_dynamics_code=
    """
    $(addToInSyn, $(ra_pre));
    """
)

"""
post-synapse model to pass values through
"""
pass_postsyn = create_custom_postsynaptic_class(
    "pass_postsyn",
    apply_input_code="""
    $(Isyn)+= $(inSyn);
    $(inSyn)= 0.0;
    """
)

"""
Connectivity init snippet for connectivity from ORs to ORNs
"""
ors_orns_connect = create_custom_sparse_connect_init_snippet_class(
    "or_type_specific",
    row_build_code=
        """
        const unsigned int row_length= $(num_post)/$(num_pre);
        const unsigned int offset= $(id_pre)*row_length;
        for (unsigned int k= 0; k < row_length; k++) {
            $(addSynapse, (offset + k));
        }
        $(endRow);
        """,
    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(num_post/num_pre))()
)

n_orn_pn= 12

"""
Initial values for the ORN to PN synapse
"""
orns_pns_ini = {
    "g": 0.15/n_orn_pn     # weight in (muS?)     
    }


"""
Parameter values for the ORN to PN post-synapse
"""
orns_pns_post_params = {
    "tau": 10.0,     # decay timescale in (ms)
    "E": 0.0         # reversal potential in (mV)
    }

"""
Connectivity init snippet for connectivity from ORNs to PNs.
"""

orns_al_connect = create_custom_sparse_connect_init_snippet_class(
    "orn_al_type_specific",
    param_names= ["n_orn", "n_trg", "n_pre"],
    col_build_code=
        """
        if (c == 0) {
        $(endCol);
        }
        const unsigned int glo= $(id_post)/((unsigned int) $(n_trg));
        const unsigned int offset= $(n_orn)*glo;
        const unsigned int tid= $(gennrand_uniform)*$(n_orn);
        $(addSynapse, offset+tid+$(id_pre_begin));
        c--;
        """,
    col_build_state_vars= {("c", "unsigned int", "$(n_pre)")},
    calc_max_col_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(pars[2]))()
)

n_orn_ln= 12   

"""
Parameter values for the ORN to LN synapse
"""
orns_lns_ini = {
    "g": 0.2/n_orn_ln     # weight in (muS?)
    }


"""
Parameter values for the ORN to LN post-synapse
"""
orns_lns_post_params = {
    "tau": 10.0,     # decay timescale in (ms)
    "E": 0.0         # reversal potential in (mV)
    }

    
"""
Connectivity init snippet for connectivity from PNs to LNs. Each PN connects to all LN in its glomerulus
"""
pns_lns_connect = create_custom_sparse_connect_init_snippet_class(
    "pns_lns_within_glo",
    param_names= ["n_pn", "n_ln"],
    row_build_code=
        """
        const unsigned int offset= (unsigned int) $(id_pre)/((unsigned int) $(n_pn))*$(n_ln);
        for (unsigned int k= 0; k < $(n_ln); k++) {
        $(addSynapse, (offset+k));
        }
        $(endRow);
        """,
    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(pars[1]))()
)


"""
Parameter values for the PN to LN synapse
"""
pns_lns_ini = {
    "g": 0.001     # weight in (muS?)
    }


"""
Parameter values for the ORN to LN post-synapse
"""
pns_lns_post_params = {
    "tau": 10.0,     # decay timescale in (ms)
    "E": 0.0       # reversal potential in (mV)
    }


"""
init var snippet for initializing dense LN to PN connections
"""
lns_pns_conn_init = create_custom_init_var_snippet_class(
    "lns_pns_conn_init",
    param_names=["n_ln", "n_pn", "g"],
    var_init_code=
    """
    const unsigned int npn= (unsigned int) $(n_pn);
    const unsigned int nln= (unsigned int) $(n_ln);
    $(value)=($(id_pre)/nln == $(id_post)/npn) ? 0.0 : $(g);
    """
)


"""
Parameter values for the LN to PN synapse
"""
lns_pns_g= 0.0003

"""
Parameter values for the LN to PN post-synapse
"""
lns_pns_post_params = {
    "tau": 20.0,     # decay timescale in (ms)
    "E": -80.0       # reversal potential in (mV)
    }

"""
init var snippet for initializing dense LN to PN connections
"""
lns_lns_conn_init = create_custom_init_var_snippet_class(
    "lns_lns_conn_init",
    param_names=["n_ln", "g"],
    var_init_code=
    """
    const unsigned int nln= (unsigned int) $(n_ln);
    $(value)=($(id_pre)/nln == $(id_post)/nln) ? 0.0 : $(g);
    """
)


"""
Parameter values for the LN to LN synapse
"""
lns_lns_g= 0.0001


"""
Parameter values for the LN to LN post-synapse
"""
lns_lns_post_params = {
    "tau": 20.0,     # decay timescale in (ms)
    "E": -80.0       # reversal potential in (mV)
    }

