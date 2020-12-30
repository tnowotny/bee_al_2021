from pygenn.genn_model import create_custom_weight_update_class, create_custom_postsynaptic_class, create_custom_sparse_connect_init_snippet_class, create_cmlf_class
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

"""
Parameter values for the ORN to PN synapse
"""
orns_pns_ini = {
    "g": 1.0     
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
orns_pns_connect = create_custom_sparse_connect_init_snippet_class(
    "orn_pn_type_specific",
    param_names= ["n_orn", "n_pn"],
    row_build_code=
        """
        unsigned int glo= (unsigned int) $(id_pre)/$(n_orn);
        unsigned int local_id= $(id_pre) - glo*$(n_orn);
        $(addSynapse, ((unsigned int) local_id/((unsigned int) ($(n_orn)/$(n_pn)))));
        $(endRow);
        """,
    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: 1)()
)


"""
Parameter values for the ORN to LN synapse
"""
orns_lns_ini = {
    "g": 1.0     # weight in (muS?)
    }


"""
Parameter values for the ORN to LN post-synapse
"""
orns_lns_post_params = {
    "tau": 10.0,     # decay timescale in (ms)
    "E": 0.0         # reversal potential in (mV)
    }

