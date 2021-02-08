"""
Model for a honeybee olfactory receptor (OR) type.
The receptor is modeled with a system of ODEs which govern the 
fraction of unbound receptors (r0), the receptors bound to odorant i
(rb0, ..., rb2), and the receptors bound and activated by odorant i
(ra0, ..., ra2).
Was formulated, the model supports 3 active chemicals at any given 
time. It is the user's responsibility to pick the "odor channel" for
each odorant by setting the corresponding (un)binding constants.
Note: If rb*,ra* are still non-zero from a previous activation (potentially
by a different chemical) this could lead to artefacts!
If needed, additional odor channels can be added by appropriate copy/paste
and adjustment of formula below.

Parameters: 

Variables: r0 - unbound receptor fraction
           rb_0 ... rb_2 - bound receptor fraction to "odor channels" 0 to 2
           ra_0 ... ra_2 - activated receptor to "odor channels" 0 to 2
           ra - sum of activated receptors
           kp1cn_0 ... kp1n_2 - binding rates (including concentration and Hill exponent) 0 to 2
           km1_0 ... km1_2 - unbinding rates 0 to 2
           kp2_0 ... kp2_2 - activation rates 0 to 2
           km2_0 ... km2_2 - inactivation rates 0 to 2
           
OR model can be used to obtain the model described in: 
T. Nowotny, Jacob S. Stierle, C. Giovanni Galizia, Paul Szyszka, Data-driven honeybee antennal lobe model suggests how stimulus-onset asynchrony can aid odour segregation, Brain Research, 1536: 119-134 (2013) DOI: 10.1016/j.brainres.2013.05.038

*or* by setting kp1cn_* appropriately the model in
H. K. Chan, F. Hersperger, E. Marachlian, B. H. Smith, F. Locatelli, P. Szyszka, T. Nowotny (2018) Odorant mixtures elicit less variable and faster responses than pure odorants. PLoS Comp Biol 14(12):e1006536. doi: 10.1371/journal.pcbi.1006536

"""

from pygenn.genn_model import create_custom_neuron_class

or_model = create_custom_neuron_class(
    "or_model",
    param_names=[],
    var_name_types=[("r0", "scalar"), ("rb_0", "scalar"), ("ra_0", "scalar"),
                    ("rb_1", "scalar"), ("ra_1", "scalar"),
                    ("rb_2", "scalar"), ("ra_2", "scalar"),
                    ("ra", "scalar"),
                    ("kp1cn_0", "scalar"), ("km1_0", "scalar"), ("kp2_0", "scalar"), ("km2_0", "scalar"),
                    ("kp1cn_1", "scalar"), ("km1_1", "scalar"), ("kp2_1", "scalar"), ("km2_1", "scalar"),
                    ("kp1cn_2", "scalar"), ("km1_2", "scalar"), ("kp2_2", "scalar"), ("km2_2", "scalar"),
                    ],
    sim_code="""
    // update all bound receptors and activated receptors
    $(rb_0)+= ($(kp1cn_0)*$(r0) - $(km1_0)*$(rb_0) + $(km2_0)*$(ra_0) - $(kp2_0)*$(rb_0))*DT;
    if ($(rb_0) > 1.0) $(rb_0)= 1.0;
    $(ra_0)+= ($(kp2_0)*$(rb_0) - $(km2_0)*$(ra_0))*DT;
    if ($(ra_0) > 1.0) $(ra_0)= 1.0;
    $(rb_1)+= ($(kp1cn_1)*$(r0) - $(km1_1)*$(rb_1) + $(km2_1)*$(ra_1) - $(kp2_1)*$(rb_1))*DT;
    if ($(rb_1) > 1.0) $(rb_1)= 1.0;
    $(ra_1)+= ($(kp2_1)*$(rb_1) - $(km2_1)*$(ra_1))*DT;
    if ($(ra_1) > 1.0) $(ra_1)= 1.0;
    $(rb_2)+= ($(kp1cn_2)*$(r0) - $(km1_2)*$(rb_2) + $(km2_2)*$(ra_2) - $(kp2_2)*$(rb_2))*DT;
    if ($(rb_2) > 1.0) $(rb_2)= 1.0;
    $(ra_2)+= ($(kp2_2)*$(rb_2) - $(km2_2)*$(ra_2))*DT;
    if ($(ra_2) > 1.0) $(ra_2)= 1.0;
    // now update ra and calculate the sum of bound receptors
    scalar rb= $(rb_0) + $(rb_1) + $(rb_2);
    if (rb > 1.0) rb= 1.0;
    $(ra)= $(ra_0) + $(ra_1) + $(ra_2);
    if ($(ra) > 1.0) $(ra)= 1.0;
    // then update r0 as a function of rb and ra
    $(r0)= 1.0 - rb - $(ra);
    if ($(r0) < 0.0) $(r0)= 0.0;
    """,
    reset_code="",
    threshold_condition_code=""
)

or_params = {}

or_ini = {"r0": 1.0,
          "rb_0": 0.0, "ra_0": 0.0,
          "rb_1": 0.0, "ra_1": 0.0,
          "rb_2": 0.0, "ra_2": 0.0,
          "ra": 0.0,
          "kp1cn_0": 0.0, "km1_0": 0.0, "kp2_0": 0.0, "km2_0": 0.0,
          "kp1cn_1": 0.0, "km1_1": 0.0, "kp2_1": 0.0, "km2_1": 0.0,
          "kp1cn_2": 0.0, "km1_2": 0.0, "kp2_2": 0.0, "km2_2": 0.0,
          }

