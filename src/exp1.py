import numpy as np
import matplotlib.pyplot as plt
import time
import os
from helper import *

# prepare writing results sensibly
timestr = time.strftime("%Y-%m-%d")
dirname= timestr+"-runs"
path = os.path.isdir(dirname)
if not path:
    os.makedirs(dirname)
dirname=dirname+"/"

n_glo= 160
n= {
    "ORNs": 60,
    "PNs": 5,
    "LNs": 1
    }

N= {
    "ORNs": n_glo*n["ORNs"],
    "PNs": n_glo*n["PNs"],
    "LNs": n_glo*n["LNs"]
}

dt_sdf= 1.0
sigma_sdf= 300.0
t_total= 20000.0

rec_state= [
#    ("ORs", "ra"),
#    ("ORNs", "V"),
#    ("ORNs", "a"),
#    ("PNs", "V")
]

rec_spikes= [
    "ORNs",
    "PNs",
    "LNs"
    ]

plot_raster= [
    "ORNs",
    "PNs",
    "LNs"
    ]

plot_sdf= {
    "ORNs": range(74,86,2),
    "PNs": range(74,87,2),
    "LNs": range(74,87,2)
    }

label= "1e-6_n07"
concentration= 1e-6
hill_exp= 0.7

od= gauss_odor(n_glo, 80, 10)
odors= od
od= gauss_odor(n_glo, 90, 10)
odors= np.vstack((odors, od))

   
protocol= [
    {
        "t": 1000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-6
        },
    {
        "t": 4000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        },
    {
        "t": 6000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-5
        },
    {
        "t": 9000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        },
    {
        "t": 11000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-4
        },
    {
        "t": 14000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        },
    {
        "t": 16000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-3
        },
    {
        "t": 19000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        }
    ]
    
