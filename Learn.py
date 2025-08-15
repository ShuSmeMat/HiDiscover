#!/usr/bin/env python3

"""
Authors: Hanyin Zhang, Haoyuan Li
Date created: 2022
Description: The hidiscover main function
"""

import json
from modContLearn import *
import sys

assert len(sys.argv)==2
file_json=sys.argv[1] 
with open(file_json)as f:
    json_input = json.load(f)
    dataset_dict = json_input['dataset'][0]  
    parameter_dict = json_input['parameters'][0]

contLearn(dataset_dict,parameter_dict,kernel=semiKmeansClusterKernel)

