#!/usr/bin/env python3

"""
Authors: Hanyin Zhang, Haoyuan Li
Date created: 2022
Description: files I/O 
"""

import numpy as np

total_numbers = 200000
ratios = [0.8, 0.1, 0.1]
num_elements = [int(total_numbers * ratio ) for ratio in ratios]
all_numbers = np.arange(1, total_numbers + 1)
np.random.shuffle(all_numbers)
partition_train = all_numbers[:num_elements[0]]
partition_val = all_numbers[num_elements[0]:num_elements[0] + num_elements[1]]
partition_test = all_numbers[num_elements[0] + num_elements[1]:num_elements[0] + num_elements[1]+num_elements[2]]

np.save("train_set_index.npy", partition_train)
np.save("val_set_index.npy", partition_val)
np.save("test_set_index.npy", partition_test)

