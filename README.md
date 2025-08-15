# HiDiscover

A python code to perform incremental learning on dataset from MD.

This code has been tested on CentOS 7.5 with python 3.9.7. It depends on the following python modules: numpy, scipy, copy, os, sys, json.

Authors: Hanyin Zhang and Haoyuan Li

# Instructions

## Example datasets and their format:

### md1/
Assume the data size is N1. This folder contains three files:

>dataSet_count.npy: A numpy array of size N1, consisting of integers from 1 to N1.

>dataSet_dist.npy: A numpy matrix of shape (N1, M1, M1). The size of the Coulombic matrix is M1 × M1.

>dataSet_name.npy: A numpy matrix of shape (N1, M1). Every row contains the atom set names.


### md2/
Assume the data size is N2. The format of the dataset is the same as md1. 

>dataSet_count.npy

>dataSet_dist.npy

>dataSet_name.npy


## Generate index for training, validation and test sets:
Enter the folder md1.

Open a python terminal:
>import numpy as np

>total_numbers = 2000000   # change this to the actural data size

>ratios = [0.8, 0.1, 0.1]

>num_elements = [int(total_numbers * ratio ) for ratio in ratios]

>all_numbers = np.arange(1, total_numbers + 1)

>np.random.shuffle(all_numbers)

>partition_train = all_numbers[:num_elements[0]]

>partition_val = all_numbers[num_elements[0]:num_elements[0] + num_elements[1]]

>partition_test = all_numbers[num_elements[0] + num_elements[1]:num_elements[0] + num_elements[1]+num_elements[2]]

>np.save("train_set_index.npy", partition_train)

>np.save("val_set_index.npy", partition_val) 

>np.save("test_set_index.npy", partition_test)

Do the same for md2

## Train the model
We use the above datasets as an example and describe the training, validation and testing processes. Assume that the code folder is in the same working directory as the datasets.

### (i). Cluster the dataset from md1

In a bash terminal, type:
>python3 ./code/Learn.py 01.json > 01.log

01.json:
>{

>    "dataset":[

>        {

>            "name_filename":"PATH_TO_DATA/md1/dataSet_name.npy",

>            "distmatrix_filename":"PATH_TO_DATA/md1/dataSet_dist.npy",

>            "index_filename":"PATH_TO_DATA/md1/dataSet_count.npy",

>            "subset_index_filename":"PATH_TO_DATA/md1/train_set_index.npy"

>        }

>    ],

>    "parameters":[

>        {

>            "ncluster":5,

>            "num_known_centroid":0,

>            "known_centroid_name_list":[],

>            "known_centroid_distmatrix_list":[],

>            "known_centroid_index_list":[],

>            "criteria":0.001

>        }

>    ]

>}

Set PATH_TO_DATA to the correct absolute path name.

We then rename the folders:
>mv centroidList 01_centroidList

>mv clusteredPoints 01_clusteredPoints

### (ii). Cluster the dataset from md2
In the bash terminal, type:
>python3 ./code/Learn.py 02.json > 02.log

>02.json:

>{

>    "dataset":[

>        {

>            "name_filename":"PATH_TO_DATA/md2/dataSet_name.npy",

>            "distmatrix_filename":"PATH_TO_DATA/md2/dataSet_dist.npy",

>            "index_filename":"PATH_TO_DATA/md2/dataSet_count.npy",

>            "subset_index_filename":"PATH_TO_DATA/md2/train_set_index.npy"

>        }

>    ],

>    "parameters":[

>        {

>            "ncluster":6,

>            "num_known_centroid":5,

>            "known_centroid_name_list": ["PATH_TO_DATA/01_centroidList/name0.npy", "PATH_TO_DATA/01_centroidList/name1.npy", "PATH_TO_DATA/01_centroidList/name2.npy", "PATH_TO_DATA/01_centroidList/name3.npy", "PATH_TO_DATA/01_centroidList/name4.npy"],

>            "known_centroid_distmatrix_list": ["PATH_TO_DATA/01_centroidList/dist0.npy", "PATH_TO_DATA/01_centroidList/dist1.npy", "PATH_TO_DATA/01_centroidList/dist2.npy", "PATH_TO_DATA/01_centroidList/dist3.npy", "PATH_TO_DATA/01_centroidList/dist4.npy"],

>            "known_centroid_index_list": ["PATH_TO_DATA/01_centroidList/index0.npy", "PATH_TO_DATA/01_centroidList/index1.npy", "PATH_TO_DATA/01_centroidList/index2.npy", "PATH_TO_DATA/01_centroidList/index3.npy", "PATH_TO_DATA/01_centroidList/index4.npy"],

>            "criteria":0.001

>        }

>    ]

>}

Set PATH_TO_DATA to the correct absolute path name.

We then rename folders:
>mv centroidList 02_centroidList

>mv clusteredPoints 02_clusteredPoints

### (iii). Cluster the subset from the last step

In the bash terminal, type:
>python3 ./code/Learn.py 03.json > 03.log

03.json:
>{

>    "dataset":[

>        {

>            "name_filename":"PATH_TO_DATA/md2/dataSet_name.npy",

>            "distmatrix_filename":"PATH_TO_DATA/md2/dataSet_dist.npy",

>            "index_filename":"PATH_TO_DATA/md2/dataSet_count.npy",

>            "subset_index_filename":"PATH_TO_DATA/02_clusteredPoints/index5.npy"

>        }

>    ],

>    "parameters":[

>        {

>            "ncluster":5,

>            "num_known_centroid":0,

>            "known_centroid_name_list": [],

>            "known_centroid_distmatrix_list": [],

>            "known_centroid_index_list": [],

>            "criteria":0.001

>        }

>    ]

>}

Set PATH_TO_DATA to the correct absolute path name.

We then rename folders:
>mv centroidList 03_centroidList

>mv clusteredPoints 03_clusteredPoints

### (iv). Calculate CCR on the validation set
In the bash terminal, type:
>python3 ./code/Learn.py 04.json > 04.log

04.json:
>{

>    "dataset":[

>        {
>            "name_filename":"PATH_TO_DATA/md1/dataSet_name.npy",

>            "distmatrix_filename":"PATH_TO_DATA/md1/dataSet_dist.npy",

>            "index_filename":"PATH_TO_DATA/md1/dataSet_count.npy",

>            "subset_index_filename":"PATH_TO_DATA/md1/val_set_index.npy"

>        }

>    ],

>    "parameters":[

>        {

>            "ncluster":10,

>            "num_known_centroid":10,

>            "known_centroid_name_list": ["PATH_TO_DATA/01_centroidList/name0.npy", "PATH_TO_DATA/01_centroidList/name1.npy", "PATH_TO_DATA/01_centroidList/name2.npy", "PATH_TO_DATA/01_centroidList/name3.npy", "PATH_TO_DATA/01_centroidList/name4.npy","PATH_TO_DATA/03_centroidList/name0.npy", "PATH_TO_DATA/03_centroidList/name1.npy", "PATH_TO_DATA/03_centroidList/name2.npy", "PATH_TO_DATA/03_centroidList/name3.npy", "PATH_TO_DATA/03_centroidList/name4.npy"],

>            "known_centroid_distmatrix_list": ["PATH_TO_DATA/01_centroidList/dist0.npy", "PATH_TO_DATA/01_centroidList/dist1.npy", "PATH_TO_DATA/01_centroidList/dist2.npy", "PATH_TO_DATA/01_centroidList/dist3.npy", "PATH_TO_DATA/01_centroidList/dist4.npy","PATH_TO_DATA/03_centroidList/dist0.npy", "PATH_TO_DATA/03_centroidList/dist1.npy", "PATH_TO_DATA/03_centroidList/dist2.npy", "PATH_TO_DATA/03_centroidList/dist3.npy", "PATH_TO_DATA/03_centroidList/dist4.npy"],

>            "known_centroid_index_list": ["PATH_TO_DATA/01_centroidList/index0.npy", "PATH_TO_DATA/01_centroidList/index1.npy", "PATH_TO_DATA/01_centroidList/index2.npy", "PATH_TO_DATA/01_centroidList/index3.npy", "PATH_TO_DATA/01_centroidList/index4.npy","PATH_TO_DATA/03_centroidList/index0.npy", "PATH_TO_DATA/03_centroidList/index1.npy", "PATH_TO_DATA/03_centroidList/index2.npy", "PATH_TO_DATA/03_centroidList/index3.npy", "PATH_TO_DATA/03_centroidList/index4.npy"],

>            "criteria":0.001

>        }

>    ]

>}

Set PATH_TO_DATA to the correct absolute path name.

We then rename folders:
>mv centroidList 04_centroidList

>mv clusteredPoints 04_clusteredPoints

In 04.log, we can find:
> ----> No.1 iterations

> cluster assign results:

> flag= 0  datasize= 3965

> flag= 1  datasize= 2832

> flag= 2  datasize= 5372

> flag= 3  datasize= 2091

> flag= 4  datasize= 5047

> flag= 5  datasize= 99

> flag= 6  datasize= 289

> flag= 7  datasize= 137

> flag= 8  datasize= 137

> flag= 9  datasize= 31

CCR on the validation set of md1 equals (3965+2832+5372+2091+5047)/(3965+2832+5372+2091+5047+99+298+137+137+31)=0.964. The actual values may vary on your computer.


### (v). Calculate entropy on the validation set (optional)
In the bash terminal, type:
>python3 ./code/Learn.py 05.json > 05.log

05.json:
>{

>    "dataset":[

>        {
>            "name_filename":"PATH_TO_DATA/md2/dataSet_name.npy",

>            "distmatrix_filename":"PATH_TO_DATA/md2/dataSet_dist.npy",

>            "index_filename":"PATH_TO_DATA/md2/dataSet_count.npy",

>            "subset_index_filename":"PATH_TO_DATA/md2/val_set_index.npy"

>        }

>    ],

>    "parameters":[

>        {

>            "ncluster":10,

>            "num_known_centroid":10,

>            "known_centroid_name_list": ["PATH_TO_DATA/01_centroidList/name0.npy", "PATH_TO_DATA/01_centroidList/name1.npy", "PATH_TO_DATA/01_centroidList/name2.npy", "PATH_TO_DATA/01_centroidList/name3.npy", "PATH_TO_DATA/01_centroidList/name4.npy","PATH_TO_DATA/03_centroidList/name0.npy", "PATH_TO_DATA/03_centroidList/name1.npy", "PATH_TO_DATA/03_centroidList/name2.npy", "PATH_TO_DATA/03_centroidList/name3.npy", "PATH_TO_DATA/03_centroidList/name4.npy"],

>            "known_centroid_distmatrix_list": ["PATH_TO_DATA/01_centroidList/dist0.npy", "PATH_TO_DATA/01_centroidList/dist1.npy", "PATH_TO_DATA/01_centroidList/dist2.npy", "PATH_TO_DATA/01_centroidList/dist3.npy", "PATH_TO_DATA/01_centroidList/dist4.npy","PATH_TO_DATA/03_centroidList/dist0.npy", "PATH_TO_DATA/03_centroidList/dist1.npy", "PATH_TO_DATA/03_centroidList/dist2.npy", "PATH_TO_DATA/03_centroidList/dist3.npy", "PATH_TO_DATA/03_centroidList/dist4.npy"],

>            "known_centroid_index_list": ["PATH_TO_DATA/01_centroidList/index0.npy", "PATH_TO_DATA/01_centroidList/index1.npy", "PATH_TO_DATA/01_centroidList/index2.npy", "PATH_TO_DATA/01_centroidList/index3.npy", "PATH_TO_DATA/01_centroidList/index4.npy","PATH_TO_DATA/03_centroidList/index0.npy", "PATH_TO_DATA/03_centroidList/index1.npy", "PATH_TO_DATA/03_centroidList/index2.npy", "PATH_TO_DATA/03_centroidList/index3.npy", "PATH_TO_DATA/03_centroidList/index4.npy"],

>            "criteria":0.001,

>            "eval_entropy":1

>        }

>    ]

>}

Set PATH_TO_DATA to the correct absolute path name.

In 05.log, we can find the following output:

average norm entropy:  0.03679774792535088

We then rename folders:
>mv centroidList 05_centroidList

>mv clusteredPoints 05_clusteredPoints


### (vi). Calculate CCR on the test set
The operations are similar to step 4 but are done on the test set. In the json input, set:

>            "name_filename":"PATH_TO_DATA/md1/dataSet_name.npy",

>            "distmatrix_filename":"PATH_TO_DATA/md1/dataSet_dist.npy",

>            "index_filename":"PATH_TO_DATA/md1/dataSet_count.npy",

>            "subset_index_filename":"PATH_TO_DATA/md1/test_set_index.npy"

Set PATH_TO_DATA to the correct absolute path name.

## Use the trained model to classify the target dataset

Assuming that md2 is our target dataset.

In the bash terminal, type:
>python3 ./code/Learn.py 07.json > 07.log

07.json:
>{

>    "dataset":[

>        {
>            "name_filename":"PATH_TO_DATA/md2/dataSet_name.npy",

>            "distmatrix_filename":"PATH_TO_DATA/md2/dataSet_dist.npy",

>            "index_filename":"PATH_TO_DATA/md2/dataSet_count.npy",

>            ""

>        }

>    ],

>    "parameters":[

>        {

>            "ncluster":10,

>            "num_known_centroid":10,

>            "known_centroid_name_list": ["PATH_TO_DATA/01_centroidList/name0.npy", "PATH_TO_DATA/01_centroidList/name1.npy", "PATH_TO_DATA/01_centroidList/name2.npy", "PATH_TO_DATA/01_centroidList/name3.npy", "PATH_TO_DATA/01_centroidList/name4.npy","PATH_TO_DATA/03_centroidList/name0.npy", "PATH_TO_DATA/03_centroidList/name1.npy", "PATH_TO_DATA/03_centroidList/name2.npy", "PATH_TO_DATA/03_centroidList/name3.npy", "PATH_TO_DATA/03_centroidList/name4.npy"],

>            "known_centroid_distmatrix_list": ["PATH_TO_DATA/01_centroidList/dist0.npy", "PATH_TO_DATA/01_centroidList/dist1.npy", "PATH_TO_DATA/01_centroidList/dist2.npy", "PATH_TO_DATA/01_centroidList/dist3.npy", "PATH_TO_DATA/01_centroidList/dist4.npy","PATH_TO_DATA/03_centroidList/dist0.npy", "PATH_TO_DATA/03_centroidList/dist1.npy", "PATH_TO_DATA/03_centroidList/dist2.npy", "PATH_TO_DATA/03_centroidList/dist3.npy", "PATH_TO_DATA/03_centroidList/dist4.npy"],

>            "known_centroid_index_list": ["PATH_TO_DATA/01_centroidList/index0.npy", "PATH_TO_DATA/01_centroidList/index1.npy", "PATH_TO_DATA/01_centroidList/index2.npy", "PATH_TO_DATA/01_centroidList/index3.npy", "PATH_TO_DATA/01_centroidList/index4.npy","PATH_TO_DATA/03_centroidList/index0.npy", "PATH_TO_DATA/03_centroidList/index1.npy", "PATH_TO_DATA/03_centroidList/index2.npy", "PATH_TO_DATA/03_centroidList/index3.npy", "PATH_TO_DATA/03_centroidList/index4.npy"],

>            "criteria":0.001

>        }

>    ]

>}

Set PATH_TO_DATA to the correct absolute path name.

We then rename folders:
>mv centroidList 06_centroidList

>mv clusteredPoints 06_clusteredPoints

In 07.log, we can find:

> flag= 0  datasize= 19412

> flag= 1  datasize= 18437

> flag= 2  datasize= 19812

> flag= 3  datasize= 18307

> flag= 4  datasize= 21485

> flag= 5  datasize= 18868

> flag= 6  datasize= 22284

> flag= 7  datasize= 20623

> flag= 8  datasize= 20424

> flag= 9  datasize= 20348

Clusters 0-4 correspond to the first context and 5-9 correspond to the second context. We can name these contexts to be labels A1 and A2 and calcualte their ratios in the target system to be: (19412+18437+19812+18307+21485)/(19412+18437+19812+18307+21485+18868+22284+20623+20424+20348)=0.487 and 1-0.487=0.513.

07_clusteredPoints/indexX.npy contains the classified results in each cluster. In this case, index[0-4].npy corresponds to A1, while index[5-9].npy corresponds to A2. We can check the numbers in each context and visualize the original MD data for a comparison. We note that, based on the way the dataset was constructed by the user, these integers can be mapped back to the frame number and residue number. For example, if in our MD system we have modeled 100 Li+ ions, an index of 150 corresponds to the 50th Li+ at the second frame.

For multiple tasks, each dataset can be assigned a combination of labels. For example, it can be (A1,B1,C2). We can then evaluate the ratios of different label combinations and check their corresponding structures in the target system for an analysis.

## Notes
- In the above examples, we have used k1=5 and k2=5. In actural cases, these need to be tuned to get optimal CCR.

- For three or more datasets, steps (ii)-(v) need to be repeated.

## Example files for training the model on a computer cluster with slurm

### Input files

> 01.json

> 02.json

> 03.json

> 04.json

> 05.json

> 06.json

> 07.json

> hidiscover-01.slurm

> hidiscover-02.slurm

> hidiscover-03.slurm

> hidiscover-04.slurm

> hidiscover-05.slurm

> hidiscover-06.slurm

> hidiscover-07.slurm

### Output files

> 01.log

> 01_centroidList

> 01_clusteredPoints

> 02.log

> 02_centroidList

> 02_clusteredPoints

> 03.log

> 03_centroidList

> 03_clusteredPoints

> 04.log

> 04_centroidList

> 04_clusteredPoints

> 05.log

> 05_centroidList

> 05_clusteredPoints

> 06.log

> 06_centroidList

> 06_clusteredPoints

> 07.log

> 07_centroidList

> 07_clusteredPoints


## Complete input files for training the model in task A on a computer cluster with slurm

In this example, we have six datasets and the last one is also the target system. We recommend do the model training on a computer cluster.

The json files required for training, validation, and testing are as follows:

> 01.json

> 01-validation.json

> 01-test.json

> 02-1.json

> 02-2.json

> 02-validation.json

> 02-test.json

> 03-1.json

> 03-2.json

> 03-validation.json

> 03-test.json

> 04-1.json

> 04-2.json

> 04-validation.json

> 04-test.json

> 05-1.json

> 05-2.json

> 05-validation.json

> 05-test.json

> 06-1.json

> 06-2.json

> 06-classification.json

The slurm files need to be executed in order.

Partition the dataset:

> hidiscover-00-partition.slurm

Train and validate the models:

> hidiscover-01-train.slurm

> hidiscover-02-1-train.slurm

> hidiscover-02-2-train.slurm

> hidiscover-01-validation.slurm

> hidiscover-03-1-train.slurm

> hidiscover-03-2-train.slurm

> hidiscover-02-validation.slurm

> hidiscover-04-1-train.slurm

> hidiscover-04-2-train.slurm

> hidiscover-03-validation.slurm

> hidiscover-05-1-train.slurm

> hidiscover-05-2-train.slurm

> hidiscover-04-validation.slurm

> hidiscover-06-1-train.slurm

> hidiscover-06-2-train.slurm

> hidiscover-05-validation.slurm

Test the model:

> hidiscover-01-test.slurm

> hidiscover-02-test.slurm

> hidiscover-03-test.slurm

> hidiscover-04-test.slurm

> hidiscover-05-test.slurm

Apply the model to predict the labels in the target system:

> hidiscover-06-classification.slurm

