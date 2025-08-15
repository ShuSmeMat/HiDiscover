
"""
Authors: Hanyin Zhang, Haoyuan Li
Date created: 2022
Description: The hidiscover continual learn function
"""

from modCustomKmeans import *
from modFileIO import *

def contLearn(dataset_dict,parameter_dict,kernel=semiKmeansClusterKernel):

    np.random.seed(1)
    
    #------------------------------loading dataset------------------------------
    dataset = load_dataset(dataset_dict)

    # select dataset
    subset_index_filename = dataset_dict['subset_index_filename']
    if subset_index_filename == "":
        print("Full dataset will be used.")
        print("Loaded %d datas"%len(dataset))
    else:
        subset_index = np.load(subset_index_filename)
        dataset = select_load(dataset,subset_index)
        print("Loaded %d datas"%len(dataset))
    
    #--------------------------------clustering---------------------------------
    centroidList,clusterDict = kernel(dataset,parameter_dict)

    #--------------------------------save data----------------------------------
    save_centroidList_as_npy(centroidList)
    save_cluster_member_as_npy(clusterDict)

    print("\n Job done!")
