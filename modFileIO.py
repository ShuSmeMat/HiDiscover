
"""
Authors: Hanyin Zhang, Haoyuan Li
Date created: 2022
Description: files I/O 
"""

import numpy as np
import os

def load_dataset(dataset_dict):
    
    name_npy = np.load(dataset_dict['name_filename'])
    distmatrix_npy = np.load(dataset_dict['distmatrix_filename'])
    index_npy = np.load(dataset_dict['index_filename'])

    dataset = name_dist_index_to_dataset(name_npy,distmatrix_npy,index_npy)

    return dataset

def select_load(dataset,linenumber_list):

    subset = []
    for number in linenumber_list:
        subset.append(dataset[number-1])

    return subset

def name_dist_index_to_dataset(name,distmatrix,index):
    
    dataset=[]
    for i in range(name.shape[0]):
        dataset.append([name[i],distmatrix[i,:,:],index[i]])
    
    return dataset

def dataset_to_name_dist_index(dataset):

    name_list = []
    distmatrix_list = []
    index_list = []

    for data in dataset:
        name_list.append(data[0])
        distmatrix_list.append(data[1])
        index_list.append(data[2])

    name_npy = np.array(name_list)
    distmatrix_npy = np.array(distmatrix_list)
    index_npy = np.array(index_list, dtype='int64')

    return name_npy, distmatrix_npy, index_npy

def save_centroidList_as_npy(centroidList):

    for i in range(len(centroidList)):
        centroid = centroidList[i]
        name_npy, distmatrix_npy, index_npy = dataset_to_name_dist_index([centroid])

        if not (os.path.exists('./centroidList')):
            os.mkdir('./centroidList')    

        name_filename = './centroidList/name'+str(i)+'.npy'
        distmatrix_filename = './centroidList/dist'+str(i)+'.npy'
        index_filename = './centroidList/index'+str(i)+'.npy'

        np.save(name_filename, name_npy)
        np.save(distmatrix_filename, distmatrix_npy)
        np.save(index_filename, index_npy)


def save_cluster_member_as_npy(clusterDict):

    for i in sorted(clusterDict.keys()):
        clusteri = clusterDict[i]
        name_npy, distmatrix_npy, index_npy = dataset_to_name_dist_index(clusteri)

        if not (os.path.exists('./clusteredPoints')):
            os.mkdir('./clusteredPoints')    

        index_filename = './clusteredPoints/index'+str(i)+'.npy'
        np.save(index_filename, index_npy)

