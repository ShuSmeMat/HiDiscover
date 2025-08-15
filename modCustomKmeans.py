
"""
Authors: Hanyin Zhang, Haoyuan Li
Date created: 2022
Description: The custom kmeans function
"""

import numpy as np
import sys
import copy
from modFileIO import *
import random

def getRandomInitCentroidList(dataSet,ncluster):
    
    centroidIndex = []
    randomCentroidList = []

    i = 0
    while i < ncluster:

        index = np.random.randint(len(dataSet))

        if index not in centroidIndex:
            centroidIndex.append(index)
            randomCentroidList.append(copy.deepcopy(dataSet[index]))
            i += 1
    print('Initialize %d RandomCentroid'%(i))

    return randomCentroidList


def loadCentroidList(CentroidList,parameter_dict):

    known_k_num = parameter_dict['num_known_centroid']
    print('Aim to load %d centroid\n'%(known_k_num))

    loaded_centroid = 0

    known_name_list = parameter_dict['known_centroid_name_list']
    known_distmatrix_list = parameter_dict['known_centroid_distmatrix_list']
    known_index_list = parameter_dict['known_centroid_index_list']

    if known_k_num == len(known_name_list) == len(known_distmatrix_list) == len(known_index_list):
        for i in range(len(known_name_list)):
            print("---> Loading No.%d centroid\n"%(i))
            name_npy = np.load(known_name_list[i])
            distmatrix_npy = np.load(known_distmatrix_list[i])
            index_npy = np.load(known_index_list[i])
            
            if len(name_npy[0]) != len(CentroidList[loaded_centroid][1]):
                print("length of loading KnownCentroid name = ", len(name_npy[0]))
                print("length of RandomCentroid name = ", len(CentroidList[loaded_centroid][1]))
                print("Loading KnownCentroid name: \n",name_npy)
                print("RandomCentroid name: \n", CentroidList[loaded_centroid][1])
                print("Loading KnownCentroid not match RandomCentroid!")
                sys.exit()

            known_Centroid = name_dist_index_to_dataset(name_npy,distmatrix_npy,index_npy)
            CentroidList[loaded_centroid] = known_Centroid[0]

            loaded_centroid += 1
    else:
        print("Known_centroids input number error!")
        sys.exit()
    
    if loaded_centroid == known_k_num:
        if loaded_centroid > 0:
            loadedCentroidList = copy.deepcopy(CentroidList[:loaded_centroid])
            print('numCentroidLoaded=',loaded_centroid)
        else:
            loadedCentroidList = []
    else:
        print('Not enough Centroids has been loaded!')
        print('loaded_centroid=',loaded_centroid)
        print('known_k_num=',known_k_num)
        sys.exit()

    return CentroidList, loadedCentroidList


def assignToCluster(dataSet,centroidList):

    clusterDict = {}

    for i in range(len(dataSet)):
        flag = 0
        min_dis = float("inf")

        for j in range(len(centroidList)):         
            centroid = centroidList[j]
            distance = calcuDistance(dataSet[i],centroid)  
            if min_dis > distance:
                flag = j
                min_dis = distance

        if flag not in clusterDict.keys():
            clusterDict[flag] = list()

        clusterDict[flag].append(dataSet[i])

    print("cluster assign results:")

    for key in range(len(centroidList)):
        if key in clusterDict:
            print('flag=',key,' datasize=',len(clusterDict[key]))       
        else:
            print('flag=',key,' datasize=',0)       

    return clusterDict

def calcuDistance(data_record_1,data_record_2):

    name_1,distmatrix_1,index_1 = data_record_1
    name_2,distmatrix_2,index_2 = data_record_2

    distance = np.sqrt(np.sum(np.square(distmatrix_1-distmatrix_2)))  

    return distance

def calcuDistance2(data_record_1,data_record_2):

    name_1,distmatrix_1,index_1 = data_record_1
    name_2,distmatrix_2,index_2 = data_record_2

    distance = np.sum(np.square(distmatrix_1-distmatrix_2))

    return distance


def calVariance(clusterDict,centroidList):

    Variance = 0.0

    for i in sorted(clusterDict.keys()):

        centroid = centroidList[i]

        for data in clusterDict[i]: 
            
            distance2 = calcuDistance2(centroid, data)
            Variance += distance2
    
    return Variance

def semiCalCentroids(clusterDict,knownCentroidList,parameter_dict):

    centroidList = []
    ncluster = parameter_dict['ncluster']
    known_k = parameter_dict['num_known_centroid']

    for centroid in knownCentroidList:
        centroidList.append(centroid)
    
    if ncluster > known_k:
        for i in range(known_k,ncluster):
        
            centroid = calClusterCentroid(clusterDict[i])  
            centroidList.append(centroid) 
    
    return centroidList

def calClusterCentroid(cluster):

    init = False

    for name,distmatrix,index in cluster:    
        if init:
            if list(name) != list(centroid_name):
                print("error, cluster name different")
                sys.exit()
            centroid_distmatrix += distmatrix
        else:
            centroid_name = name
            centroid_distmatrix = distmatrix
            init = True

    centroid_distmatrix /= len(cluster)

    return [centroid_name,centroid_distmatrix,0]

def extract_symmetric_matrix(matrix):
    n = matrix.shape[0] 
    upper_tri_indices = np.triu_indices(n, k=1) 
    upper_tri_values = matrix[upper_tri_indices]

    return upper_tri_values


def evalEntropy(dataSet,clusterDict,centroidList):

    ncluster=len(centroidList)
    tot_data_size=len(dataSet)

    part1_ratio = 0.9

    clusters_part1 = {}
    clusters_part2 = {}

    for i_cluster in range(ncluster):
        clusters_part1[i_cluster] = []
        clusters_part2[i_cluster] = []

    tot_size_clusters_part1 = 0
    tot_size_clusters_part2 = 0

    for key in clusterDict:
        cluster_size = len(clusterDict[key])
        if cluster_size == 0: 
            continue

        part1_size = int(cluster_size * part1_ratio)
        part2_size = cluster_size - part1_size

        sampled_records = random.sample(clusterDict[key], part1_size + part2_size)
        part1_records = sampled_records[:part1_size]

        for record in part1_records:
            data_matrix = record[1]
            upper_tri_values = extract_symmetric_matrix(data_matrix)
            clusters_part1[key].append(upper_tri_values)
            tot_size_clusters_part1 += 1

        part2_records = sampled_records[part1_size:part1_size + part2_size]
        for record in part2_records:
            data_matrix = record[1]
            upper_tri_values = extract_symmetric_matrix(data_matrix)
            clusters_part2[key].append(upper_tri_values)
            tot_size_clusters_part2 += 1

        clusters_part1[key] = np.array(clusters_part1[key])
        clusters_part2[key] = np.array(clusters_part2[key])

    prior_ratios=[]
    for key in clusters_part1:
        if len(clusters_part1[key]) > 0 and len(clusters_part2[key]) > 0:
            prior_ratios.append(len(clusters_part1[key])/float(tot_size_clusters_part1))
    prior_ratios=np.array(prior_ratios)

    cluster_centers_used=[]
    for i in range(ncluster):
        if len(clusters_part1[i])>0 and len(clusters_part2[i])>0:
            centroid_matrix=centroidList[i][1]
            upper_tri_values = extract_symmetric_matrix(centroid_matrix)
            cluster_centers_used.append(upper_tri_values)
    ncluster_used=len(cluster_centers_used)

    clusters_sigma = {} 
    cluster_ids=[]
    for i_cluster in range(ncluster):
        if len(clusters_part1[i_cluster]) > 0 and len(clusters_part2[i_cluster]) > 0:
            cluster_ids.append(i_cluster)
            this_cluster = clusters_part1[i_cluster]
            sigma = np.cov(this_cluster, rowvar=False)
            clusters_sigma[i_cluster] = sigma
    identity_covariance = [clusters_sigma[i] for i in cluster_ids]

    dict_sampled_dataset = {}
    for key in clusters_part2:
        if len(clusters_part1[key]) > 0 and len(clusters_part2[key]) > 0:
            dict_sampled_dataset[key] = clusters_part2[key]

    from scipy.stats import multivariate_normal
    entropy_list=[]
    for key in dict_sampled_dataset:

        likelihoods = np.array( [[multivariate_normal.pdf(x, mean=mu, cov=cov) for mu, cov in zip(cluster_centers_used, identity_covariance) ] for x in dict_sampled_dataset[key] ])

        marginal_probs = np.sum(likelihoods * prior_ratios, axis=1, keepdims=True)

        posteriors = (likelihoods * prior_ratios) / marginal_probs  

        for i_data in range(len(dict_sampled_dataset[key])):
            entropy=0.0
            for i_centroid in range(ncluster_used):
                if posteriors[i_data,i_centroid]>0.0:
                    entropy-=posteriors[i_data,i_centroid]*np.log(posteriors[i_data,i_centroid])
            entropy_list.append(entropy)

    entropy_list=np.array(entropy_list)
    print()
    print("average norm entropy: ",np.mean(entropy_list)/np.log(ncluster))

def semiKmeansClusterKernel(dataSet,parameter_dict):

    ncluster = parameter_dict['ncluster']
    criteria = parameter_dict['criteria']

    randomCentroidList = getRandomInitCentroidList(dataSet,ncluster)

    centroidList,KnownCentroidList = loadCentroidList(randomCentroidList,parameter_dict)

    clusterDict = assignToCluster(dataSet,centroidList)

    newVar = calVariance(clusterDict, centroidList)
    oldVar = -1 

    print("initial variance",newVar)

    i = 1
    while abs(newVar - oldVar) >= criteria: 
        print('\n----> No.%d iterations'%i)  

        centroidList = semiCalCentroids(clusterDict,KnownCentroidList,parameter_dict)  
        clusterDict = assignToCluster(dataSet, centroidList)  

        oldVar = newVar
        newVar = calVariance(clusterDict, centroidList)  
        print('Variance=',newVar)  

        i += 1

    if "eval_entropy" in parameter_dict:
        if parameter_dict["eval_entropy"] == 1:
            pass
            evalEntropy(dataSet,clusterDict,centroidList)

    return centroidList, clusterDict
