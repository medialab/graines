from .utils import (co_coccurence_target, compute_coordinate_cooc_network)
import pickle
import pandas as pd
import os

def co_occurence_compute(path, global_filter, n_neighbours, edge_list, embeddings):

    if not os.path.isdir(path + 'co_occurence_edge_list/'):
    	os.mkdir(path + 'co_occurence_edge_list/')

    print ('Computing co_occurence networks...')
    dict_cooc = {}
    for variable in set(edge_list.target_name):
        beep = edge_list[edge_list.target_name == variable]
        cooc = co_coccurence_target(beep)
        dict_cooc[variable] = cooc

        with open(path + 'co_occurence_edge_list/' + 'co_occurence_edge_list.pkl', "wb") as f:
            pickle.dump(dict_cooc, f)


    if not os.path.isdir(path + 'co_occurence_coordinate/'):
    	os.mkdir(path + 'co_occurence_coordinate/')

    print ('Computing coordinates of the cooccurence network...')

    full_dict= {}
    for variable, edge_list in dict_cooc.items():
        dict_co_occ_network = compute_coordinate_cooc_network(path, embeddings, edge_list, global_filter = global_filter, n_neighbours = n_neighbours)
        full_dict[variable] = dict_co_occ_network

    with open(path + 'co_occurence_coordinate.pkl', "wb") as f:
    	pickle.dump(full_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

