
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def co_coccurence_target(edge_list):
    '''source is the pivot variable'''
    cooc_edge_list = pd.merge(edge_list, edge_list, on = ['target_name', 'source'])
    cooc_edge_list['product'] = cooc_edge_list.norm_weight_x*cooc_edge_list.norm_weight_y
    final = cooc_edge_list.groupby(['target_x', 'target_y'])['product'].sum().reset_index()
    final.columns = ['source', 'target', 'weight'] 

    return final

def co_coccurence_source(edge_list):
    '''source is the pivot variable'''
    cooc_edge_list = pd.merge(edge_list, edge_list, on = ['target_name', 'target'])
    cooc_edge_list['product'] = cooc_edge_list.norm_weight_x*cooc_edge_list.norm_weight_y
    final = cooc_edge_list.groupby(['source_x', 'source_y'])['product'].sum().reset_index()
    final.columns = ['source', 'target', 'weight'] 

    return final

def from_cooc_to_edge_similarity(cooc_df, global_filter =  0.7, n_neighbours = 6):
    '''transform an edge_list with weights into an edge_list 
    whose weights are cosinues similarity with two filters'''
    pivot = cooc_df.pivot('source', 'target', 'weight')
    pivot = pivot.fillna(0)
    similarity = cosine_similarity(pivot) # compute cosine similarity
    df_sim = pd.DataFrame(similarity, index=pivot.index, columns=pivot.columns)
    df_sim = df_sim[(df_sim >= global_filter)]
    df_sim['nodes'] = df_sim.index

    res_g = pd.melt(df_sim, id_vars=['nodes']).sort_values('nodes') #time
    res_g = res_g.dropna()
    res_g.columns = ['source', 'target', 'weight']
    res_g = res_g.sort_values('source')


    duplicates = []
    for x, y, i in zip(res_g.source, res_g.target,res_g.index):
        if x == y:
            duplicates.append(i)

    new_edge = res_g.drop(index = duplicates)
    new_edge = new_edge[new_edge.weight != 0]
    
    # Neighbours
    new_edge['rank'] = new_edge.groupby(['source'])['weight'].rank(method='first', ascending=False)
    new_edge = new_edge[new_edge['rank'] <= n_neighbours]

    return new_edge
    

def make_coordinate(filter_res, new_edge):

    ''' filter_res: name dim_1 dim_2
        avec le filtre'''
    
    coord_X = {filter_res.name.iloc[x]:filter_res['dim_1'].iloc[x] for x in range(len(filter_res))}
    coord_Y = {filter_res.name.iloc[x]:filter_res['dim_2'].iloc[x] for x in range(len(filter_res))}

    new_edge['source_coord_X_start'] = new_edge['source'].apply(lambda x: coord_X.get(x))
    new_edge['source_coord_X_last']  = new_edge['target'].apply(lambda x: coord_X.get(x))
    new_edge['source_coord_Y_start'] = new_edge['source'].apply(lambda x: coord_Y.get(x))
    new_edge['source_coord_Y_last']  = new_edge['target'].apply(lambda x: coord_Y.get(x))
    
    X = [list(x) for x in zip(new_edge['source_coord_X_start'], new_edge['source_coord_X_last'])]
    Y = [list(x) for x in zip(new_edge['source_coord_Y_start'], new_edge['source_coord_Y_last'])]

    return X,Y

def compute_coordinate_cooc_network(path, embeddings, edge_list, global_filter, n_neighbours):
    '''Get X, Y coordinate to display the co-occurence links'''
    #Compute similarity matrix and X,Y coordinates for co-occurence network
    cos_crjourn_cosinus = from_cooc_to_edge_similarity(edge_list, global_filter = global_filter, n_neighbours = n_neighbours)

    # compute the coordinate for the clusters
    filter_embeddings = embeddings.drop('cluster_type', axis=1)

    dict_co_occ_network = {}
    for reduction in set(filter_embeddings.reduction_type):
        new_emb = filter_embeddings[(filter_embeddings.reduction_type == reduction)]
        X,Y = make_coordinate(new_emb, cos_crjourn_cosinus)
        dict_co_occ_network[reduction] = (X,Y)

    return dict_co_occ_network


