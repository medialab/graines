import pandas as pd

def preprocessing_edge_list(edge_list, weight_edge_column):

	if weight_edge_column == []:
	    edge_list['weight'] = 1
	else:
	 	edge_list['weight'] = edge_list[weight_edge_column]

	# groupby weight
	edge_list = edge_list.groupby(['source', 'target', 'target_name'])['weight'].sum().rename('weight').sort_values(ascending=False).reset_index()
	edge_list['norm_weight'] = edge_list['weight'] / edge_list.groupby('target_name')['weight'].transform('sum')

	return edge_list

def preprocessing_node_list(node_list, size_column):

	if size_column == []:
		node_list['size_bins'] = 20
	else:
		node_list['size_bins'] = node_list.groupby("entity_name")[size_column].apply(lambda x: pd.qcut(x.rank(method='first'), 4, labels = [15, 20, 25, 30]))
		node_list['size_bins'] = node_list['size_bins'].astype(int)

	return node_list
