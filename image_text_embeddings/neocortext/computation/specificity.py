import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2

def specifity_feature(X, y, X_names, p_value_limit = 0.99, top_n = 100):

	'''
	exemple:
		
	être  prendre  donner avoir etc cluster
	1       O        1      0    0    1

	2       2        0       1   1    2

	X_names = ['être', 'prendre', 'donner', 'avoir', 'etc']
	X = ([1,0,1,0,0], [2,2,0,1,1])
	y = [1, 2]

	'''

	p_value_limit = p_value_limit
	dtf_features = pd.DataFrame()
	count=0

	for cat in np.unique(y):
		chi2_results, p = chi2(X, y==cat)
		dtf_features = dtf_features.append(pd.DataFrame({"feature":X_names, "score":1-p, "y":cat}))
		dtf_features = dtf_features.sort_values(["y","score"], ascending=[True,False])
		dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
		
	new = pd.DataFrame()
	for cluster_number in list(set(dtf_features.y)):
		beep = dtf_features[dtf_features.y == cluster_number].head(top_n)
		features = list(beep.feature)
		new[cluster_number] = features

	new.to_csv('specificity_measure.csv')
