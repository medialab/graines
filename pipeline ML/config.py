#seeds used in _triangular_classifier.py and _bayesian.py
seeds = [1, 2, 3, 4, 5, 6]

#chose one or several embeddings from ["bert", "images", "features", "topology", "bayesian", "tfidf", doc2vec]
# If several models are selected, the vectors will be concatenated.
type_of_model = ["bert", "bayesian", "topology"]
type_of_model = ["bayesian"]
#type_of_model= ["bert", "images", "features", "topology", "bayesian", "tfidf", 'doc2vec']

#chose one objective from ["report", "classification"]
objective = "classification"
prevalence=39/1796