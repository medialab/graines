
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Import a dataframe "data" with 2 columns text and cluster
def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20, cluster_column = 'cluster_number'):
    words = count.get_feature_names()
    labels = list(docs_per_topic[cluster_column])
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df, cluster_column = 'cluster_number'):
    topic_sizes = (df.groupby([cluster_column])
                     .text
                     .count()
                     .reset_index()
                     .rename({"text": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def topic_modeling(data, cluster_column = 'cluster_number', text_column = 'text', ngram_range = (1,1), top_n = 20):
    '''Group by cluster_number_column and aggregate the text_column'''

    docs_per_topic = data.groupby([cluster_column], as_index = False).agg({text_column: ' '.join}) # group data by label
    tf_idf, count = c_tf_idf(docs_per_topic[text_column].values, m=len(data), ngram_range=ngram_range) # get the ctfidf-results
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=top_n) # Get the top n words
    topic_sizes = extract_topic_sizes(data) # get the topic size ( number if words)

    keys = list(top_n_words.keys())
    new_dict = {}

    for key in keys:
        items = []
        for item in top_n_words[key]:
            items.append(item[0])

        new_dict[key] = items
      
    final = pd.DataFrame(new_dict)
    return final

if __name__ == "__main__":
    data = pd.read_csv('/Users/charlesdedampierre/Desktop/new_node2vec/project/sentence_embedding/final.csv', index_col=[0])
    final = topic_modeling(data)
    final.to_csv('/Users/charlesdedampierre/Desktop/new_node2vec/project/sentence_embedding/top_topics.csv')




