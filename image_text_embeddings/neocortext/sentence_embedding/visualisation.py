
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.plotting import figure, show
from bokeh.models import CategoricalColorMapper, ColumnDataSource
import colorcet as cc
import pandas as pd

def color_mapper_function(data, projection):
    # data
    # projection is the column name

    full_list = data[projection]
    full_list = full_list.astype(str)

    feature_colors = list(set(full_list))
    feature_colors = list(map(str, feature_colors))

    palette = cc.glasbey[: len(feature_colors)]

    # Check for the cluster -1 of hbdscan
    if "-1" in feature_colors:
        np_cluster = feature_colors.index('-1')
        palette[np_cluster] = '#BDBDBD'

    color_mapper = CategoricalColorMapper(palette=palette, factors=feature_colors)
    return color_mapper

def visualisation(data):

	labels = 'cluster_number'
	data[labels] = data[labels].astype(str)

	color_mapper = color_mapper_function(data, labels)

	source = ColumnDataSource(data=data)

	TOOLTIPS = [
	    ("cluster_number", "@cluster_number"),
	    ("text", "@text")
	]

	p = figure(
	    plot_width=1000,
	    plot_height=800,
	    toolbar_location="below",
	    tooltips=TOOLTIPS,
	    title = 'Text Embeddings'
	)

	p.title.text_font_size = '20pt'

	p.scatter('dim_1', 
	          'dim_2', 
	          source=source, size=10,
	          fill_color={'field': labels, 'transform': color_mapper})

	output_file("sentence_embeddings.html")
	show(p)

if __name__ == "__main__":
	data = pd.read_csv('/Users/charlesdedampierre/Desktop/new_node2vec/project/sentence_embedding/final.csv', index_col=[0])
	visualisation(data, '/Users/charlesdedampierre/Desktop/new_node2vec/project/sentence_embedding/')

