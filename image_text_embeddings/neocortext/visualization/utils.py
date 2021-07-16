
from bokeh.models import CategoricalColorMapper
import colorcet as cc


def tools_name(node_list):

    def add_at(x):
        return '@' + str(x)

    column_name = list(node_list.columns)


    column_name.remove('cluster_type')
    column_name.remove('dim_1')
    column_name.remove('dim_2')
    column_name.remove('reduction_type')
    try:
        column_name.remove('size')
        column_name.remove('norm_size')
    except:
        pass

    column_name_bokeh = list(map(add_at, column_name))
    tools = list(zip(column_name, column_name_bokeh))


    return tools

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



def adjust_graph_frame(data, reduction):

   # Change the range
    min_x = min(data[data.reduction_type == reduction]["dim_1"])
    max_x = max(data[data.reduction_type == reduction]["dim_1"])

    min_y = min(data[data.reduction_type == reduction]["dim_2"])
    max_y = max(data[data.reduction_type == reduction]["dim_2"])

    def padding_space_max(element, size=1/4):
        if element > 0:
            new = element + size*element
        else:
            new = element - size*element
        return new

    def padding_space_min(element, size=1/4):
        if element > 0:
            new = element - size*element
        else:
            new = element + size*element
        return new

    min_x = padding_space_min(min_x)
    max_x = padding_space_max(max_x)
    min_y = padding_space_min(min_y)
    max_y = padding_space_max(max_y)

    return min_x, max_x, min_y, max_y





