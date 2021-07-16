from bokeh.transform import factor_cmap
from bokeh.layouts import row, column
from bokeh.models import Select, CheckboxGroup, CustomJS, Div
from bokeh.plotting import figure, output_file, save
from bokeh.models.glyphs import Text
from bokeh.models import (
    Range1d,
    Circle,
    ColumnDataSource,
    MultiLine,
    Slider,
    CheckboxButtonGroup,
    Select,
    TextInput,
    CheckboxButtonGroup,
    Button,
)
import bokeh.models as bmo
from bokeh.palettes import d3
from bokeh.io import output_notebook, save, curdoc
from bokeh.plotting import figure, show
from bokeh.models.ranges import FactorRange
from bokeh.models import CategoricalColorMapper, ColumnDataSource, ColorBar, LabelSet
from bokeh.palettes import RdBu
import colorcet as cc
import pandas as pd
import pickle
from os.path import dirname, join
import numpy as np
from bokeh.models import FactorRange
from bokeh.transform import factor_cmap, factor_mark
import os
from bokeh.models.widgets import FileInput
from .display_modules import modules
from .utils import tools_name, color_mapper_function, adjust_graph_frame


def node2vec_bokeh(data, variables, dict_cooc):

    '''with open(path + "co_occurence_network.pkl", "rb") as f:
        co_occurence_network = pickle.load(f)'''

    #data['node_size'] = 20
    data.cluster_number = data.cluster_number.astype(str)

    # Get the elements of the text
    (desc, 
      entity_div, 
      checkbox_group, 
      label_drop_cluster_type, 
      select_size, 
      label_drop, 
      reduc_drop, 
      name_input, 
      button_download,
      button_co_occurence,
      co_occ_projection, 
      label_drop_projection_type,
      size_text_slider) = modules(data, variables)


    initial = data[(data.cluster_type == 'hdbscan') & (data.reduction_type == 'tsne')]

    source = ColumnDataSource(data=initial)
    TOOLTIPS = tools_name(data)

    color_mapper = color_mapper_function(data, 'cluster_number')
     
    p = figure(
        plot_width=1400,
        plot_height=980,
        toolbar_location="below",
        tooltips=TOOLTIPS,
    )


    list_entities = list(set(initial.entity_name))
    markers = ["hex", "circle", "triangle", "diamond", "square"]
    markers = markers[:len(list_entities)]

    colors = {'field': label_drop_projection_type.value, 'transform': color_mapper}

    p.scatter('dim_1',
              'dim_2', 
              source=source,
              marker=factor_mark('entity_name', markers, list_entities), 
              fill_color=colors,
              size='size_bins', alpha=0.6)


    def update(attrname, old, new):

            active_entity = checkbox_group.active
            entity = [list_entities[i] for i in active_entity]
            reduction = reduc_drop.value

            input_text = name_input.value
            input_text_larger = input_text.capitalize()
            input_all_capital = input_text.upper()

            searchfor = [input_text, input_text_larger, input_all_capital]
            
            source.data = data[
            (data["reduction_type"] == reduction)
            & (data["size_bins"] >= select_size.value)
            & (data["cluster_type"] == label_drop_cluster_type.value)
            & (data["entity_name"].isin(entity))
            & (data["name"].str.contains('|'.join(searchfor), na=False))]

    def update_other(attrname, old, new):

        p = figure(
                plot_width=1400,
                plot_height=980,
                toolbar_location="below",
                tooltips=TOOLTIPS,
            )

        color_mapper = color_mapper_function(data, label_drop_projection_type.value)
        p.scatter(
            "dim_1",
            "dim_2",
            source=source,
            marker=factor_mark("entity_name", markers, list_entities),
            fill_color={"field": label_drop_projection_type.value, "transform": color_mapper},
            size="size_bins",
            alpha=0.6,
        )

        # Add text or not
        if "yes" in label_drop.value:
            font_size = size_text_slider.value
            font_size = str(font_size) + 'pt'
            p.text(
                "dim_1",
                "dim_2",
                text="name",
                text_font_size=font_size,
                source=source,
                alpha=0.6,
                x_offset=19,
            )
        else:
            pass

        if co_occ_projection.value == 'None':
            X = []
            Y = []
        else:
            coord_cooc = dict_cooc[co_occ_projection.value][reduc_drop.value]
            X = coord_cooc[0]
            Y = coord_cooc[1]
         
        p.multi_line(X,Y)

        layout_with_widgets.children[1] = p


    checkbox_group.on_change("active", update)
    select_size.on_change("value", update)
    label_drop.on_change("value", update_other)
    size_text_slider.on_change("value", update_other)
    reduc_drop.on_change("value", update)
    name_input.on_change("value", update)
    label_drop_cluster_type.on_change("value", update)
    label_drop_projection_type.on_change("value", update_other)
    co_occ_projection.on_change("value", update_other)

    layout_with_widgets = row(
            column(
                desc,
                entity_div,
                checkbox_group,
                label_drop_cluster_type,
                reduc_drop,
                label_drop,
                select_size,
                size_text_slider,
                name_input,
                co_occ_projection, 
                label_drop_projection_type
            ),
            p,
        )

    curdoc().add_root(layout_with_widgets)