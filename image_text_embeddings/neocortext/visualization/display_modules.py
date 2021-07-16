
import numpy as np
from bokeh.models import (
    Slider,
    CheckboxButtonGroup,
    Select,
    TextInput,
    CheckboxButtonGroup,
    Button,
    CustomJS,
    Div
)
import codecs

def modules(data, variables):

    # Display the description of the tool
    #file = codecs.open(".description.html", "r", "utf-8")
    #desc = Div(text=open(".neocortext/visualization/templates/description.html").read())
    #desc = Div(text=file.read())
    desc = Div(text="")


    # Chose entity type
    entity_div = Div(
        text="Chose the entities to be displayed",
        style={"font-size": "100%", "color": "black"},
    )

    # first chekbox for entities
    entities = list(set(data.entity_name))
    checkbox_group = CheckboxButtonGroup(labels=entities, 
        active=list(np.arange(len(entities))), 
        button_type="success", 
        sizing_mode = "scale_width")
    
    checkbox_group.js_on_click(
        CustomJS(
            code="""console.log('checkbox_group: active=' + this.active, this.toString())"""
        )
    )

    # Chose Cluster Type
    cluster_types = list(set(data.cluster_type))
    cluster_types_index = cluster_types.index('hdbscan')

    label_drop_cluster_type = Select(
        title="Chose the clustering method ", options=cluster_types, value=cluster_types[cluster_types_index]
    )

    # Chose Cluster Type
    projection_types = variables['independent_variables']
    cluster_number_index = projection_types.index('cluster_number')

    label_drop_projection_type = Select(
        title="Project variable ", options=projection_types, value=projection_types[cluster_number_index]
    )
    

    # Chose the node sizes
    select_size = Slider(
        title="filter by the entity size",
        value=15,
        start=15,
        end=30,
        step=5,
    )

    size_text_slider = Slider(
        title="filter the text size",
        value =15,
        start=0,
        end=19,
        step=3,
    )

    # Chose to displya a label or not
    checkbox_options = ["yes", "no"]
    label_drop = Select(
        title="Show labels? ", options=checkbox_options, value=checkbox_options[1]
    )

    # Chose the Dimension ReductionAlgorithm
    reduction = list(set(data.reduction_type))
    tsne_index = reduction.index('tsne')
    reduc_drop = Select(title="Reduction Algorithm", options=reduction, value=reduction[tsne_index])

    # Chose a node by its name
    name_input = TextInput(value="", title="Name contains")

    # Download Data
    button_download = Button(label="Download Data", button_type="success")

    button_co_occurence = Button(label="Co_occurence_network", button_type="success")

    # Chose the co-occurence projection
    targets = variables['targets']
    #co_occ_projection = CheckboxButtonGroup(labels=targets, active=[], sizing_mode="scale_width")

    co_occ_list = ['None'] + targets
    co_occ_projection = Select(title="Projection of co-occurence networks", options=co_occ_list, value=co_occ_list[0])

    return (desc, 
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
        size_text_slider)
