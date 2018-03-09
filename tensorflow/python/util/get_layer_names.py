# function to extract layer names
def get_layer_names(graph=None):
    """ Fetches layer names of a tensorflow model graph
    Args: 
        [Optional] graph: graph of the model
    Returns:
        A list of layer names of the model.

        example:
            layers, graph = get_layer_names(graph)

    """
    if graph is None:
        graph = tf.get_default_graph()
    L = []
    for i in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        v = re.search(r'\w+\/\w+\:\d', str(i)).group(0)
        v = v.split(":")[0]
        layerName = v.split("/")[0]
        L.append(layerName)
    return sorted(list(set(L))), graph