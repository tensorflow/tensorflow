def get_variable_values(layer, graph=tf.get_default_graph()):
    """ Fetches Variables of the given layer
    Args: 
        layer: layer of the model
    Returns:
        A Dictionary containing Kernels and Biases.
        
        example:
            param_dict = get_variables(layer, graph)
        
    """

    M = []
    P = []
    param_dict = {}
    P_Wb = {}

    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=layer):
        m = re.search(r'\w+\/\w+\:\d', str(i)).group(0)
        M.append(m)
        v = m.split(":")[0]
        param_type = ''.join(v.split("/")[1])
        P.append(param_type)
        print(("saving: {}".format(m)))
    param_dict[P[0]] = graph.get_tensor_by_name(M[0]).eval()  # Weights
    param_dict[P[1]] = graph.get_tensor_by_name(M[1]).eval()  # biases
    P_Wb[layer] = param_dict
    return P_Wb
