import tensorflow as tf
from tensorflow.python.framework import ops
from copy import deepcopy
 
 
def copy_variable_to_graph(org_instance, to_graph, namespace,
                           copied_variables={}):
    """
    Copies the Variable instance 'org_instance' into the graph
    'to_graph', under the given namespace.
    The dict 'copied_variables', if provided, will be updated with
    mapping the new variable's name to the instance.
    """
 
    if not isinstance(org_instance, tf.Variable):
        raise TypeError(str(org_instance) + " is not a Variable")
 
    #The name of the new variable
    if namespace != '':
        new_name = (namespace + '/' +
                    org_instance.name[:org_instance.name.index(':')])
    else:
        new_name = org_instance.name[:org_instance.name.index(':')]
 
    #Get the collections that the new instance needs to be added to.
    #The new collections will also be a part of the given namespace,
    #except the special ones required for variable initialization and
    #training.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
            if (name == ops.GraphKeys.VARIABLES or
                name == ops.GraphKeys.TRAINABLE_VARIABLES or
                namespace == ''):
                collections.append(name)
            else:
                collections.append(namespace + '/' + name)
 
    #See if its trainable.
    trainable = (org_instance in org_instance.graph.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES))
    #Get the initial value
    with org_instance.graph.as_default():
        temp_session = tf.Session()
        init_value = temp_session.run(org_instance.initialized_value())
 
    #Initialize the new variable
    with to_graph.as_default():
        new_var = tf.Variable(init_value,
                              trainable,
                              name=new_name,
                              collections=collections,
                              validate_shape=False)
 
    #Add to the copied_variables dict
    copied_variables[new_var.name] = new_var
 
    return new_var
 
 
def copy_to_graph(org_instance, to_graph, copied_variables={}, namespace=""):
    """
    Makes a copy of the Operation/Tensor instance 'org_instance'
    for the graph 'to_graph', recursively. Therefore, all required
    structures linked to org_instance will be automatically copied.
    'copied_variables' should be a dict mapping pertinent copied variable
    names to the copied instances.
     
    The new instances are automatically inserted into the given 'namespace'.
    If namespace='', it is inserted into the graph's global namespace.
    However, to avoid naming conflicts, its better to provide a namespace.
    If the instance(s) happens to be a part of collection(s), they are
    are added to the appropriate collections in to_graph as well.
    For example, for collection 'C' which the instance happens to be a
    part of, given a namespace 'N', the new instance will be a part of
    'N/C' in to_graph.
 
    Returns the corresponding instance with respect to to_graph.
 
    TODO: Order of insertion into collections is not preserved
    """
 
    #The name of the new instance
    if namespace != '':
        new_name = namespace + '/' + org_instance.name
    else:
        new_name = org_instance.name
 
    #If a variable by the new name already exists, return the
    #correspondng tensor that will act as an input
    if new_name in copied_variables:
        return to_graph.get_tensor_by_name(
            copied_variables[new_name].name)
 
    #If an instance of the same name exists, return appropriately
    try:
        already_present = to_graph.as_graph_element(new_name,
                                                    allow_tensor=True,
                                                    allow_operation=True)
        return already_present
    except:
        pass
 
    #Get the collections that the new instance needs to be added to.
    #The new collections will also be a part of the given namespace.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
            if namespace == '':
                collections.append(name)
            else:
                collections.append(namespace + '/' + name)
     
    #Take action based on the class of the instance
 
    if isinstance(org_instance, tf.python.framework.ops.Tensor):
 
        #If its a Tensor, it is one of the outputs of the underlying
        #op. Therefore, copy the op itself and return the appropriate
        #output.
        op = org_instance.op
        new_op = copy_to_graph(op, to_graph, copied_variables, namespace)
        output_index = op.outputs.index(org_instance)
        new_tensor = new_op.outputs[output_index]
        #Add to collections if any
        for collection in collections:
            to_graph.add_to_collection(collection, new_tensor)
 
        return new_tensor
 
    elif isinstance(org_instance, tf.python.framework.ops.Operation):
 
        op = org_instance
 
        #If it has an original_op parameter, copy it
        if op._original_op is not None:
            new_original_op = copy_to_graph(op._original_op, to_graph,
                                            copied_variables, namespace)
        else:
            new_original_op = None
 
        #If it has control inputs, call this function recursively on each.
        new_control_inputs = [copy_to_graph(x, to_graph, copied_variables,
                                            namespace)
                              for x in op.control_inputs]
 
        #If it has inputs, call this function recursively on each.
        new_inputs = [copy_to_graph(x, to_graph, copied_variables,
                                    namespace)
                      for x in op.inputs]
 
        #Make a new node_def based on that of the original.
        #An instance of tensorflow.core.framework.graph_pb2.NodeDef, it
        #stores String-based info such as name, device and type of the op.
        #Unique to every Operation instance.
        new_node_def = deepcopy(op._node_def)
        #Change the name
        new_node_def.name = new_name
 
        #Copy the other inputs needed for initialization
        output_types = op._output_types[:]
        input_types = op._input_types[:]
 
        #Make a copy of the op_def too.
        #Its unique to every _type_ of Operation.
        op_def = deepcopy(op._op_def)
 
        #Initialize a new Operation instance
        new_op = tf.python.framework.ops.Operation(new_node_def,
                                                   to_graph,
                                                   new_inputs,
                                                   output_types,
                                                   new_control_inputs,
                                                   input_types,
                                                   new_original_op,
                                                   op_def)
        #Use Graph's hidden methods to add the op
        to_graph._add_op(new_op)
        to_graph._record_op_seen_by_control_dependencies(new_op)
        for device_function in reversed(to_graph._device_function_stack):
            new_op._set_device(device_function(new_op))
 
        return new_op
 
    else:
        raise TypeError("Could not copy instance: " + str(org_instance))
 
 
def get_copied(original, graph, copied_variables={}, namespace=""):
    """
    Get a copy of the instance 'original', present in 'graph', under
    the given 'namespace'.
    'copied_variables' is a dict mapping pertinent variable names to the
    copy instances.
    """
 
    #The name of the copied instance
    if namespace != '':
        new_name = namespace + '/' + original.name
    else:
        new_name = original.name
 
    #If a variable by the name already exists, return it
    if new_name in copied_variables:
        return copied_variables[new_name]
 
    return graph.as_graph_element(new_name, allow_tensor=True,
                                  allow_operation=True)
