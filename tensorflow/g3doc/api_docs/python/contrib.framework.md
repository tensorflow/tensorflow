<!-- This file is machine generated: DO NOT EDIT! -->

# Framework (contrib)
[TOC]

Framework utilities.

- - -

### `tf.contrib.framework.assert_same_float_dtype(tensors=None, dtype=None)` {#assert_same_float_dtype}

Validate and return float type based on `tensors` and `dtype`.

For ops such as matrix multiplication, inputs and weights must be of the
same float type. This function validates that all `tensors` are the same type,
validates that type is `dtype` (if supplied), and returns the type. Type must
be `dtypes.float32` or `dtypes.float64`. If neither `tensors` nor
`dtype` is supplied, default to `dtypes.float32`.

##### Args:


*  <b>`tensors`</b>: Tensors of input values. Can include `None` elements, which will be
      ignored.
*  <b>`dtype`</b>: Expected type.

##### Returns:

  Validated type.

##### Raises:


*  <b>`ValueError`</b>: if neither `tensors` nor `dtype` is supplied, or result is not
      float.


- - -

### `tf.contrib.framework.assert_scalar_int(tensor)` {#assert_scalar_int}

Assert `tensor` is 0-D, of type `tf.int32` or `tf.int64`.

##### Args:


*  <b>`tensor`</b>: Tensor to test.

##### Returns:

  `tensor`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: if `tensor` is not 0-D, of type `tf.int32` or `tf.int64`.


- - -

### `tf.contrib.framework.convert_to_tensor_or_sparse_tensor(value, dtype=None, name=None, as_ref=False)` {#convert_to_tensor_or_sparse_tensor}

Converts value to a `SparseTensor` or `Tensor`.

##### Args:


*  <b>`value`</b>: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a
    registered `Tensor` conversion function.
*  <b>`dtype`</b>: Optional element type for the returned tensor. If missing, the
    type is inferred from the type of `value`.
*  <b>`name`</b>: Optional name to use if a new `Tensor` is created.
*  <b>`as_ref`</b>: True if we want the result as a ref tensor. Only used if a new
    `Tensor` is created.

##### Returns:

  A `SparseTensor` or `Tensor` based on `value`.

##### Raises:


*  <b>`RuntimeError`</b>: If result type is incompatible with `dtype`.


- - -

### `tf.contrib.framework.get_graph_from_inputs(op_input_list, graph=None)` {#get_graph_from_inputs}

Returns the appropriate graph to use for the given inputs.

1. If `graph` is provided, we validate that all inputs in `op_input_list` are
   from the same graph.
2. Otherwise, we attempt to select a graph from the first Operation- or
   Tensor-valued input in `op_input_list`, and validate that all other
   such inputs are in the same graph.
3. If the graph was not specified and it could not be inferred from
   `op_input_list`, we attempt to use the default graph.

##### Args:


*  <b>`op_input_list`</b>: A list of inputs to an operation, which may include `Tensor`,
    `Operation`, and other objects that may be converted to a graph element.
*  <b>`graph`</b>: (Optional) The explicit graph to use.

##### Raises:


*  <b>`TypeError`</b>: If `op_input_list` is not a list or tuple, or if graph is not a
    Graph.
*  <b>`ValueError`</b>: If a graph is explicitly passed and not all inputs are from it,
    or if the inputs are from multiple graphs, or we could not find a graph
    and there was no default graph.

##### Returns:

  The appropriate graph to use for the given inputs.


- - -

### `tf.is_numeric_tensor(tensor)` {#is_numeric_tensor}




- - -

### `tf.is_non_decreasing(x, name=None)` {#is_non_decreasing}

Returns `True` if `x` is non-decreasing.

Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
If `x` has less than two elements, it is trivially non-decreasing.

See also:  `is_strictly_increasing`

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "is_non_decreasing"

##### Returns:

  Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.

##### Raises:


*  <b>`TypeError`</b>: if `x` is not a numeric tensor.


- - -

### `tf.is_strictly_increasing(x, name=None)` {#is_strictly_increasing}

Returns `True` if `x` is strictly increasing.

Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.
If `x` has less than two elements, it is trivially strictly increasing.

See also:  `is_non_decreasing`

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`name`</b>: A name for this operation (optional).
    Defaults to "is_strictly_increasing"

##### Returns:

  Boolean `Tensor`, equal to `True` iff `x` is strictly increasing.

##### Raises:


*  <b>`TypeError`</b>: if `x` is not a numeric tensor.


- - -

### `tf.contrib.framework.reduce_sum_n(tensors, name=None)` {#reduce_sum_n}

Reduce tensors to a scalar sum.

This reduces each tensor in `tensors` to a scalar via `tf.reduce_sum`, then
adds them via `tf.add_n`.

##### Args:


*  <b>`tensors`</b>: List of tensors, all of the same numeric type.
*  <b>`name`</b>: Tensor name, and scope for all other ops.

##### Returns:

  Total loss tensor, or None if no losses have been configured.

##### Raises:


*  <b>`ValueError`</b>: if `losses` is missing or empty.


- - -

### `tf.contrib.framework.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights=None, combiner='mean', default_id=None, name=None, partition_strategy='div')` {#safe_embedding_lookup_sparse}

Lookup embedding results, accounting for invalid IDs and empty features.

The partitioned embedding in `embedding_weights` must all be the same shape
except for the first dimension. The first dimension is allowed to vary as the
vocabulary size is not necessarily a multiple of `P`.

Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
with non-positive weight. For an entry with no features, the embedding vector
for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

##### Args:


*  <b>`embedding_weights`</b>: A list of `P` float tensors or values representing
      partitioned embedding tensors.
*  <b>`sparse_ids`</b>: `SparseTensor` of shape `[batch_size, ?]` containing the ids.
*  <b>`sparse_weights`</b>: `SparseTensor` of same shape as `sparse_ids`, containing
      float weights corresponding to `sparse_ids`, or `None` if all weights
      are be assumed to be 1.0.
*  <b>`combiner`</b>: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
      the default.
*  <b>`default_id`</b>: The id to use for an entry with no features.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy.
      Currently `"div"` and `"mod"` are supported. Default is `"div"`.


##### Returns:

  Dense tensor of shape `[batch_size, embed_dim]`.

##### Raises:


*  <b>`ValueError`</b>: if `embedding_weights` is empty.


- - -

### `tf.contrib.framework.with_shape(expected_shape, tensor)` {#with_shape}

Asserts tensor has expected shape.

If tensor shape and expected_shape, are fully defined, assert they match.
Otherwise, add assert op that will validate the shape when tensor is
evaluated, and set shape on tensor.

##### Args:


*  <b>`expected_shape`</b>: Expected shape to assert, as a 1D array of ints, or tensor
      of same.
*  <b>`tensor`</b>: Tensor whose shape we're validating.

##### Returns:

  tensor, perhaps with a dependent assert operation.

##### Raises:


*  <b>`ValueError`</b>: if tensor has an invalid shape.


- - -

### `tf.contrib.framework.with_same_shape(expected_tensor, tensor)` {#with_same_shape}

Assert tensors are the same shape, from the same graph.

##### Args:


*  <b>`expected_tensor`</b>: Tensor with expected shape.
*  <b>`tensor`</b>: Tensor of actual values.

##### Returns:

  Tuple of (actual_tensor, label_tensor), possibly with assert ops added.




## Arg_Scope
- - -

### `tf.contrib.framework.arg_scope(list_ops_or_scope, **kwargs)` {#arg_scope}

Stores the default arguments for the given set of list_ops.

For usage, please see examples at top of the file.

##### Args:


*  <b>`list_ops_or_scope`</b>: List or tuple of operations to set argument scope for or
    a dictionary containg the current scope. When list_ops_or_scope is a dict,
    kwargs must be empty. When list_ops_or_scope is a list or tuple, then
    every op in it need to be decorated with @add_arg_scope to work.
*  <b>`**kwargs`</b>: keyword=value that will define the defaults for each op in
            list_ops. All the ops need to accept the given set of arguments.

##### Yields:

  the current_scope, which is a dictionary of {op: {arg: value}}

##### Raises:


*  <b>`TypeError`</b>: if list_ops is not a list or a tuple.
*  <b>`ValueError`</b>: if any op in list_ops has not be decorated with @add_arg_scope.


- - -

### `tf.contrib.framework.add_arg_scope(func)` {#add_arg_scope}

Decorates a function with args so it can be used within an arg_scope.

##### Args:


*  <b>`func`</b>: function to decorate.

##### Returns:

  A tuple with the decorated function func_with_args().


- - -

### `tf.contrib.framework.has_arg_scope(func)` {#has_arg_scope}

Checks whether a func has been decorated with @add_arg_scope or not.

##### Args:


*  <b>`func`</b>: function to check.

##### Returns:

  a boolean.


- - -

### `tf.contrib.framework.arg_scoped_arguments(func)` {#arg_scoped_arguments}

Returns the list kwargs that arg_scope can set for a func.

##### Args:


*  <b>`func`</b>: function which has been decorated with @add_arg_scope.

##### Returns:

  a list of kwargs names.



## Variables
- - -

### `tf.contrib.framework.add_model_variable(var)` {#add_model_variable}

Adds a variable to the MODEL_VARIABLES collection.

##### Args:


*  <b>`var`</b>: a variable.


- - -

### `tf.contrib.framework.assert_global_step(global_step_tensor)` {#assert_global_step}

Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.

##### Args:


*  <b>`global_step_tensor`</b>: `Tensor` to test.


- - -

### `tf.contrib.framework.assert_or_get_global_step(graph=None, global_step_tensor=None)` {#assert_or_get_global_step}

Verifies that a global step tensor is valid or gets one if None is given.

If `global_step_tensor` is not None, check that it is a valid global step
tensor (using `assert_global_step`). Otherwise find a global step tensor using
`get_global_step` and return it.

##### Args:


*  <b>`graph`</b>: The graph to find the global step tensor for.
*  <b>`global_step_tensor`</b>: The tensor to check for suitability as a global step.
    If None is given (the default), find a global step tensor.

##### Returns:

  A tensor suitable as a global step, or `None` if none was provided and none
  was found.


- - -

### `tf.contrib.framework.create_global_step(graph=None)` {#create_global_step}

Create global step tensor in graph.

##### Args:


*  <b>`graph`</b>: The graph in which to create the global step. If missing, use default
      graph.

##### Returns:

  Global step tensor.

##### Raises:


*  <b>`ValueError`</b>: if global step key is already defined.


- - -

### `tf.contrib.framework.get_global_step(graph=None)` {#get_global_step}

Get the global step tensor.

The global step tensor must be an integer variable. We first try to find it
in the collection `GLOBAL_STEP`, or by name `global_step:0`.

##### Args:


*  <b>`graph`</b>: The graph to find the global step in. If missing, use default graph.

##### Returns:

  The global step variable, or `None` if none was found.

##### Raises:


*  <b>`TypeError`</b>: If the global step tensor has a non-integer type, or if it is not
    a `Variable`.


- - -

### `tf.contrib.framework.get_or_create_global_step(graph=None)` {#get_or_create_global_step}

Returns and create (if necessary) the global step variable.

##### Args:


*  <b>`graph`</b>: The graph in which to create the global step. If missing, use default
      graph.

##### Returns:

  the tensor representing the global step variable.


- - -

### `tf.contrib.framework.get_local_variables(scope=None, suffix=None)` {#get_local_variables}

Gets the list of model variables, filtered by scope and/or suffix.

##### Args:


*  <b>`scope`</b>: an optional scope for filtering the variables to return.
*  <b>`suffix`</b>: an optional suffix for filtering the variables to return.

##### Returns:

  a list of variables in colelction with scope and suffix.


- - -

### `tf.contrib.framework.get_model_variables(scope=None, suffix=None)` {#get_model_variables}

Gets the list of model variables, filtered by scope and/or suffix.

##### Args:


*  <b>`scope`</b>: an optional scope for filtering the variables to return.
*  <b>`suffix`</b>: an optional suffix for filtering the variables to return.

##### Returns:

  a list of variables in colelction with scope and suffix.


- - -

### `tf.contrib.framework.get_unique_variable(var_op_name)` {#get_unique_variable}

Gets the variable uniquely identified by that var_op_name.

##### Args:


*  <b>`var_op_name`</b>: the full name of the variable op, including the scope.

##### Returns:

  a tensorflow variable.

##### Raises:


*  <b>`ValueError`</b>: if no variable uniquely identified by the name exists.


- - -

### `tf.contrib.framework.get_variables_by_name(given_name, scope=None)` {#get_variables_by_name}

Gets the list of variables that were given that name.

##### Args:


*  <b>`given_name`</b>: name given to the variable without any scope.
*  <b>`scope`</b>: an optional scope for filtering the variables to return.

##### Returns:

  a copied list of variables with the given name and scope.


- - -

### `tf.contrib.framework.get_variables_by_suffix(suffix, scope=None)` {#get_variables_by_suffix}

Gets the list of variables that end with the given suffix.

##### Args:


*  <b>`suffix`</b>: suffix for filtering the variables to return.
*  <b>`scope`</b>: an optional scope for filtering the variables to return.

##### Returns:

  a copied list of variables with the given name and prefix.


- - -

### `tf.contrib.framework.get_variables_to_restore(include=None, exclude=None)` {#get_variables_to_restore}

Gets the list of the variables to restore.

##### Args:


*  <b>`include`</b>: an optional list/tuple of scope strings for filtering which
    variables from the VARIABLES collection to include. None would include all
    the variables.
*  <b>`exclude`</b>: an optional list/tuple of scope strings for filtering which
    variables from the VARIABLES collection to exclude. None it would not
    exclude any.

##### Returns:

  a list of variables to restore.

##### Raises:


*  <b>`TypeError`</b>: include or exclude is provided but is not a list or a tuple.


- - -

### `tf.contrib.framework.get_variables(scope=None, suffix=None, collection='variables')` {#get_variables}

Gets the list of variables, filtered by scope and/or suffix.

##### Args:


*  <b>`scope`</b>: an optional scope for filtering the variables to return.
*  <b>`suffix`</b>: an optional suffix for filtering the variables to return.
*  <b>`collection`</b>: in which collection search for. Defaults to GraphKeys.VARIABLES.

##### Returns:

  a list of variables in colelction with scope and suffix.


- - -

### `tf.contrib.framework.local_variable(initial_value, validate_shape=True, name=None)` {#local_variable}

Create variable and add it to `GraphKeys.LOCAL_VARIABLES` collection.

##### Args:


*  <b>`initial_value`</b>: See variables.Variable.__init__.
*  <b>`validate_shape`</b>: See variables.Variable.__init__.
*  <b>`name`</b>: See variables.Variable.__init__.

##### Returns:

  New variable.


- - -

### `tf.contrib.framework.model_variable(*args, **kwargs)` {#model_variable}

Gets an existing model variable with these parameters or creates a new one.

##### Args:


*  <b>`name`</b>: the name of the new or existing variable.
*  <b>`shape`</b>: shape of the new or existing variable.
*  <b>`dtype`</b>: type of the new or existing variable (defaults to `DT_FLOAT`).
*  <b>`initializer`</b>: initializer for the variable if one is created.
*  <b>`regularizer`</b>: a (Tensor -> Tensor or None) function; the result of
      applying it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
*  <b>`trainable`</b>: If `True` also add the variable to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`collections`</b>: A list of collection names to which the Variable will be added.
    Note that the variable is always also added to the tf.GraphKeys.VARIABLES
    and MODEL_VARIABLES collections.
*  <b>`caching_device`</b>: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's
      device.
*  <b>`device`</b>: Optional device to place the variable. It can be an string or a
    function that is called to get the device for the variable.

##### Returns:

  The created or existing variable.


- - -

### `tf.contrib.framework.variable(*args, **kwargs)` {#variable}

Gets an existing variable with these parameters or creates a new one.

##### Args:


*  <b>`name`</b>: the name of the new or existing variable.
*  <b>`shape`</b>: shape of the new or existing variable.
*  <b>`dtype`</b>: type of the new or existing variable (defaults to `DT_FLOAT`).
*  <b>`initializer`</b>: initializer for the variable if one is created.
*  <b>`regularizer`</b>: a (Tensor -> Tensor or None) function; the result of
      applying it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
*  <b>`trainable`</b>: If `True` also add the variable to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`collections`</b>: A list of collection names to which the Variable will be added.
    If None it would default to tf.GraphKeys.VARIABLES.
*  <b>`caching_device`</b>: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's
      device.
*  <b>`device`</b>: Optional device to place the variable. It can be an string or a
    function that is called to get the device for the variable.

##### Returns:

  The created or existing variable.


- - -

### `class tf.contrib.framework.VariableDeviceChooser` {#VariableDeviceChooser}

Device chooser for variables.

When using a parameter server it will assign them in a round-robin fashion.
When not using a parameter server it allows GPU or CPU placement.
- - -

#### `tf.contrib.framework.VariableDeviceChooser.__init__(num_tasks=0, job_name='ps', device_type='CPU', device_index=0)` {#VariableDeviceChooser.__init__}

Initialize VariableDeviceChooser.

##### Usage:

  To use with 2 parameter servers:
    VariableDeviceChooser(2)

  To use without parameter servers:
    VariableDeviceChooser()
    VariableDeviceChooser(device_type='GPU') # For GPU placement

##### Args:


*  <b>`num_tasks`</b>: number of tasks.
*  <b>`job_name`</b>: String, a name for the parameter server job.
*  <b>`device_type`</b>: Optional device type string (e.g. "CPU" or "GPU")
*  <b>`device_index`</b>: int.  Optional device index.  If left
    unspecified, device represents 'any' device_index.



