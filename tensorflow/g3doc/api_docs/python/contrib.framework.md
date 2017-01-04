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

### `tf.contrib.framework.assert_scalar_int(tensor, name=None)` {#assert_scalar_int}

Assert `tensor` is 0-D, of type `tf.int32` or `tf.int64`.

##### Args:


*  <b>`tensor`</b>: `Tensor` to test.
*  <b>`name`</b>: Name of the op and of the new `Tensor` if one is created.

##### Returns:

  `tensor`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: if `tensor` is not 0-D, of type `tf.int32` or `tf.int64`.


- - -

### `tf.convert_to_tensor_or_sparse_tensor(value, dtype=None, name=None)` {#convert_to_tensor_or_sparse_tensor}

Converts value to a `SparseTensor` or `Tensor`.

##### Args:


*  <b>`value`</b>: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a
    registered `Tensor` conversion function.
*  <b>`dtype`</b>: Optional element type for the returned tensor. If missing, the
    type is inferred from the type of `value`.
*  <b>`name`</b>: Optional name to use if a new `Tensor` is created.

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

### `tf.contrib.framework.is_tensor(x)` {#is_tensor}

Check for tensor types.

Check whether an object is a tensor. Equivalent to
`isinstance(x, [tf.Tensor, tf.SparseTensor, tf.Variable])`.

##### Args:


*  <b>`x`</b>: An python object to check.

##### Returns:

  `True` if `x` is a tensor, `False` if not.


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



## Deprecation
- - -

### `tf.contrib.framework.deprecated(date, instructions)` {#deprecated}

Decorator for marking functions or methods deprecated.

This decorator logs a deprecation warning whenever the decorated function is
called. It has the following format:

  <function> (from <module>) is deprecated and will be removed after <date>.
  Instructions for updating:
  <instructions>

<function> will include the class name if it is a method.

It also edits the docstring of the function: ' (deprecated)' is appended
to the first line of the docstring and a deprecation notice is prepended
to the rest of the docstring.

##### Args:


*  <b>`date`</b>: String. The date the function is scheduled to be removed. Must be
    ISO 8601 (YYYY-MM-DD).
*  <b>`instructions`</b>: String. Instructions on how to update code using the
    deprecated function.

##### Returns:

  Decorated function or method.

##### Raises:


*  <b>`ValueError`</b>: If date is not in ISO 8601 format, or instructions are empty.


- - -

### `tf.contrib.framework.deprecated_args(date, instructions, *deprecated_arg_names_or_tuples)` {#deprecated_args}

Decorator for marking specific function arguments as deprecated.

This decorator logs a deprecation warning whenever the decorated function is
called with the deprecated argument. It has the following format:

  Calling <function> (from <module>) with <arg> is deprecated and will be
  removed after <date>. Instructions for updating:
    <instructions>

<function> will include the class name if it is a method.

It also edits the docstring of the function: ' (deprecated arguments)' is
appended to the first line of the docstring and a deprecation notice is
prepended to the rest of the docstring.

##### Args:


*  <b>`date`</b>: String. The date the function is scheduled to be removed. Must be
    ISO 8601 (YYYY-MM-DD).
*  <b>`instructions`</b>: String. Instructions on how to update code using the
    deprecated function.
*  <b>`*deprecated_arg_names_or_tuples`</b>: String. or 2-Tuple(String,
    [ok_vals]).  The string is the deprecated argument name.
    Optionally, an ok-value may be provided.  If the user provided
    argument equals this value, the warning is suppressed.

##### Returns:

  Decorated function or method.

##### Raises:


*  <b>`ValueError`</b>: If date is not in ISO 8601 format, instructions are
    empty, the deprecated arguments are not present in the function
    signature, or the second element of a deprecated_tuple is not a
    list.


- - -

### `tf.contrib.framework.deprecated_arg_values(date, instructions, **deprecated_kwargs)` {#deprecated_arg_values}

Decorator for marking specific function argument values as deprecated.

This decorator logs a deprecation warning whenever the decorated function is
called with the deprecated argument values. It has the following format:

  Calling <function> (from <module>) with <arg>=<value> is deprecated and
  will be removed after <date>. Instructions for updating:
    <instructions>

<function> will include the class name if it is a method.

It also edits the docstring of the function: ' (deprecated arguments)' is
appended to the first line of the docstring and a deprecation notice is
prepended to the rest of the docstring.

##### Args:


*  <b>`date`</b>: String. The date the function is scheduled to be removed. Must be
    ISO 8601 (YYYY-MM-DD).
*  <b>`instructions`</b>: String. Instructions on how to update code using the
    deprecated function.
*  <b>`**deprecated_kwargs`</b>: The deprecated argument values.

##### Returns:

  Decorated function or method.

##### Raises:


*  <b>`ValueError`</b>: If date is not in ISO 8601 format, or instructions are empty.



## Arg_Scope
- - -

### `tf.contrib.framework.arg_scope(list_ops_or_scope, **kwargs)` {#arg_scope}

Stores the default arguments for the given set of list_ops.

For usage, please see examples at top of the file.

##### Args:


*  <b>`list_ops_or_scope`</b>: List or tuple of operations to set argument scope for or
    a dictionary containing the current scope. When list_ops_or_scope is a
    dict, kwargs must be empty. When list_ops_or_scope is a list or tuple,
    then every op in it need to be decorated with @add_arg_scope to work.
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

Adds a variable to the `GraphKeys.MODEL_VARIABLES` collection.

##### Args:


*  <b>`var`</b>: a variable.


- - -

### `tf.train.assert_global_step(global_step_tensor)` {#assert_global_step}

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

### `tf.contrib.framework.assign_from_checkpoint(model_path, var_list)` {#assign_from_checkpoint}

Creates an operation to assign specific variables from a checkpoint.

##### Args:


*  <b>`model_path`</b>: The full path to the model checkpoint. To get latest checkpoint
      use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
*  <b>`var_list`</b>: A list of `Variable` objects or a dictionary mapping names in the
      checkpoint to the corresponding variables to initialize. If empty or
      None, it would return  no_op(), None.

##### Returns:

  the restore_op and the feed_dict that need to be run to restore var_list.

##### Raises:


*  <b>`ValueError`</b>: If the checkpoint specified at `model_path` is missing one of
    the variables in `var_list`.


- - -

### `tf.contrib.framework.assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False, reshape_variables=False)` {#assign_from_checkpoint_fn}

Returns a function that assigns specific variables from a checkpoint.

##### Args:


*  <b>`model_path`</b>: The full path to the model checkpoint. To get latest checkpoint
      use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
*  <b>`var_list`</b>: A list of `Variable` objects or a dictionary mapping names in the
      checkpoint to the correspoing variables to initialize. If empty or None,
      it would return  no_op(), None.
*  <b>`ignore_missing_vars`</b>: Boolean, if True it would ignore variables missing in
      the checkpoint with a warning instead of failing.
*  <b>`reshape_variables`</b>: Boolean, if True it would automatically reshape variables
      which are of different shape then the ones stored in the checkpoint but
      which have the same number of elements.

##### Returns:

  A function that takes a single argument, a `tf.Session`, that applies the
  assignment operation.

##### Raises:


*  <b>`ValueError`</b>: If the checkpoint specified at `model_path` is missing one of
    the variables in `var_list`.


- - -

### `tf.contrib.framework.assign_from_values(var_names_to_values)` {#assign_from_values}

Creates an assignment operation from a given mapping.

This function provides a mechanism for performing assignment of variables
to values in a way that does not fill the graph with large assignment values.

##### Args:


*  <b>`var_names_to_values`</b>: A map from variable names to values.

##### Returns:


*  <b>`assign_op`</b>: An `Operation` that assigns each of the given variables to the
    requested values.
*  <b>`feed_dict`</b>: The feed dictionary to use when evaluating `assign_op`.

##### Raises:


*  <b>`ValueError`</b>: if any of the given variable names were not found.


- - -

### `tf.contrib.framework.assign_from_values_fn(var_names_to_values)` {#assign_from_values_fn}

Returns a function that assigns specific variables from the given values.

This function provides a mechanism for performing assignment of variables
to values in a way that does not fill the graph with large assignment values.

##### Args:


*  <b>`var_names_to_values`</b>: A map from variable names to values.

##### Returns:

  A function that takes a single argument, a `tf.Session`, that applies the
  assignment operation.

##### Raises:


*  <b>`ValueError`</b>: if any of the given variable names were not found.


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

### `tf.train.get_global_step(graph=None)` {#get_global_step}

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

Gets the list of local variables, filtered by scope and/or suffix.

##### Args:


*  <b>`scope`</b>: an optional scope for filtering the variables to return.
*  <b>`suffix`</b>: an optional suffix for filtering the variables to return.

##### Returns:

  a list of variables in collection with scope and suffix.


- - -

### `tf.contrib.framework.get_model_variables(scope=None, suffix=None)` {#get_model_variables}

Gets the list of model variables, filtered by scope and/or suffix.

##### Args:


*  <b>`scope`</b>: an optional scope for filtering the variables to return.
*  <b>`suffix`</b>: an optional suffix for filtering the variables to return.

##### Returns:

  a list of variables in collection with scope and suffix.


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


*  <b>`scope`</b>: an optional scope for filtering the variables to return. Can be a
    variable scope or a string.
*  <b>`suffix`</b>: an optional suffix for filtering the variables to return.
*  <b>`collection`</b>: in which collection search for. Defaults to
    `GraphKeys.GLOBAL_VARIABLES`.

##### Returns:

  a list of variables in collection with scope and suffix.


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
    `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
*  <b>`collections`</b>: A list of collection names to which the Variable will be added.
    Note that the variable is always also added to the
    `GraphKeys.GLOBAL_VARIABLES` and `GraphKeys.MODEL_VARIABLES` collections.
*  <b>`caching_device`</b>: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's
      device.
*  <b>`device`</b>: Optional device to place the variable. It can be an string or a
    function that is called to get the device for the variable.
*  <b>`partitioner`</b>: Optional callable that accepts a fully defined `TensorShape`
    and dtype of the `Variable` to be created, and returns a list of
    partitions for each axis (currently only one axis can be partitioned).
*  <b>`custom_getter`</b>: Callable that allows overwriting the internal
    get_variable method and has to have the same signature.

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
    `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
*  <b>`collections`</b>: A list of collection names to which the Variable will be added.
    If None it would default to `tf.GraphKeys.GLOBAL_VARIABLES`.
*  <b>`caching_device`</b>: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's
      device.
*  <b>`device`</b>: Optional device to place the variable. It can be an string or a
    function that is called to get the device for the variable.
*  <b>`partitioner`</b>: Optional callable that accepts a fully defined `TensorShape`
    and dtype of the `Variable` to be created, and returns a list of
    partitions for each axis (currently only one axis can be partitioned).
*  <b>`custom_getter`</b>: Callable that allows overwriting the internal
    get_variable method and has to have the same signature.

##### Returns:

  The created or existing variable.


- - -

### `class tf.contrib.framework.VariableDeviceChooser` {#VariableDeviceChooser}

Device chooser for variables.

When using a parameter server it will assign them in a round-robin fashion.
When not using a parameter server it allows GPU or CPU placement.
- - -

#### `tf.contrib.framework.VariableDeviceChooser.__call__(op)` {#VariableDeviceChooser.__call__}




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



- - -

### `tf.contrib.framework.zero_initializer(ref, use_locking=True, name='zero_initializer')` {#zero_initializer}

Initialize 'ref' with all zeros, ref tensor should be uninitialized.
If already initialized, you will get ValueError. This op is intended to
save memory during initialization.

##### Args:


*  <b>`ref`</b>: ref of the tensor need to be zero initialized.
*  <b>`name`</b>: optional name for this operation.

##### Returns:

  ref that initialized.

##### Raises:


*  <b>`ValueError`</b>: If ref tensor is initialized.


