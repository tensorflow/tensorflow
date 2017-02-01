<!-- This file is machine generated: DO NOT EDIT! -->

# Building Graphs
[TOC]

Classes and functions for building TensorFlow graphs.

## Core graph data structures

- - -

### `class tf.Graph` {#Graph}

A TensorFlow computation, represented as a dataflow graph.

A `Graph` contains a set of
[`Operation`](../../api_docs/python/framework.md#Operation) objects,
which represent units of computation; and
[`Tensor`](../../api_docs/python/framework.md#Tensor) objects, which represent
the units of data that flow between operations.

A default `Graph` is always registered, and accessible by calling
[`tf.get_default_graph()`](../../api_docs/python/framework.md#get_default_graph).
To add an operation to the default graph, simply call one of the functions
that defines a new `Operation`:

```python
c = tf.constant(4.0)
assert c.graph is tf.get_default_graph()
```

Another typical usage involves the
[`Graph.as_default()`](../../api_docs/python/framework.md#Graph.as_default)
context manager, which overrides the current default graph for the
lifetime of the context:

```python
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g
```

Important note: This class *is not* thread-safe for graph construction. All
operations should be created from a single thread, or external
synchronization must be provided. Unless otherwise specified, all methods
are not thread-safe.

- - -

#### `tf.Graph.__init__()` {#Graph.__init__}

Creates a new, empty Graph.


- - -

#### `tf.Graph.as_default()` {#Graph.as_default}

Returns a context manager that makes this `Graph` the default graph.

This method should be used if you want to create multiple graphs
in the same process. For convenience, a global default graph is
provided, and all ops will be added to this graph if you do not
create a new graph explicitly. Use this method with the `with` keyword
to specify that ops created within the scope of a block should be
added to this graph.

The default graph is a property of the current thread. If you
create a new thread, and wish to use the default graph in that
thread, you must explicitly add a `with g.as_default():` in that
thread's function.

The following code examples are equivalent:

```python
# 1. Using Graph.as_default():
g = tf.Graph()
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g

# 2. Constructing and making default:
with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  assert c.graph is g
```

##### Returns:

  A context manager for using this graph as the default graph.


- - -

#### `tf.Graph.as_graph_def(from_version=None, add_shapes=False)` {#Graph.as_graph_def}

Returns a serialized `GraphDef` representation of this graph.

The serialized `GraphDef` can be imported into another `Graph`
(using [`import_graph_def()`](#import_graph_def)) or used with the
[C++ Session API](../../api_docs/cc/index.md).

This method is thread-safe.

##### Args:


*  <b>`from_version`</b>: Optional.  If this is set, returns a `GraphDef`
    containing only the nodes that were added to this graph since
    its `version` property had the given value.
*  <b>`add_shapes`</b>: If true, adds an "_output_shapes" list attr to each
    node with the inferred shapes of each of its outputs.

##### Returns:

  A [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer.

##### Raises:


*  <b>`ValueError`</b>: If the `graph_def` would be too large.


- - -

#### `tf.Graph.finalize()` {#Graph.finalize}

Finalizes this graph, making it read-only.

After calling `g.finalize()`, no new operations can be added to
`g`.  This method is used to ensure that no operations are added
to a graph when it is shared between multiple threads, for example
when using a [`QueueRunner`](../../api_docs/python/train.md#QueueRunner).


- - -

#### `tf.Graph.finalized` {#Graph.finalized}

True if this graph has been finalized.



- - -

#### `tf.Graph.control_dependencies(control_inputs)` {#Graph.control_dependencies}

Returns a context manager that specifies control dependencies.

Use with the `with` keyword to specify that all operations constructed
within the context should have control dependencies on
`control_inputs`. For example:

```python
with g.control_dependencies([a, b, c]):
  # `d` and `e` will only run after `a`, `b`, and `c` have executed.
  d = ...
  e = ...
```

Multiple calls to `control_dependencies()` can be nested, and in
that case a new `Operation` will have control dependencies on the union
of `control_inputs` from all active contexts.

```python
with g.control_dependencies([a, b]):
  # Ops constructed here run after `a` and `b`.
  with g.control_dependencies([c, d]):
    # Ops constructed here run after `a`, `b`, `c`, and `d`.
```

You can pass None to clear the control dependencies:

```python
with g.control_dependencies([a, b]):
  # Ops constructed here run after `a` and `b`.
  with g.control_dependencies(None):
    # Ops constructed here run normally, not waiting for either `a` or `b`.
    with g.control_dependencies([c, d]):
      # Ops constructed here run after `c` and `d`, also not waiting
      # for either `a` or `b`.
```

*N.B.* The control dependencies context applies *only* to ops that
are constructed within the context. Merely using an op or tensor
in the context does not add a control dependency. The following
example illustrates this point:

```python
# WRONG
def my_func(pred, tensor):
  t = tf.matmul(tensor, tensor)
  with tf.control_dependencies([pred]):
    # The matmul op is created outside the context, so no control
    # dependency will be added.
    return t

# RIGHT
def my_func(pred, tensor):
  with tf.control_dependencies([pred]):
    # The matmul op is created in the context, so a control dependency
    # will be added.
    return tf.matmul(tensor, tensor)
```

##### Args:


*  <b>`control_inputs`</b>: A list of `Operation` or `Tensor` objects which
    must be executed or computed before running the operations
    defined in the context.  Can also be `None` to clear the control
    dependencies.

##### Returns:

 A context manager that specifies control dependencies for all
 operations constructed within the context.

##### Raises:


*  <b>`TypeError`</b>: If `control_inputs` is not a list of `Operation` or
    `Tensor` objects.


- - -

#### `tf.Graph.device(device_name_or_function)` {#Graph.device}

Returns a context manager that specifies the default device to use.

The `device_name_or_function` argument may either be a device name
string, a device function, or None:

* If it is a device name string, all operations constructed in
  this context will be assigned to the device with that name, unless
  overridden by a nested `device()` context.
* If it is a function, it will be treated as a function from
  Operation objects to device name strings, and invoked each time
  a new Operation is created. The Operation will be assigned to
  the device with the returned name.
* If it is None, all `device()` invocations from the enclosing context
  will be ignored.

For information about the valid syntax of device name strings, see
the documentation in
[`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).

For example:

```python
with g.device('/gpu:0'):
  # All operations constructed in this context will be placed
  # on GPU 0.
  with g.device(None):
    # All operations constructed in this context will have no
    # assigned device.

# Defines a function from `Operation` to device string.
def matmul_on_gpu(n):
  if n.type == "MatMul":
    return "/gpu:0"
  else:
    return "/cpu:0"

with g.device(matmul_on_gpu):
  # All operations of type "MatMul" constructed in this context
  # will be placed on GPU 0; all other operations will be placed
  # on CPU 0.
```

**N.B.** The device scope may be overridden by op wrappers or
other library code. For example, a variable assignment op
`v.assign()` must be colocated with the `tf.Variable` `v`, and
incompatible device scopes will be ignored.

##### Args:


*  <b>`device_name_or_function`</b>: The device name or function to use in
    the context.

##### Returns:

  A context manager that specifies the default device to use for newly
  created ops.


- - -

#### `tf.Graph.name_scope(name)` {#Graph.name_scope}

Returns a context manager that creates hierarchical names for operations.

A graph maintains a stack of name scopes. A `with name_scope(...):`
statement pushes a new name onto the stack for the lifetime of the context.

The `name` argument will be interpreted as follows:

* A string (not ending with '/') will create a new name scope, in which
  `name` is appended to the prefix of all operations created in the
  context. If `name` has been used before, it will be made unique by
  calling `self.unique_name(name)`.
* A scope previously captured from a `with g.name_scope(...) as
  scope:` statement will be treated as an "absolute" name scope, which
  makes it possible to re-enter existing scopes.
* A value of `None` or the empty string will reset the current name scope
  to the top-level (empty) name scope.

For example:

```python
with tf.Graph().as_default() as g:
  c = tf.constant(5.0, name="c")
  assert c.op.name == "c"
  c_1 = tf.constant(6.0, name="c")
  assert c_1.op.name == "c_1"

  # Creates a scope called "nested"
  with g.name_scope("nested") as scope:
    nested_c = tf.constant(10.0, name="c")
    assert nested_c.op.name == "nested/c"

    # Creates a nested scope called "inner".
    with g.name_scope("inner"):
      nested_inner_c = tf.constant(20.0, name="c")
      assert nested_inner_c.op.name == "nested/inner/c"

    # Create a nested scope called "inner_1".
    with g.name_scope("inner"):
      nested_inner_1_c = tf.constant(30.0, name="c")
      assert nested_inner_1_c.op.name == "nested/inner_1/c"

      # Treats `scope` as an absolute name scope, and
      # switches to the "nested/" scope.
      with g.name_scope(scope):
        nested_d = tf.constant(40.0, name="d")
        assert nested_d.op.name == "nested/d"

        with g.name_scope(""):
          e = tf.constant(50.0, name="e")
          assert e.op.name == "e"
```

The name of the scope itself can be captured by `with
g.name_scope(...) as scope:`, which stores the name of the scope
in the variable `scope`. This value can be used to name an
operation that represents the overall result of executing the ops
in a scope. For example:

```python
inputs = tf.constant(...)
with g.name_scope('my_layer') as scope:
  weights = tf.Variable(..., name="weights")
  biases = tf.Variable(..., name="biases")
  affine = tf.matmul(inputs, weights) + biases
  output = tf.nn.relu(affine, name=scope)
```

NOTE: This constructor validates the given `name`. Valid scope
names match one of the following regular expressions:

    [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (for scopes at the root)
    [A-Za-z0-9_.\\-/]* (for other scopes)

##### Args:


*  <b>`name`</b>: A name for the scope.

##### Returns:

  A context manager that installs `name` as a new name scope.

##### Raises:


*  <b>`ValueError`</b>: If `name` is not a valid scope name. The rules are the



A `Graph` instance supports an arbitrary number of "collections"
that are identified by name. For convenience when building a large
graph, collections can store groups of related objects: for
example, the `tf.Variable` uses a collection (named
[`tf.GraphKeys.GLOBAL_VARIABLES`](../../api_docs/python/framework.md#GraphKeys)) for
all variables that are created during the construction of a graph. The caller
may define additional collections by specifying a new name.

- - -

#### `tf.Graph.add_to_collection(name, value)` {#Graph.add_to_collection}

Stores `value` in the collection with the given `name`.

Note that collections are not sets, so it is possible to add a value to
a collection several times.

##### Args:


*  <b>`name`</b>: The key for the collection. The `GraphKeys` class
    contains many standard names for collections.
*  <b>`value`</b>: The value to add to the collection.


- - -

#### `tf.Graph.add_to_collections(names, value)` {#Graph.add_to_collections}

Stores `value` in the collections given by `names`.

Note that collections are not sets, so it is possible to add a value to
a collection several times. This function makes sure that duplicates in
`names` are ignored, but it will not check for pre-existing membership of
`value` in any of the collections in `names`.

`names` can be any iterable, but if `names` is a string, it is treated as a
single collection name.

##### Args:


*  <b>`names`</b>: The keys for the collections to add to. The `GraphKeys` class
    contains many standard names for collections.
*  <b>`value`</b>: The value to add to the collections.


- - -

#### `tf.Graph.get_collection(name, scope=None)` {#Graph.get_collection}

Returns a list of values in the collection with the given `name`.

This is different from `get_collection_ref()` which always returns the
actual collection list if it exists in that it returns a new list each time
it is called.

##### Args:


*  <b>`name`</b>: The key for the collection. For example, the `GraphKeys` class
    contains many standard names for collections.
*  <b>`scope`</b>: (Optional.) If supplied, the resulting list is filtered to include
    only items whose `name` attribute matches using `re.match`. Items
    without a `name` attribute are never returned if a scope is supplied and
    the choice or `re.match` means that a `scope` without special tokens
    filters by prefix.

##### Returns:

  The list of values in the collection with the given `name`, or
  an empty list if no value has been added to that collection. The
  list contains the values in the order under which they were
  collected.


- - -

#### `tf.Graph.get_collection_ref(name)` {#Graph.get_collection_ref}

Returns a list of values in the collection with the given `name`.

If the collection exists, this returns the list itself, which can
be modified in place to change the collection.  If the collection does
not exist, it is created as an empty list and the list is returned.

This is different from `get_collection()` which always returns a copy of
the collection list if it exists and never creates an empty collection.

##### Args:


*  <b>`name`</b>: The key for the collection. For example, the `GraphKeys` class
    contains many standard names for collections.

##### Returns:

  The list of values in the collection with the given `name`, or an empty
  list if no value has been added to that collection.



- - -

#### `tf.Graph.as_graph_element(obj, allow_tensor=True, allow_operation=True)` {#Graph.as_graph_element}

Returns the object referred to by `obj`, as an `Operation` or `Tensor`.

This function validates that `obj` represents an element of this
graph, and gives an informative error message if it is not.

This function is the canonical way to get/validate an object of
one of the allowed types from an external argument reference in the
Session API.

This method may be called concurrently from multiple threads.

##### Args:


*  <b>`obj`</b>: A `Tensor`, an `Operation`, or the name of a tensor or operation.
    Can also be any object with an `_as_graph_element()` method that returns
    a value of one of these types.
*  <b>`allow_tensor`</b>: If true, `obj` may refer to a `Tensor`.
*  <b>`allow_operation`</b>: If true, `obj` may refer to an `Operation`.

##### Returns:

  The `Tensor` or `Operation` in the Graph corresponding to `obj`.

##### Raises:


*  <b>`TypeError`</b>: If `obj` is not a type we support attempting to convert
    to types.
*  <b>`ValueError`</b>: If `obj` is of an appropriate type but invalid. For
    example, an invalid string.
*  <b>`KeyError`</b>: If `obj` is not an object in the graph.


- - -

#### `tf.Graph.get_operation_by_name(name)` {#Graph.get_operation_by_name}

Returns the `Operation` with the given `name`.

This method may be called concurrently from multiple threads.

##### Args:


*  <b>`name`</b>: The name of the `Operation` to return.

##### Returns:

  The `Operation` with the given `name`.

##### Raises:


*  <b>`TypeError`</b>: If `name` is not a string.
*  <b>`KeyError`</b>: If `name` does not correspond to an operation in this graph.


- - -

#### `tf.Graph.get_tensor_by_name(name)` {#Graph.get_tensor_by_name}

Returns the `Tensor` with the given `name`.

This method may be called concurrently from multiple threads.

##### Args:


*  <b>`name`</b>: The name of the `Tensor` to return.

##### Returns:

  The `Tensor` with the given `name`.

##### Raises:


*  <b>`TypeError`</b>: If `name` is not a string.
*  <b>`KeyError`</b>: If `name` does not correspond to a tensor in this graph.


- - -

#### `tf.Graph.get_operations()` {#Graph.get_operations}

Return the list of operations in the graph.

You can modify the operations in place, but modifications
to the list such as inserts/delete have no effect on the
list of operations known to the graph.

This method may be called concurrently from multiple threads.

##### Returns:

  A list of Operations.



- - -

#### `tf.Graph.seed` {#Graph.seed}

The graph-level random seed of this graph.


- - -

#### `tf.Graph.unique_name(name, mark_as_used=True)` {#Graph.unique_name}

Return a unique operation name for `name`.

Note: You rarely need to call `unique_name()` directly.  Most of
the time you just need to create `with g.name_scope()` blocks to
generate structured names.

`unique_name` is used to generate structured names, separated by
`"/"`, to help identify operations when debugging a graph.
Operation names are displayed in error messages reported by the
TensorFlow runtime, and in various visualization tools such as
TensorBoard.

If `mark_as_used` is set to `True`, which is the default, a new
unique name is created and marked as in use. If it's set to `False`,
the unique name is returned without actually being marked as used.
This is useful when the caller simply wants to know what the name
to be created will be.

##### Args:


*  <b>`name`</b>: The name for an operation.
*  <b>`mark_as_used`</b>: Whether to mark this name as being used.

##### Returns:

  A string to be passed to `create_op()` that will be used
  to name the operation being created.


- - -

#### `tf.Graph.version` {#Graph.version}

Returns a version number that increases as ops are added to the graph.

Note that this is unrelated to the
[GraphDef version](#Graph.graph_def_version).


- - -

#### `tf.Graph.graph_def_versions` {#Graph.graph_def_versions}

The GraphDef version information of this graph.

For details on the meaning of each version, see
[`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto).

##### Returns:

  A `VersionDef`.



- - -

#### `tf.Graph.create_op(op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_shapes=True, compute_device=True)` {#Graph.create_op}

Creates an `Operation` in this graph.

This is a low-level interface for creating an `Operation`. Most
programs will not call this method directly, and instead use the
Python op constructors, such as `tf.constant()`, which add ops to
the default graph.

##### Args:


*  <b>`op_type`</b>: The `Operation` type to create. This corresponds to the
    `OpDef.name` field for the proto that defines the operation.
*  <b>`inputs`</b>: A list of `Tensor` objects that will be inputs to the `Operation`.
*  <b>`dtypes`</b>: A list of `DType` objects that will be the types of the tensors
    that the operation produces.
*  <b>`input_types`</b>: (Optional.) A list of `DType`s that will be the types of
    the tensors that the operation consumes. By default, uses the base
    `DType` of each input in `inputs`. Operations that expect
    reference-typed inputs must specify `input_types` explicitly.
*  <b>`name`</b>: (Optional.) A string name for the operation. If not specified, a
    name is generated based on `op_type`.
*  <b>`attrs`</b>: (Optional.) A dictionary where the key is the attribute name (a
    string) and the value is the respective `attr` attribute of the
    `NodeDef` proto that will represent the operation (an `AttrValue`
    proto).
*  <b>`op_def`</b>: (Optional.) The `OpDef` proto that describes the `op_type` that
    the operation will have.
*  <b>`compute_shapes`</b>: (Optional.) If True, shape inference will be performed
    to compute the shapes of the outputs.
*  <b>`compute_device`</b>: (Optional.) If True, device functions will be executed
    to compute the device property of the Operation.

##### Raises:


*  <b>`TypeError`</b>: if any of the inputs is not a `Tensor`.
*  <b>`ValueError`</b>: if colocation conflicts with existing device assignment.

##### Returns:

  An `Operation` object.


- - -

#### `tf.Graph.gradient_override_map(op_type_map)` {#Graph.gradient_override_map}

EXPERIMENTAL: A context manager for overriding gradient functions.

This context manager can be used to override the gradient function
that will be used for ops within the scope of the context.

For example:

```python
@tf.RegisterGradient("CustomSquare")
def _custom_square_grad(op, grad):
  # ...

with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  s_1 = tf.square(c)  # Uses the default gradient for tf.square.
  with g.gradient_override_map({"Square": "CustomSquare"}):
    s_2 = tf.square(s_2)  # Uses _custom_square_grad to compute the
                          # gradient of s_2.
```

##### Args:


*  <b>`op_type_map`</b>: A dictionary mapping op type strings to alternative op
    type strings.

##### Returns:

  A context manager that sets the alternative op type to be used for one
  or more ops created in that context.

##### Raises:


*  <b>`TypeError`</b>: If `op_type_map` is not a dictionary mapping strings to
    strings.



#### Other Methods
- - -

#### `tf.Graph.building_function` {#Graph.building_function}

Returns True iff this graph represents a function.


- - -

#### `tf.Graph.clear_collection(name)` {#Graph.clear_collection}

Clears all values in a collection.

##### Args:


*  <b>`name`</b>: The key for the collection. The `GraphKeys` class contains many
    standard names for collections.


- - -

#### `tf.Graph.colocate_with(op, ignore_existing=False)` {#Graph.colocate_with}

Returns a context manager that specifies an op to colocate with.

Note: this function is not for public use, only for internal libraries.

For example:

```python
a = tf.Variable([1.0])
with g.colocate_with(a):
  b = tf.constant(1.0)
  c = tf.add(a, b)
```

`b` and `c` will always be colocated with `a`, no matter where `a`
is eventually placed.

**NOTE** Using a colocation scope resets any existing device constraints.

If `op` is `None` then `ignore_existing` must be `True` and the new
scope resets all colocation and device constraints.

##### Args:


*  <b>`op`</b>: The op to colocate all created ops with, or `None`.
*  <b>`ignore_existing`</b>: If true, only applies colocation of this op within
    the context, rather than applying all colocation properties
    on the stack.  If `op` is `None`, this value must be `True`.

##### Raises:


*  <b>`ValueError`</b>: if op is None but ignore_existing is False.

##### Yields:

  A context manager that specifies the op with which to colocate
  newly created ops.


- - -

#### `tf.Graph.container(container_name)` {#Graph.container}

Returns a context manager that specifies the resource container to use.

Stateful operations, such as variables and queues, can maintain their
states on devices so that they can be shared by multiple processes.
A resource container is a string name under which these stateful
operations are tracked. These resources can be released or cleared
with `tf.Session.reset()`.

For example:

```python
with g.container('experiment0'):
  # All stateful Operations constructed in this context will be placed
  # in resource container "experiment0".
  v1 = tf.Variable([1.0])
  v2 = tf.Variable([2.0])
  with g.container("experiment1"):
    # All stateful Operations constructed in this context will be
    # placed in resource container "experiment1".
    v3 = tf.Variable([3.0])
    q1 = tf.FIFOQueue(10, tf.float32)
  # All stateful Operations constructed in this context will be
  # be created in the "experiment0".
  v4 = tf.Variable([4.0])
  q1 = tf.FIFOQueue(20, tf.float32)
  with g.container(""):
    # All stateful Operations constructed in this context will be
    # be placed in the default resource container.
    v5 = tf.Variable([5.0])
    q3 = tf.FIFOQueue(30, tf.float32)

# Resets container "experiment0", after which the state of v1, v2, v4, q1
# will become undefined (such as uninitialized).
tf.Session.reset(target, ["experiment0"])
```

##### Args:


*  <b>`container_name`</b>: container name string.

##### Returns:

  A context manager for defining resource containers for stateful ops,
    yields the container name.


- - -

#### `tf.Graph.get_all_collection_keys()` {#Graph.get_all_collection_keys}

Returns a list of collections used in this graph.


- - -

#### `tf.Graph.is_feedable(tensor)` {#Graph.is_feedable}

Returns `True` if and only if `tensor` is feedable.


- - -

#### `tf.Graph.is_fetchable(tensor_or_op)` {#Graph.is_fetchable}

Returns `True` if and only if `tensor_or_op` is fetchable.


- - -

#### `tf.Graph.prevent_feeding(tensor)` {#Graph.prevent_feeding}

Marks the given `tensor` as unfeedable in this graph.


- - -

#### `tf.Graph.prevent_fetching(op)` {#Graph.prevent_fetching}

Marks the given `op` as unfetchable in this graph.



- - -

### `class tf.Operation` {#Operation}

Represents a graph node that performs computation on tensors.

An `Operation` is a node in a TensorFlow `Graph` that takes zero or
more `Tensor` objects as input, and produces zero or more `Tensor`
objects as output. Objects of type `Operation` are created by
calling a Python op constructor (such as
[`tf.matmul()`](../../api_docs/python/math_ops.md#matmul))
or [`Graph.create_op()`](../../api_docs/python/framework.md#Graph.create_op).

For example `c = tf.matmul(a, b)` creates an `Operation` of type
"MatMul" that takes tensors `a` and `b` as input, and produces `c`
as output.

After the graph has been launched in a session, an `Operation` can
be executed by passing it to
[`Session.run()`](../../api_docs/python/client.md#Session.run).
`op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.

- - -

#### `tf.Operation.name` {#Operation.name}

The full name of this operation.


- - -

#### `tf.Operation.type` {#Operation.type}

The type of the op (e.g. `"MatMul"`).


- - -

#### `tf.Operation.inputs` {#Operation.inputs}

The list of `Tensor` objects representing the data inputs of this op.


- - -

#### `tf.Operation.control_inputs` {#Operation.control_inputs}

The `Operation` objects on which this op has a control dependency.

Before this op is executed, TensorFlow will ensure that the
operations in `self.control_inputs` have finished executing. This
mechanism can be used to run ops sequentially for performance
reasons, or to ensure that the side effects of an op are observed
in the correct order.

##### Returns:

  A list of `Operation` objects.


- - -

#### `tf.Operation.outputs` {#Operation.outputs}

The list of `Tensor` objects representing the outputs of this op.


- - -

#### `tf.Operation.device` {#Operation.device}

The name of the device to which this op has been assigned, if any.

##### Returns:

  The string name of the device to which this op has been
  assigned, or an empty string if it has not been assigned to a
  device.


- - -

#### `tf.Operation.graph` {#Operation.graph}

The `Graph` that contains this operation.



- - -

#### `tf.Operation.run(feed_dict=None, session=None)` {#Operation.run}

Runs this operation in a `Session`.

Calling this method will execute all preceding operations that
produce the inputs needed for this operation.

*N.B.* Before invoking `Operation.run()`, its graph must have been
launched in a session, and either a default session must be
available, or `session` must be specified explicitly.

##### Args:


*  <b>`feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    See [`Session.run()`](../../api_docs/python/client.md#Session.run)
    for a description of the valid feed values.
*  <b>`session`</b>: (Optional.) The `Session` to be used to run to this operation. If
    none, the default session will be used.



- - -

#### `tf.Operation.get_attr(name)` {#Operation.get_attr}

Returns the value of the attr of this op with the given `name`.

##### Args:


*  <b>`name`</b>: The name of the attr to fetch.

##### Returns:

  The value of the attr, as a Python object.

##### Raises:


*  <b>`ValueError`</b>: If this op does not have an attr with the given `name`.


- - -

#### `tf.Operation.traceback` {#Operation.traceback}

Returns the call stack from when this operation was constructed.



#### Other Methods
- - -

#### `tf.Operation.__init__(node_def, g, inputs=None, output_types=None, control_inputs=None, input_types=None, original_op=None, op_def=None)` {#Operation.__init__}

Creates an `Operation`.

NOTE: This constructor validates the name of the `Operation` (passed
as `node_def.name`). Valid `Operation` names match the following
regular expression:

    [A-Za-z0-9.][A-Za-z0-9_.\\-/]*

##### Args:


*  <b>`node_def`</b>: `node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`.
    Used for attributes of `node_def_pb2.NodeDef`, typically `name`,
    `op`, and `device`.  The `input` attribute is irrelevant here
    as it will be computed when generating the model.
*  <b>`g`</b>: `Graph`. The parent graph.
*  <b>`inputs`</b>: list of `Tensor` objects. The inputs to this `Operation`.
*  <b>`output_types`</b>: list of `DType` objects.  List of the types of the
    `Tensors` computed by this operation.  The length of this list indicates
    the number of output endpoints of the `Operation`.
*  <b>`control_inputs`</b>: list of operations or tensors from which to have a
    control dependency.
*  <b>`input_types`</b>: List of `DType` objects representing the
    types of the tensors accepted by the `Operation`.  By default
    uses `[x.dtype.base_dtype for x in inputs]`.  Operations that expect
    reference-typed inputs must specify these explicitly.
*  <b>`original_op`</b>: Optional. Used to associate the new `Operation` with an
    existing `Operation` (for example, a replica with the op that was
    replicated).
*  <b>`op_def`</b>: Optional. The `op_def_pb2.OpDef` proto that describes the
    op type that this `Operation` represents.

##### Raises:


*  <b>`TypeError`</b>: if control inputs are not Operations or Tensors,
    or if `node_def` is not a `NodeDef`,
    or if `g` is not a `Graph`,
    or if `inputs` are not tensors,
    or if `inputs` and `input_types` are incompatible.
*  <b>`ValueError`</b>: if the `node_def` name is not valid.


- - -

#### `tf.Operation.__repr__()` {#Operation.__repr__}




- - -

#### `tf.Operation.__str__()` {#Operation.__str__}




- - -

#### `tf.Operation.colocation_groups()` {#Operation.colocation_groups}

Returns the list of colocation groups of the op.


- - -

#### `tf.Operation.node_def` {#Operation.node_def}

Returns a serialized `NodeDef` representation of this operation.

##### Returns:

  A
  [`NodeDef`](https://www.tensorflow.org/code/tensorflow/core/framework/node_def.proto)
  protocol buffer.


- - -

#### `tf.Operation.op_def` {#Operation.op_def}

Returns the `OpDef` proto that represents the type of this op.

##### Returns:

  An
  [`OpDef`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto)
  protocol buffer.


- - -

#### `tf.Operation.values()` {#Operation.values}

DEPRECATED: Use outputs.



- - -

### `class tf.Tensor` {#Tensor}

Represents one of the outputs of an `Operation`.

A `Tensor` is a symbolic handle to one of the outputs of an
`Operation`. It does not hold the values of that operation's output,
but instead provides a means of computing those values in a
TensorFlow [`Session`](../../api_docs/python/client.md#Session).

This class has two primary purposes:

1. A `Tensor` can be passed as an input to another `Operation`.
   This builds a dataflow connection between operations, which
   enables TensorFlow to execute an entire `Graph` that represents a
   large, multi-step computation.

2. After the graph has been launched in a session, the value of the
   `Tensor` can be computed by passing it to
   [`Session.run()`](../../api_docs/python/client.md#Session.run).
   `t.eval()` is a shortcut for calling
   `tf.get_default_session().run(t)`.

In the following example, `c`, `d`, and `e` are symbolic `Tensor`
objects, whereas `result` is a numpy array that stores a concrete
value:

```python
# Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(e)
```

- - -

#### `tf.Tensor.dtype` {#Tensor.dtype}

The `DType` of elements in this tensor.


- - -

#### `tf.Tensor.name` {#Tensor.name}

The string name of this tensor.


- - -

#### `tf.Tensor.value_index` {#Tensor.value_index}

The index of this tensor in the outputs of its `Operation`.


- - -

#### `tf.Tensor.graph` {#Tensor.graph}

The `Graph` that contains this tensor.


- - -

#### `tf.Tensor.op` {#Tensor.op}

The `Operation` that produces this tensor as an output.


- - -

#### `tf.Tensor.consumers()` {#Tensor.consumers}

Returns a list of `Operation`s that consume this tensor.

##### Returns:

  A list of `Operation`s.



- - -

#### `tf.Tensor.eval(feed_dict=None, session=None)` {#Tensor.eval}

Evaluates this tensor in a `Session`.

Calling this method will execute all preceding operations that
produce the inputs needed for the operation that produces this
tensor.

*N.B.* Before invoking `Tensor.eval()`, its graph must have been
launched in a session, and either a default session must be
available, or `session` must be specified explicitly.

##### Args:


*  <b>`feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    See [`Session.run()`](../../api_docs/python/client.md#Session.run) for a
    description of the valid feed values.
*  <b>`session`</b>: (Optional.) The `Session` to be used to evaluate this tensor. If
    none, the default session will be used.

##### Returns:

  A numpy array corresponding to the value of this tensor.



- - -

#### `tf.Tensor.get_shape()` {#Tensor.get_shape}

Alias of Tensor.shape.


- - -

#### `tf.Tensor.shape` {#Tensor.shape}

Returns the `TensorShape` that represents the shape of this tensor.

The shape is computed using shape inference functions that are
registered in the Op for each `Operation`.  See
[`TensorShape`](../../api_docs/python/framework.md#TensorShape)
for more details of what a shape represents.

The inferred shape of a tensor is used to provide shape
information without having to launch the graph in a session. This
can be used for debugging, and providing early error messages. For
example:

```python
c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(c.shape)
==> TensorShape([Dimension(2), Dimension(3)])

d = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

print(d.shape)
==> TensorShape([Dimension(4), Dimension(2)])

# Raises a ValueError, because `c` and `d` do not have compatible
# inner dimensions.
e = tf.matmul(c, d)

f = tf.matmul(c, d, transpose_a=True, transpose_b=True)

print(f.shape)
==> TensorShape([Dimension(3), Dimension(4)])
```

In some cases, the inferred shape may have unknown dimensions. If
the caller has additional information about the values of these
dimensions, `Tensor.set_shape()` can be used to augment the
inferred shape.

##### Returns:

  A `TensorShape` representing the shape of this tensor.


- - -

#### `tf.Tensor.set_shape(shape)` {#Tensor.set_shape}

Updates the shape of this tensor.

This method can be called multiple times, and will merge the given
`shape` with the current shape of this tensor. It can be used to
provide additional information about the shape of this tensor that
cannot be inferred from the graph alone. For example, this can be used
to provide additional information about the shapes of images:

```python
_, image_data = tf.TFRecordReader(...).read(...)
image = tf.image.decode_png(image_data, channels=3)

# The height and width dimensions of `image` are data dependent, and
# cannot be computed without executing the op.
print(image.shape)
==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])

# We know that each image in this dataset is 28 x 28 pixels.
image.set_shape([28, 28, 3])
print(image.shape)
==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])
```

##### Args:


*  <b>`shape`</b>: A `TensorShape` representing the shape of this tensor.

##### Raises:


*  <b>`ValueError`</b>: If `shape` is not compatible with the current shape of
    this tensor.



#### Other Methods
- - -

#### `tf.Tensor.__abs__(x, name=None)` {#Tensor.__abs__}

Computes the absolute value of a tensor.

Given a tensor of real numbers `x`, this operation returns a tensor
containing the absolute value of each element in `x`. For example, if x is
an input element and y is an output element, this operation computes
\\(y = |x|\\).

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor` of type `float32`, `float64`, `int32`, or
    `int64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` the same size and type as `x` with absolute
    values.


- - -

#### `tf.Tensor.__add__(x, y)` {#Tensor.__add__}

Returns x + y element-wise.

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.


- - -

#### `tf.Tensor.__and__(x, y)` {#Tensor.__and__}

Returns the truth value of x AND y element-wise.

*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`y`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__bool__()` {#Tensor.__bool__}

Dummy method to prevent a tensor from being used as a Python `bool`.

This overload raises a `TypeError` when the user inadvertently
treats a `Tensor` as a boolean (e.g. in an `if` statement). For
example:

```python
if tf.constant(True):  # Will raise.
  # ...

if tf.constant(5) < tf.constant(7):  # Will raise.
  # ...
```

This disallows ambiguities between testing the Python value vs testing the
dynamic condition of the `Tensor`.

##### Raises:

  `TypeError`.


- - -

#### `tf.Tensor.__div__(x, y)` {#Tensor.__div__}

Divide two values using Python 2 semantics. Used for Tensor.__div__.

##### Args:


*  <b>`x`</b>: `Tensor` numerator of real numeric type.
*  <b>`y`</b>: `Tensor` denominator of real numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `x / y` returns the quotient of x and y.


- - -

#### `tf.Tensor.__eq__(other)` {#Tensor.__eq__}




- - -

#### `tf.Tensor.__floordiv__(x, y)` {#Tensor.__floordiv__}

Divides `x / y` elementwise, rounding toward the most negative integer.

The same as `tf.div(x,y)` for integers, but uses `tf.floor(tf.div(x,y))` for
floating point arguments so that the result is always an integer (though
possibly an integer represented as floating point).  This op is generated by
`x // y` floor division in Python 3 and in Python 2.7 with
`from __future__ import division`.

Note that for efficiency, `floordiv` uses C semantics for negative numbers
(unlike Python and Numpy).

`x` and `y` must have the same type, and the result will have the same type
as well.

##### Args:


*  <b>`x`</b>: `Tensor` numerator of real numeric type.
*  <b>`y`</b>: `Tensor` denominator of real numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `x / y` rounded down (except possibly towards zero for negative integers).

##### Raises:


*  <b>`TypeError`</b>: If the inputs are complex.


- - -

#### `tf.Tensor.__ge__(x, y, name=None)` {#Tensor.__ge__}

Returns the truth value of (x >= y) element-wise.

*NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__getitem__(tensor, slice_spec, var=None)` {#Tensor.__getitem__}

Overload for Tensor.__getitem__.

This operation extracts the specified region from the tensor.
The notation is similar to NumPy with the restriction that
currently only support basic indexing. That means that
using a tensor as input is not currently allowed

Some useful examples:

```python
# strip leading and trailing 2 elements
foo = tf.constant([1,2,3,4,5,6])
print(foo[2:-2].eval()) # => [3,4]

# skip every row and reverse every column
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[::2,::-1].eval()) # => [[3,2,1], [9,8,7]]

# Insert another dimension
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[tf.newaxis, :, :].eval()) # => [[[3,2,1], [9,8,7]]]
print(foo[:, tf.newaxis, :].eval()) # => [[[3,2,1]], [[9,8,7]]]
print(foo[:, :, tf.newaxis].eval()) # => [[[3],[2],[1]], [[9],[8],[7]]]

# Ellipses (3 equivalent operations)
print(foo[tf.newaxis, :, :].eval()) # => [[[3,2,1], [9,8,7]]]
print(foo[tf.newaxis, ...].eval()) # => [[[3,2,1], [9,8,7]]]
print(foo[tf.newaxis].eval()) # => [[[3,2,1], [9,8,7]]]
```

##### Notes:

  - `tf.newaxis` is `None` as in NumPy.
  - An implicit ellipsis is placed at the end of the `slice_spec`
  - NumPy advanced indexing is currently not supported.

##### Args:


*  <b>`tensor`</b>: An ops.Tensor object.
*  <b>`slice_spec`</b>: The arguments to Tensor.__getitem__.
*  <b>`var`</b>: In the case of variable slice assignment, the Variable
    object to slice (i.e. tensor is the read-only view of this
    variable).

##### Returns:

  The appropriate slice of "tensor", based on "slice_spec".

##### Raises:


*  <b>`ValueError`</b>: If a slice range is negative size.
*  <b>`TypeError`</b>: If the slice indices aren't int, slice, or Ellipsis.


- - -

#### `tf.Tensor.__gt__(x, y, name=None)` {#Tensor.__gt__}

Returns the truth value of (x > y) element-wise.

*NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__hash__()` {#Tensor.__hash__}




- - -

#### `tf.Tensor.__init__(op, value_index, dtype)` {#Tensor.__init__}

Creates a new `Tensor`.

##### Args:


*  <b>`op`</b>: An `Operation`. `Operation` that computes this tensor.
*  <b>`value_index`</b>: An `int`. Index of the operation's endpoint that produces
    this tensor.
*  <b>`dtype`</b>: A `DType`. Type of elements stored in this tensor.

##### Raises:


*  <b>`TypeError`</b>: If the op is not an `Operation`.


- - -

#### `tf.Tensor.__invert__(x, name=None)` {#Tensor.__invert__}

Returns the truth value of NOT x element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__iter__()` {#Tensor.__iter__}

Dummy method to prevent iteration. Do not call.

NOTE(mrry): If we register __getitem__ as an overloaded operator,
Python will valiantly attempt to iterate over the Tensor from 0 to
infinity.  Declaring this method prevents this unintended
behavior.

##### Raises:


*  <b>`TypeError`</b>: when invoked.


- - -

#### `tf.Tensor.__le__(x, y, name=None)` {#Tensor.__le__}

Returns the truth value of (x <= y) element-wise.

*NOTE*: `LessEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__lt__(x, y, name=None)` {#Tensor.__lt__}

Returns the truth value of (x < y) element-wise.

*NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__mod__(x, y)` {#Tensor.__mod__}

Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: `FloorMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.


- - -

#### `tf.Tensor.__mul__(x, y)` {#Tensor.__mul__}

Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse".


- - -

#### `tf.Tensor.__neg__(x, name=None)` {#Tensor.__neg__}

Computes numerical negative value element-wise.

I.e., \\(y = -x\\).

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.


- - -

#### `tf.Tensor.__nonzero__()` {#Tensor.__nonzero__}

Dummy method to prevent a tensor from being used as a Python `bool`.

This is the Python 2.x counterpart to `__bool__()` above.

##### Raises:

  `TypeError`.


- - -

#### `tf.Tensor.__or__(x, y)` {#Tensor.__or__}

Returns the truth value of x OR y element-wise.

*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`y`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__pow__(x, y)` {#Tensor.__pow__}

Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```
# tensor 'x' is [[2, 2], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
```

##### Args:


*  <b>`x`</b>: A `Tensor` of type `float32`, `float64`, `int32`, `int64`, `complex64`,
   or `complex128`.
*  <b>`y`</b>: A `Tensor` of type `float32`, `float64`, `int32`, `int64`, `complex64`,
   or `complex128`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`.


- - -

#### `tf.Tensor.__radd__(y, x)` {#Tensor.__radd__}

Returns x + y element-wise.

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.


- - -

#### `tf.Tensor.__rand__(y, x)` {#Tensor.__rand__}

Returns the truth value of x AND y element-wise.

*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`y`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__rdiv__(y, x)` {#Tensor.__rdiv__}

Divide two values using Python 2 semantics. Used for Tensor.__div__.

##### Args:


*  <b>`x`</b>: `Tensor` numerator of real numeric type.
*  <b>`y`</b>: `Tensor` denominator of real numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `x / y` returns the quotient of x and y.


- - -

#### `tf.Tensor.__repr__()` {#Tensor.__repr__}




- - -

#### `tf.Tensor.__rfloordiv__(y, x)` {#Tensor.__rfloordiv__}

Divides `x / y` elementwise, rounding toward the most negative integer.

The same as `tf.div(x,y)` for integers, but uses `tf.floor(tf.div(x,y))` for
floating point arguments so that the result is always an integer (though
possibly an integer represented as floating point).  This op is generated by
`x // y` floor division in Python 3 and in Python 2.7 with
`from __future__ import division`.

Note that for efficiency, `floordiv` uses C semantics for negative numbers
(unlike Python and Numpy).

`x` and `y` must have the same type, and the result will have the same type
as well.

##### Args:


*  <b>`x`</b>: `Tensor` numerator of real numeric type.
*  <b>`y`</b>: `Tensor` denominator of real numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `x / y` rounded down (except possibly towards zero for negative integers).

##### Raises:


*  <b>`TypeError`</b>: If the inputs are complex.


- - -

#### `tf.Tensor.__rmod__(y, x)` {#Tensor.__rmod__}

Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: `FloorMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.


- - -

#### `tf.Tensor.__rmul__(y, x)` {#Tensor.__rmul__}

Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse".


- - -

#### `tf.Tensor.__ror__(y, x)` {#Tensor.__ror__}

Returns the truth value of x OR y element-wise.

*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`y`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Tensor.__rpow__(y, x)` {#Tensor.__rpow__}

Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```
# tensor 'x' is [[2, 2], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
```

##### Args:


*  <b>`x`</b>: A `Tensor` of type `float32`, `float64`, `int32`, `int64`, `complex64`,
   or `complex128`.
*  <b>`y`</b>: A `Tensor` of type `float32`, `float64`, `int32`, `int64`, `complex64`,
   or `complex128`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`.


- - -

#### `tf.Tensor.__rsub__(y, x)` {#Tensor.__rsub__}

Returns x - y element-wise.

*NOTE*: `Sub` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.


- - -

#### `tf.Tensor.__rtruediv__(y, x)` {#Tensor.__rtruediv__}




- - -

#### `tf.Tensor.__rxor__(y, x)` {#Tensor.__rxor__}

x ^ y = (x | y) & ~(x & y).


- - -

#### `tf.Tensor.__str__()` {#Tensor.__str__}




- - -

#### `tf.Tensor.__sub__(x, y)` {#Tensor.__sub__}

Returns x - y element-wise.

*NOTE*: `Sub` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.


- - -

#### `tf.Tensor.__truediv__(x, y)` {#Tensor.__truediv__}




- - -

#### `tf.Tensor.__xor__(x, y)` {#Tensor.__xor__}

x ^ y = (x | y) & ~(x & y).


- - -

#### `tf.Tensor.device` {#Tensor.device}

The name of the device on which this tensor will be produced, or None.




## Tensor types

- - -

### `class tf.DType` {#DType}

Represents the type of the elements in a `Tensor`.

The following `DType` objects are defined:

* `tf.float16`: 16-bit half-precision floating-point.
* `tf.float32`: 32-bit single-precision floating-point.
* `tf.float64`: 64-bit double-precision floating-point.
* `tf.bfloat16`: 16-bit truncated floating-point.
* `tf.complex64`: 64-bit single-precision complex.
* `tf.complex128`: 128-bit double-precision complex.
* `tf.int8`: 8-bit signed integer.
* `tf.uint8`: 8-bit unsigned integer.
* `tf.uint16`: 16-bit unsigned integer.
* `tf.int16`: 16-bit signed integer.
* `tf.int32`: 32-bit signed integer.
* `tf.int64`: 64-bit signed integer.
* `tf.bool`: Boolean.
* `tf.string`: String.
* `tf.qint8`: Quantized 8-bit signed integer.
* `tf.quint8`: Quantized 8-bit unsigned integer.
* `tf.qint16`: Quantized 16-bit signed integer.
* `tf.quint16`: Quantized 16-bit unsigned integer.
* `tf.qint32`: Quantized 32-bit signed integer.
* `tf.resource`: Handle to a mutable resource.

In addition, variants of these types with the `_ref` suffix are
defined for reference-typed tensors.

The `tf.as_dtype()` function converts numpy types and string type
names to a `DType` object.

- - -

#### `tf.DType.is_compatible_with(other)` {#DType.is_compatible_with}

Returns True if the `other` DType will be converted to this DType.

The conversion rules are as follows:

```python
DType(T)       .is_compatible_with(DType(T))        == True
DType(T)       .is_compatible_with(DType(T).as_ref) == True
DType(T).as_ref.is_compatible_with(DType(T))        == False
DType(T).as_ref.is_compatible_with(DType(T).as_ref) == True
```

##### Args:


*  <b>`other`</b>: A `DType` (or object that may be converted to a `DType`).

##### Returns:

  True if a Tensor of the `other` `DType` will be implicitly converted to
  this `DType`.


- - -

#### `tf.DType.name` {#DType.name}

Returns the string name for this `DType`.


- - -

#### `tf.DType.base_dtype` {#DType.base_dtype}

Returns a non-reference `DType` based on this `DType`.


- - -

#### `tf.DType.real_dtype` {#DType.real_dtype}

Returns the dtype correspond to this dtype's real part.


- - -

#### `tf.DType.is_bool` {#DType.is_bool}

Returns whether this is a boolean data type


- - -

#### `tf.DType.is_floating` {#DType.is_floating}

Returns whether this is a (non-quantized, real) floating point type.


- - -

#### `tf.DType.is_complex` {#DType.is_complex}

Returns whether this is a complex floating point type.


- - -

#### `tf.DType.is_integer` {#DType.is_integer}

Returns whether this is a (non-quantized) integer type.


- - -

#### `tf.DType.is_quantized` {#DType.is_quantized}

Returns whether this is a quantized data type.


- - -

#### `tf.DType.is_unsigned` {#DType.is_unsigned}

Returns whether this type is unsigned.

Non-numeric, unordered, and quantized types are not considered unsigned, and
this function returns `False`.

##### Returns:

  Whether a `DType` is unsigned.



- - -

#### `tf.DType.as_numpy_dtype` {#DType.as_numpy_dtype}

Returns a `numpy.dtype` based on this `DType`.


- - -

#### `tf.DType.as_datatype_enum` {#DType.as_datatype_enum}

Returns a `types_pb2.DataType` enum value based on this `DType`.



- - -

#### `tf.DType.limits` {#DType.limits}

Return intensity limits, i.e. (min, max) tuple, of the dtype.

##### Args:

  clip_negative : bool, optional
      If True, clip the negative range (i.e. return 0 for min intensity)
      even if the image dtype allows negative values.
Returns
  min, max : tuple
    Lower and upper intensity limits.



#### Other Methods
- - -

#### `tf.DType.__eq__(other)` {#DType.__eq__}

Returns True iff this DType refers to the same type as `other`.


- - -

#### `tf.DType.__hash__()` {#DType.__hash__}




- - -

#### `tf.DType.__init__(type_enum)` {#DType.__init__}

Creates a new `DataType`.

NOTE(mrry): In normal circumstances, you should not need to
construct a `DataType` object directly. Instead, use the
`tf.as_dtype()` function.

##### Args:


*  <b>`type_enum`</b>: A `types_pb2.DataType` enum value.

##### Raises:


*  <b>`TypeError`</b>: If `type_enum` is not a value `types_pb2.DataType`.


- - -

#### `tf.DType.__ne__(other)` {#DType.__ne__}

Returns True iff self != other.


- - -

#### `tf.DType.__repr__()` {#DType.__repr__}




- - -

#### `tf.DType.__str__()` {#DType.__str__}




- - -

#### `tf.DType.is_numpy_compatible` {#DType.is_numpy_compatible}




- - -

#### `tf.DType.max` {#DType.max}

Returns the maximum representable value in this data type.

##### Raises:


*  <b>`TypeError`</b>: if this is a non-numeric, unordered, or quantized type.


- - -

#### `tf.DType.min` {#DType.min}

Returns the minimum representable value in this data type.

##### Raises:


*  <b>`TypeError`</b>: if this is a non-numeric, unordered, or quantized type.


- - -

#### `tf.DType.size` {#DType.size}





- - -

### `tf.as_dtype(type_value)` {#as_dtype}

Converts the given `type_value` to a `DType`.

##### Args:


*  <b>`type_value`</b>: A value that can be converted to a `tf.DType`
    object. This may currently be a `tf.DType` object, a
    [`DataType` enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
    a string type name, or a `numpy.dtype`.

##### Returns:

  A `DType` corresponding to `type_value`.

##### Raises:


*  <b>`TypeError`</b>: If `type_value` cannot be converted to a `DType`.



## Utility functions

- - -

### `tf.device(device_name_or_function)` {#device}

Wrapper for `Graph.device()` using the default graph.

See
[`Graph.device()`](../../api_docs/python/framework.md#Graph.device)
for more details.

##### Args:


*  <b>`device_name_or_function`</b>: The device name or function to use in
    the context.

##### Returns:

  A context manager that specifies the default device to use for newly
  created ops.


- - -

### `tf.container(container_name)` {#container}

Wrapper for `Graph.container()` using the default graph.

##### Args:


*  <b>`container_name`</b>: The container string to use in the context.

##### Returns:

  A context manager that specifies the default container to use for newly
  created stateful ops.


- - -

### `tf.name_scope(name, default_name=None, values=None)` {#name_scope}

Returns a context manager for use when defining a Python op.

This context manager validates that the given `values` are from the
same graph, makes that graph the default graph, and pushes a
name scope in that graph (see
[`Graph.name_scope()`](../../api_docs/python/framework.md#Graph.name_scope)
for more details on that).

For example, to define a new Python op called `my_op`:

```python
def my_op(a, b, c, name=None):
  with tf.name_scope(name, "MyOp", [a, b, c]) as scope:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    c = tf.convert_to_tensor(c, name="c")
    # Define some computation that uses `a`, `b`, and `c`.
    return foo_op(..., name=scope)
```

##### Args:


*  <b>`name`</b>: The name argument that is passed to the op function.
*  <b>`default_name`</b>: The default name to use if the `name` argument is `None`.
*  <b>`values`</b>: The list of `Tensor` arguments that are passed to the op function.

##### Returns:

  A context manager for use in defining Python ops. Yields the name scope.

##### Raises:


*  <b>`ValueError`</b>: if neither `name` nor `default_name` is provided
    but `values` are.


- - -

### `tf.control_dependencies(control_inputs)` {#control_dependencies}

Wrapper for `Graph.control_dependencies()` using the default graph.

See [`Graph.control_dependencies()`](../../api_docs/python/framework.md#Graph.control_dependencies)
for more details.

##### Args:


*  <b>`control_inputs`</b>: A list of `Operation` or `Tensor` objects which
    must be executed or computed before running the operations
    defined in the context.  Can also be `None` to clear the control
    dependencies.

##### Returns:

 A context manager that specifies control dependencies for all
 operations constructed within the context.


- - -

### `tf.convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None)` {#convert_to_tensor}

Converts the given `value` to a `Tensor`.

This function converts Python objects of various types to `Tensor`
objects. It accepts `Tensor` objects, numpy arrays, Python lists,
and Python scalars. For example:

```python
import numpy as np

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
```

This function can be useful when composing a new operation in Python
(such as `my_func` in the example above). All standard Python op
constructors apply this function to each of their Tensor-valued
inputs, which allows those ops to accept numpy arrays, Python lists,
and scalars in addition to `Tensor` objects.

##### Args:


*  <b>`value`</b>: An object whose type has a registered `Tensor` conversion function.
*  <b>`dtype`</b>: Optional element type for the returned tensor. If missing, the
    type is inferred from the type of `value`.
*  <b>`name`</b>: Optional name to use if a new `Tensor` is created.
*  <b>`preferred_dtype`</b>: Optional element type for the returned tensor,
    used when dtype is None. In some cases, a caller may not have a
    dtype in mind when converting to a tensor, so preferred_dtype
    can be used as a soft preference.  If the conversion to
    `preferred_dtype` is not possible, this argument has no effect.

##### Returns:

  An `Output` based on `value`.

##### Raises:


*  <b>`TypeError`</b>: If no conversion function is registered for `value`.
*  <b>`RuntimeError`</b>: If a registered conversion function returns an invalid value.


- - -

### `tf.convert_to_tensor_or_indexed_slices(value, dtype=None, name=None)` {#convert_to_tensor_or_indexed_slices}

Converts the given object to a `Tensor` or an `IndexedSlices`.

If `value` is an `IndexedSlices` or `SparseTensor` it is returned
unmodified. Otherwise, it is converted to a `Tensor` using
`convert_to_tensor()`.

##### Args:


*  <b>`value`</b>: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
    by `convert_to_tensor()`.
*  <b>`dtype`</b>: (Optional.) The required `DType` of the returned `Tensor` or
    `IndexedSlices`.
*  <b>`name`</b>: (Optional.) A name to use if a new `Tensor` is created.

##### Returns:

  An `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

##### Raises:


*  <b>`ValueError`</b>: If `dtype` does not match the element type of `value`.


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

### `tf.get_default_graph()` {#get_default_graph}

Returns the default graph for the current thread.

The returned graph will be the innermost graph on which a
`Graph.as_default()` context has been entered, or a global default
graph if none has been explicitly created.

NOTE: The default graph is a property of the current thread. If you
create a new thread, and wish to use the default graph in that
thread, you must explicitly add a `with g.as_default():` in that
thread's function.

##### Returns:

  The default `Graph` being used in the current thread.


- - -

### `tf.reset_default_graph()` {#reset_default_graph}

Clears the default graph stack and resets the global default graph.

NOTE: The default graph is a property of the current thread. This
function applies only to the current thread.  Calling this function while
a `tf.Session` or `tf.InteractiveSession` is active will result in undefined
behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
after calling this function will result in undefined behavior.


- - -

### `tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None, producer_op_list=None)` {#import_graph_def}

Imports the TensorFlow graph in `graph_def` into the Python `Graph`.

This function provides a way to import a serialized TensorFlow
[`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
protocol buffer, and extract individual objects in the `GraphDef` as
[`Tensor`](#Tensor) and [`Operation`](#Operation) objects. See
[`Graph.as_graph_def()`](#Graph.as_graph_def) for a way to create a
`GraphDef` proto.

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` proto containing operations to be imported into
    the default graph.
*  <b>`input_map`</b>: A dictionary mapping input names (as strings) in `graph_def`
    to `Tensor` objects. The values of the named input tensors in the
    imported graph will be re-mapped to the respective `Tensor` values.
*  <b>`return_elements`</b>: A list of strings containing operation names in
    `graph_def` that will be returned as `Operation` objects; and/or
    tensor names in `graph_def` that will be returned as `Tensor` objects.
*  <b>`name`</b>: (Optional.) A prefix that will be prepended to the names in
    `graph_def`. Defaults to `"import"`.
*  <b>`op_dict`</b>: (Optional.) A dictionary mapping op type names to `OpDef` protos.
    Must contain an `OpDef` proto for each op type named in `graph_def`.
    If omitted, uses the `OpDef` protos registered in the global registry.
*  <b>`producer_op_list`</b>: (Optional.) An `OpList` proto with the (possibly stripped)
    list of `OpDef`s used by the producer of the graph. If provided, attrs
    for ops in `graph_def` that are not in `op_dict` that have their default
    value according to `producer_op_list` will be removed. This will allow
    some more `GraphDef`s produced by later binaries to be accepted by
    earlier binaries.

##### Returns:

  A list of `Operation` and/or `Tensor` objects from the imported graph,
  corresponding to the names in `return_elements`.

##### Raises:


*  <b>`TypeError`</b>: If `graph_def` is not a `GraphDef` proto,
    `input_map` is not a dictionary mapping strings to `Tensor` objects,
    or `return_elements` is not a list of strings.
*  <b>`ValueError`</b>: If `input_map`, or `return_elements` contains names that
    do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
    it refers to an unknown tensor).


- - -

### `tf.load_file_system_library(library_filename)` {#load_file_system_library}

Loads a TensorFlow plugin, containing file system implementation.

Pass `library_filename` to a platform-specific mechanism for dynamically
loading a library. The rules for determining the exact location of the
library are platform-specific and are not documented here.

##### Args:


*  <b>`library_filename`</b>: Path to the plugin.
    Relative or absolute filesystem path to a dynamic library file.

##### Returns:

  None.

##### Raises:


*  <b>`RuntimeError`</b>: when unable to load the library.


- - -

### `tf.load_op_library(library_filename)` {#load_op_library}

Loads a TensorFlow plugin, containing custom ops and kernels.

Pass "library_filename" to a platform-specific mechanism for dynamically
loading a library. The rules for determining the exact location of the
library are platform-specific and are not documented here. When the
library is loaded, ops and kernels registered in the library via the
`REGISTER_*` macros are made available in the TensorFlow process. Note
that ops with the same name as an existing op are rejected and not
registered with the process.

##### Args:


*  <b>`library_filename`</b>: Path to the plugin.
    Relative or absolute filesystem path to a dynamic library file.

##### Returns:

  A python module containing the Python wrappers for Ops defined in
  the plugin.

##### Raises:


*  <b>`RuntimeError`</b>: when unable to load the library or get the python wrappers.



## Graph collections

- - -

### `tf.add_to_collection(name, value)` {#add_to_collection}

Wrapper for `Graph.add_to_collection()` using the default graph.

See [`Graph.add_to_collection()`](../../api_docs/python/framework.md#Graph.add_to_collection)
for more details.

##### Args:


*  <b>`name`</b>: The key for the collection. For example, the `GraphKeys` class
    contains many standard names for collections.
*  <b>`value`</b>: The value to add to the collection.


- - -

### `tf.get_collection(key, scope=None)` {#get_collection}

Wrapper for `Graph.get_collection()` using the default graph.

See [`Graph.get_collection()`](../../api_docs/python/framework.md#Graph.get_collection)
for more details.

##### Args:


*  <b>`key`</b>: The key for the collection. For example, the `GraphKeys` class
    contains many standard names for collections.
*  <b>`scope`</b>: (Optional.) If supplied, the resulting list is filtered to include
    only items whose `name` attribute matches using `re.match`. Items
    without a `name` attribute are never returned if a scope is supplied and
    the choice or `re.match` means that a `scope` without special tokens
    filters by prefix.

##### Returns:

  The list of values in the collection with the given `name`, or
  an empty list if no value has been added to that collection. The
  list contains the values in the order under which they were
  collected.


- - -

### `tf.get_collection_ref(key)` {#get_collection_ref}

Wrapper for `Graph.get_collection_ref()` using the default graph.

See [`Graph.get_collection_ref()`](../../api_docs/python/framework.md#Graph.get_collection_ref)
for more details.

##### Args:


*  <b>`key`</b>: The key for the collection. For example, the `GraphKeys` class
    contains many standard names for collections.

##### Returns:

  The list of values in the collection with the given `name`, or an empty
  list if no value has been added to that collection.  Note that this returns
  the collection list itself, which can be modified in place to change the
  collection.


- - -

### `class tf.GraphKeys` {#GraphKeys}

Standard names to use for graph collections.

The standard library uses various well-known names to collect and
retrieve values associated with a graph. For example, the
`tf.Optimizer` subclasses default to optimizing the variables
collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
specified, but it is also possible to pass an explicit list of
variables.

The following standard keys are defined:

* `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
  across distributed environment (model variables are subset of these). See
  [`tf.global_variables()`](../../api_docs/python/state_ops.md#global_variables)
  for more details.
  Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
  and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
* `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
  machine. Usually used for temporarily variables, like counters.
  Note: use `tf.contrib.framework.local_variable` to add to this collection.
* `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
  model for inference (feed forward). Note: use
  `tf.contrib.framework.model_variable` to add to this collection.
* `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
  be trained by an optimizer. See
  [`tf.trainable_variables()`](../../api_docs/python/state_ops.md#trainable_variables)
  for more details.
* `SUMMARIES`: the summary `Tensor` objects that have been created in the
  graph. See
  [`tf.summary.merge_all()`](../../api_docs/python/summary.md#merge_all)
  for more details.
* `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
  produce input for a computation. See
  [`tf.start_queue_runners()`](../../api_docs/python/train.md#start_queue_runners)
  for more details.
* `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
  keep moving averages.  See
  [`tf.moving_average_variables()`](../../api_docs/python/state_ops.md#moving_average_variables)
  for more details.
* `REGULARIZATION_LOSSES`: regularization losses collected during graph
  construction.
* `WEIGHTS`: weights inside neural network layers
* `BIASES`: biases inside neural network layers
* `ACTIVATIONS`: activations of neural network layers


## Defining new operations

- - -

### `class tf.RegisterGradient` {#RegisterGradient}

A decorator for registering the gradient function for an op type.

This decorator is only used when defining a new op type. For an op
with `m` inputs and `n` outputs, the gradient function is a function
that takes the original `Operation` and `n` `Tensor` objects
(representing the gradients with respect to each output of the op),
and returns `m` `Tensor` objects (representing the partial gradients
with respect to each input of the op).

For example, assuming that operations of type `"Sub"` take two
inputs `x` and `y`, and return a single output `x - y`, the
following gradient function would be registered:

```python
@tf.RegisterGradient("Sub")
def _sub_grad(unused_op, grad):
  return grad, tf.negative(grad)
```

The decorator argument `op_type` is the string type of an
operation. This corresponds to the `OpDef.name` field for the proto
that defines the operation.

- - -

#### `tf.RegisterGradient.__init__(op_type)` {#RegisterGradient.__init__}

Creates a new decorator with `op_type` as the Operation type.

##### Args:


*  <b>`op_type`</b>: The string type of an operation. This corresponds to the
    `OpDef.name` field for the proto that defines the operation.



#### Other Methods
- - -

#### `tf.RegisterGradient.__call__(f)` {#RegisterGradient.__call__}

Registers the function `f` as gradient function for `op_type`.



- - -

### `tf.NotDifferentiable(op_type)` {#NotDifferentiable}

Specifies that ops of type `op_type` is not differentiable.

This function should *not* be used for operations that have a
well-defined gradient that is not yet implemented.

This function is only used when defining a new op type. It may be
used for ops such as `tf.size()` that are not differentiable.  For
example:

```python
tf.NotDifferentiable("Size")
```

The gradient computed for 'op_type' will then propagate zeros.

For ops that have a well-defined gradient but are not yet implemented,
no declaration should be made, and an error *must* be thrown if
an attempt to request its gradient is made.

##### Args:


*  <b>`op_type`</b>: The string type of an operation. This corresponds to the
    `OpDef.name` field for the proto that defines the operation.

##### Raises:


*  <b>`TypeError`</b>: If `op_type` is not a string.


- - -

### `tf.NoGradient(op_type)` {#NoGradient}

Specifies that ops of type `op_type` is not differentiable.

This function should *not* be used for operations that have a
well-defined gradient that is not yet implemented.

This function is only used when defining a new op type. It may be
used for ops such as `tf.size()` that are not differentiable.  For
example:

```python
tf.NotDifferentiable("Size")
```

The gradient computed for 'op_type' will then propagate zeros.

For ops that have a well-defined gradient but are not yet implemented,
no declaration should be made, and an error *must* be thrown if
an attempt to request its gradient is made.

##### Args:


*  <b>`op_type`</b>: The string type of an operation. This corresponds to the
    `OpDef.name` field for the proto that defines the operation.

##### Raises:


*  <b>`TypeError`</b>: If `op_type` is not a string.


- - -

### `class tf.TensorShape` {#TensorShape}

Represents the shape of a `Tensor`.

A `TensorShape` represents a possibly-partial shape specification for a
`Tensor`. It may be one of the following:

* *Fully-known shape:* has a known number of dimensions and a known size
  for each dimension.
* *Partially-known shape:* has a known number of dimensions, and an unknown
  size for one or more dimension.
* *Unknown shape:* has an unknown number of dimensions, and an unknown
  size in all dimensions.

If a tensor is produced by an operation of type `"Foo"`, its shape
may be inferred if there is a registered shape function for
`"Foo"`. See [`Shape functions in
C++`](../../how_tos/adding_an_op/index.md#shape-functions-in-c) for
details of shape functions and how to register them. Alternatively,
the shape may be set explicitly using
[`Tensor.set_shape()`](../../api_docs/python/framework.md#Tensor.set_shape).

- - -

#### `tf.TensorShape.merge_with(other)` {#TensorShape.merge_with}

Returns a `TensorShape` combining the information in `self` and `other`.

The dimensions in `self` and `other` are merged elementwise,
according to the rules defined for `Dimension.merge_with()`.

##### Args:


*  <b>`other`</b>: Another `TensorShape`.

##### Returns:

  A `TensorShape` containing the combined information of `self` and
  `other`.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` are not compatible.


- - -

#### `tf.TensorShape.concatenate(other)` {#TensorShape.concatenate}

Returns the concatenation of the dimension in `self` and `other`.

*N.B.* If either `self` or `other` is completely unknown,
concatenation will discard information about the other shape. In
future, we might support concatenation that preserves this
information for use with slicing.

##### Args:


*  <b>`other`</b>: Another `TensorShape`.

##### Returns:

  A `TensorShape` whose dimensions are the concatenation of the
  dimensions in `self` and `other`.



- - -

#### `tf.TensorShape.ndims` {#TensorShape.ndims}

Returns the rank of this shape, or None if it is unspecified.


- - -

#### `tf.TensorShape.dims` {#TensorShape.dims}

Returns a list of Dimensions, or None if the shape is unspecified.


- - -

#### `tf.TensorShape.as_list()` {#TensorShape.as_list}

Returns a list of integers or `None` for each dimension.

##### Returns:

  A list of integers or `None` for each dimension.

##### Raises:


*  <b>`ValueError`</b>: If `self` is an unknown shape with an unknown rank.


- - -

#### `tf.TensorShape.as_proto()` {#TensorShape.as_proto}

Returns this shape as a `TensorShapeProto`.


- - -

#### `tf.TensorShape.is_compatible_with(other)` {#TensorShape.is_compatible_with}

Returns True iff `self` is compatible with `other`.

Two possibly-partially-defined shapes are compatible if there
exists a fully-defined shape that both shapes can represent. Thus,
compatibility allows the shape inference code to reason about
partially-defined shapes. For example:

* TensorShape(None) is compatible with all shapes.

* TensorShape([None, None]) is compatible with all two-dimensional
  shapes, such as TensorShape([32, 784]), and also TensorShape(None). It is
  not compatible with, for example, TensorShape([None]) or
  TensorShape([None, None, None]).

* TensorShape([32, None]) is compatible with all two-dimensional shapes
  with size 32 in the 0th dimension, and also TensorShape([None, None])
  and TensorShape(None). It is not compatible with, for example,
  TensorShape([32]), TensorShape([32, None, 1]) or TensorShape([64, None]).

* TensorShape([32, 784]) is compatible with itself, and also
  TensorShape([32, None]), TensorShape([None, 784]), TensorShape([None,
  None]) and TensorShape(None). It is not compatible with, for example,
  TensorShape([32, 1, 784]) or TensorShape([None]).

The compatibility relation is reflexive and symmetric, but not
transitive. For example, TensorShape([32, 784]) is compatible with
TensorShape(None), and TensorShape(None) is compatible with
TensorShape([4, 4]), but TensorShape([32, 784]) is not compatible with
TensorShape([4, 4]).

##### Args:


*  <b>`other`</b>: Another TensorShape.

##### Returns:

  True iff `self` is compatible with `other`.


- - -

#### `tf.TensorShape.is_fully_defined()` {#TensorShape.is_fully_defined}

Returns True iff `self` is fully defined in every dimension.



- - -

#### `tf.TensorShape.with_rank(rank)` {#TensorShape.with_rank}

Returns a shape based on `self` with the given rank.

This method promotes a completely unknown shape to one with a
known rank.

##### Args:


*  <b>`rank`</b>: An integer.

##### Returns:

  A shape that is at least as specific as `self` with the given rank.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not represent a shape with the given `rank`.


- - -

#### `tf.TensorShape.with_rank_at_least(rank)` {#TensorShape.with_rank_at_least}

Returns a shape based on `self` with at least the given rank.

##### Args:


*  <b>`rank`</b>: An integer.

##### Returns:

  A shape that is at least as specific as `self` with at least the given
  rank.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not represent a shape with at least the given
    `rank`.


- - -

#### `tf.TensorShape.with_rank_at_most(rank)` {#TensorShape.with_rank_at_most}

Returns a shape based on `self` with at most the given rank.

##### Args:


*  <b>`rank`</b>: An integer.

##### Returns:

  A shape that is at least as specific as `self` with at most the given
  rank.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not represent a shape with at most the given
    `rank`.



- - -

#### `tf.TensorShape.assert_has_rank(rank)` {#TensorShape.assert_has_rank}

Raises an exception if `self` is not compatible with the given `rank`.

##### Args:


*  <b>`rank`</b>: An integer.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not represent a shape with the given `rank`.


- - -

#### `tf.TensorShape.assert_same_rank(other)` {#TensorShape.assert_same_rank}

Raises an exception if `self` and `other` do not have compatible ranks.

##### Args:


*  <b>`other`</b>: Another `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` do not represent shapes with the
    same rank.


- - -

#### `tf.TensorShape.assert_is_compatible_with(other)` {#TensorShape.assert_is_compatible_with}

Raises exception if `self` and `other` do not represent the same shape.

This method can be used to assert that there exists a shape that both
`self` and `other` represent.

##### Args:


*  <b>`other`</b>: Another TensorShape.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` do not represent the same shape.


- - -

#### `tf.TensorShape.assert_is_fully_defined()` {#TensorShape.assert_is_fully_defined}

Raises an exception if `self` is not fully defined in every dimension.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not have a known value for every dimension.



#### Other Methods
- - -

#### `tf.TensorShape.__bool__()` {#TensorShape.__bool__}

Returns True if this shape contains non-zero information.


- - -

#### `tf.TensorShape.__eq__(other)` {#TensorShape.__eq__}

Returns True if `self` is equivalent to `other`.


- - -

#### `tf.TensorShape.__getitem__(key)` {#TensorShape.__getitem__}

Returns the value of a dimension or a shape, depending on the key.

##### Args:


*  <b>`key`</b>: If `key` is an integer, returns the dimension at that index;
    otherwise if `key` is a slice, returns a TensorShape whose
    dimensions are those selected by the slice from `self`.

##### Returns:

  A dimension if `key` is an integer, or a `TensorShape` if `key` is a
  slice.

##### Raises:


*  <b>`ValueError`</b>: If `key` is a slice, and any of its elements are negative, or
    if `self` is completely unknown and the step is set.


- - -

#### `tf.TensorShape.__init__(dims)` {#TensorShape.__init__}

Creates a new TensorShape with the given dimensions.

##### Args:


*  <b>`dims`</b>: A list of Dimensions, or None if the shape is unspecified.
*  <b>`DEPRECATED`</b>: A single integer is treated as a singleton list.

##### Raises:


*  <b>`TypeError`</b>: If dims cannot be converted to a list of dimensions.


- - -

#### `tf.TensorShape.__iter__()` {#TensorShape.__iter__}

Returns `self.dims` if the rank is known, otherwise raises ValueError.


- - -

#### `tf.TensorShape.__len__()` {#TensorShape.__len__}

Returns the rank of this shape, or raises ValueError if unspecified.


- - -

#### `tf.TensorShape.__ne__(other)` {#TensorShape.__ne__}

Returns True if `self` is known to be different from `other`.


- - -

#### `tf.TensorShape.__nonzero__()` {#TensorShape.__nonzero__}

Returns True if this shape contains non-zero information.


- - -

#### `tf.TensorShape.__repr__()` {#TensorShape.__repr__}




- - -

#### `tf.TensorShape.__str__()` {#TensorShape.__str__}




- - -

#### `tf.TensorShape.num_elements()` {#TensorShape.num_elements}

Returns the total number of elements, or none for incomplete shapes.



- - -

### `class tf.Dimension` {#Dimension}

Represents the value of one dimension in a TensorShape.
- - -

#### `tf.Dimension.__add__(other)` {#Dimension.__add__}

Returns the sum of `self` and `other`.

Dimensions are summed as follows:

  Dimension(m)    + Dimension(n)    == Dimension(m + n)
  Dimension(m)    + Dimension(None) == Dimension(None)
  Dimension(None) + Dimension(n)    == Dimension(None)
  Dimension(None) + Dimension(None) == Dimension(None)

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension whose value is the sum of `self` and `other`.


- - -

#### `tf.Dimension.__div__(other)` {#Dimension.__div__}

DEPRECATED: Use `__floordiv__` via `x // y` instead.

This function exists only for backwards compatibility purposes; new code
should use `__floordiv__` via the syntax `x // y`.  Using `x // y`
communicates clearly that the result rounds down, and is forward compatible
to Python 3.

##### Args:


*  <b>`other`</b>: Another `Dimension`.

##### Returns:

  A `Dimension` whose value is the integer quotient of `self` and `other`.


- - -

#### `tf.Dimension.__eq__(other)` {#Dimension.__eq__}

Returns true if `other` has the same known value as this Dimension.


- - -

#### `tf.Dimension.__floordiv__(other)` {#Dimension.__floordiv__}

Returns the quotient of `self` and `other` rounded down.

Dimensions are divided as follows:

  Dimension(m)    // Dimension(n)    == Dimension(m // n)
  Dimension(m)    // Dimension(None) == Dimension(None)
  Dimension(None) // Dimension(n)    == Dimension(None)
  Dimension(None) // Dimension(None) == Dimension(None)

##### Args:


*  <b>`other`</b>: Another `Dimension`.

##### Returns:

  A `Dimension` whose value is the integer quotient of `self` and `other`.


- - -

#### `tf.Dimension.__ge__(other)` {#Dimension.__ge__}

Returns True if `self` is known to be greater than or equal to `other`.

Dimensions are compared as follows:

  Dimension(m)    >= Dimension(n)    == m >= n
  Dimension(m)    >= Dimension(None) == None
  Dimension(None) >= Dimension(n)    == None
  Dimension(None) >= Dimension(None) == None

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  The value of `self.value >= other.value` if both are known, otherwise
  None.


- - -

#### `tf.Dimension.__gt__(other)` {#Dimension.__gt__}

Returns True if `self` is known to be greater than `other`.

Dimensions are compared as follows:

  Dimension(m)    > Dimension(n)    == m > n
  Dimension(m)    > Dimension(None) == None
  Dimension(None) > Dimension(n)    == None
  Dimension(None) > Dimension(None) == None

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  The value of `self.value > other.value` if both are known, otherwise
  None.


- - -

#### `tf.Dimension.__index__()` {#Dimension.__index__}




- - -

#### `tf.Dimension.__init__(value)` {#Dimension.__init__}

Creates a new Dimension with the given value.


- - -

#### `tf.Dimension.__int__()` {#Dimension.__int__}




- - -

#### `tf.Dimension.__le__(other)` {#Dimension.__le__}

Returns True if `self` is known to be less than or equal to `other`.

Dimensions are compared as follows:

  Dimension(m)    <= Dimension(n)    == m <= n
  Dimension(m)    <= Dimension(None) == None
  Dimension(None) <= Dimension(n)    == None
  Dimension(None) <= Dimension(None) == None

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  The value of `self.value <= other.value` if both are known, otherwise
  None.


- - -

#### `tf.Dimension.__lt__(other)` {#Dimension.__lt__}

Returns True if `self` is known to be less than `other`.

Dimensions are compared as follows:

  Dimension(m)    < Dimension(n)    == m < n
  Dimension(m)    < Dimension(None) == None
  Dimension(None) < Dimension(n)    == None
  Dimension(None) < Dimension(None) == None

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  The value of `self.value < other.value` if both are known, otherwise
  None.


- - -

#### `tf.Dimension.__mod__(other)` {#Dimension.__mod__}

Returns `self` modulo `other.

Dimension moduli are computed  as follows:

  Dimension(m)    % Dimension(n)     == Dimension(m % n)
  Dimension(m)    % Dimension(None)  == Dimension(None)
  Dimension(None) % Dimension(n)     == Dimension(None)
  Dimension(None) %  Dimension(None) == Dimension(None)

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension whose value is `self` modulo `other`.


- - -

#### `tf.Dimension.__mul__(other)` {#Dimension.__mul__}

Returns the product of `self` and `other`.

Dimensions are summed as follows:

```
  Dimension(m)    * Dimension(n)    == Dimension(m * n)
  Dimension(m)    * Dimension(None) == Dimension(None)
  Dimension(None) * Dimension(n)    == Dimension(None)
  Dimension(None) * Dimension(None) == Dimension(None)
```

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension whose value is the product of `self` and `other`.


- - -

#### `tf.Dimension.__ne__(other)` {#Dimension.__ne__}

Returns true if `other` has a different known value from `self`.


- - -

#### `tf.Dimension.__repr__()` {#Dimension.__repr__}




- - -

#### `tf.Dimension.__str__()` {#Dimension.__str__}




- - -

#### `tf.Dimension.__sub__(other)` {#Dimension.__sub__}

Returns the subtraction of `other` from `self`.

Dimensions are subtracted as follows:

  Dimension(m)    - Dimension(n)    == Dimension(m - n)
  Dimension(m)    - Dimension(None) == Dimension(None)
  Dimension(None) - Dimension(n)    == Dimension(None)
  Dimension(None) - Dimension(None) == Dimension(None)

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension whose value is the subtraction of sum of `other` from `self`.


- - -

#### `tf.Dimension.assert_is_compatible_with(other)` {#Dimension.assert_is_compatible_with}

Raises an exception if `other` is not compatible with this Dimension.

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` are not compatible (see
    is_compatible_with).


- - -

#### `tf.Dimension.is_compatible_with(other)` {#Dimension.is_compatible_with}

Returns true if `other` is compatible with this Dimension.

Two known Dimensions are compatible if they have the same value.
An unknown Dimension is compatible with all other Dimensions.

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  True if this Dimension and `other` are compatible.


- - -

#### `tf.Dimension.merge_with(other)` {#Dimension.merge_with}

Returns a Dimension that combines the information in `self` and `other`.

Dimensions are combined as follows:

```python
    Dimension(n)   .merge_with(Dimension(n))    == Dimension(n)
    Dimension(n)   .merge_with(Dimension(None)) == Dimension(n)
    Dimension(None).merge_with(Dimension(n))    == Dimension(n)
    Dimension(None).merge_with(Dimension(None)) == Dimension(None)
    Dimension(n)   .merge_with(Dimension(m)) raises ValueError for n != m
```

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension containing the combined information of `self` and
  `other`.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` are not compatible (see
    is_compatible_with).


- - -

#### `tf.Dimension.value` {#Dimension.value}

The value of this dimension, or None if it is unknown.



- - -

### `tf.op_scope(values, name, default_name=None)` {#op_scope}

DEPRECATED. Same as name_scope above, just different argument order.


- - -

### `tf.get_seed(op_seed)` {#get_seed}

Returns the local seeds an operation should use given an op-specific seed.

Given operation-specific seed, `op_seed`, this helper function returns two
seeds derived from graph-level and op-level seeds. Many random operations
internally use the two seeds to allow user to change the seed globally for a
graph, or for only specific operations.

For details on how the graph-level seed interacts with op seeds, see
@{set_random_seed}.

##### Args:


*  <b>`op_seed`</b>: integer.

##### Returns:

  A tuple of two integers that should be used for the local seed of this
  operation.



## For libraries building on TensorFlow

- - -

### `tf.register_tensor_conversion_function(base_type, conversion_func, priority=100)` {#register_tensor_conversion_function}

Registers a function for converting objects of `base_type` to `Tensor`.

The conversion function must have the following signature:

```python
    def conversion_func(value, dtype=None, name=None, as_ref=False):
      # ...
```

It must return a `Tensor` with the given `dtype` if specified. If the
conversion function creates a new `Tensor`, it should use the given
`name` if specified. All exceptions will be propagated to the caller.

The conversion function may return `NotImplemented` for some
inputs. In this case, the conversion process will continue to try
subsequent conversion functions.

If `as_ref` is true, the function must return a `Tensor` reference,
such as a `Variable`.

NOTE: The conversion functions will execute in order of priority,
followed by order of registration. To ensure that a conversion function
`F` runs before another conversion function `G`, ensure that `F` is
registered with a smaller priority than `G`.

##### Args:


*  <b>`base_type`</b>: The base type or tuple of base types for all objects that
    `conversion_func` accepts.
*  <b>`conversion_func`</b>: A function that converts instances of `base_type` to
    `Tensor`.
*  <b>`priority`</b>: Optional integer that indicates the priority for applying this
    conversion function. Conversion functions with smaller priority values
    run earlier than conversion functions with larger priority values.
    Defaults to 100.

##### Raises:


*  <b>`TypeError`</b>: If the arguments do not have the appropriate type.



## Other Functions and Classes
- - -

### `class tf.DeviceSpec` {#DeviceSpec}

Represents a (possibly partial) specification for a TensorFlow device.

`DeviceSpec`s are used throughout TensorFlow to describe where state is stored
and computations occur. Using `DeviceSpec` allows you to parse device spec
strings to verify their validity, merge them or compose them programmatically.

Example:

```python
# Place the operations on device "GPU:0" in the "ps" job.
device_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
with tf.device(device_spec):
  # Both my_var and squared_var will be placed on /job:ps/device:GPU:0.
  my_var = tf.Variable(..., name="my_variable")
  squared_var = tf.square(my_var)
```

If a `DeviceSpec` is partially specified, it will be merged with other
`DeviceSpec`s according to the scope in which it is defined. `DeviceSpec`
components defined in inner scopes take precedence over those defined in
outer scopes.

```python
with tf.device(DeviceSpec(job="train", )):
  with tf.device(DeviceSpec(job="ps", device_type="GPU", device_index=0):
    # Nodes created here will be assigned to /job:ps/device:GPU:0.
  with tf.device(DeviceSpec(device_type="GPU", device_index=1):
    # Nodes created here will be assigned to /job:train/device:GPU:1.
```

A `DeviceSpec` consists of 5 components -- each of
which is optionally specified:

* Job: The job name.
* Replica: The replica index.
* Task: The task index.
* Device type: The device type string (e.g. "CPU" or "GPU").
* Device index: The device index.
- - -

#### `tf.DeviceSpec.__init__(job=None, replica=None, task=None, device_type=None, device_index=None)` {#DeviceSpec.__init__}

Create a new `DeviceSpec` object.

##### Args:


*  <b>`job`</b>: string.  Optional job name.
*  <b>`replica`</b>: int.  Optional replica index.
*  <b>`task`</b>: int.  Optional task index.
*  <b>`device_type`</b>: Optional device type string (e.g. "CPU" or "GPU")
*  <b>`device_index`</b>: int.  Optional device index.  If left
    unspecified, device represents 'any' device_index.


- - -

#### `tf.DeviceSpec.from_string(spec)` {#DeviceSpec.from_string}

Construct a `DeviceSpec` from a string.

##### Args:


*  <b>`spec`</b>: a string of the form
   /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
  or
   /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
  as cpu and gpu are mutually exclusive.
  All entries are optional.

##### Returns:

  A DeviceSpec.


- - -

#### `tf.DeviceSpec.job` {#DeviceSpec.job}




- - -

#### `tf.DeviceSpec.merge_from(dev)` {#DeviceSpec.merge_from}

Merge the properties of "dev" into this `DeviceSpec`.

##### Args:


*  <b>`dev`</b>: a `DeviceSpec`.


- - -

#### `tf.DeviceSpec.parse_from_string(spec)` {#DeviceSpec.parse_from_string}

Parse a `DeviceSpec` name into its components.

##### Args:


*  <b>`spec`</b>: a string of the form
   /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
  or
   /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
  as cpu and gpu are mutually exclusive.
  All entries are optional.

##### Returns:

  The `DeviceSpec`.

##### Raises:


*  <b>`ValueError`</b>: if the spec was not valid.


- - -

#### `tf.DeviceSpec.replica` {#DeviceSpec.replica}




- - -

#### `tf.DeviceSpec.task` {#DeviceSpec.task}




- - -

#### `tf.DeviceSpec.to_string()` {#DeviceSpec.to_string}

Return a string representation of this `DeviceSpec`.

##### Returns:

  a string of the form
  /job:<name>/replica:<id>/task:<id>/device:<device_type>:<id>.



