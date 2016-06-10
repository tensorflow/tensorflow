See the [Variables How To](../../how_tos/variables/index.md) for a high
level overview.

A variable maintains state in the graph across calls to `run()`. You add a
variable to the graph by constructing an instance of the class `Variable`.

The `Variable()` constructor requires an initial value for the variable,
which can be a `Tensor` of any type and shape. The initial value defines the
type and shape of the variable. After construction, the type and shape of
the variable are fixed. The value can be changed using one of the assign
methods.

If you want to change the shape of a variable later you have to use an
`assign` Op with `validate_shape=False`.

Just like any `Tensor`, variables created with `Variable()` can be used as
inputs for other Ops in the graph. Additionally, all the operators
overloaded for the `Tensor` class are carried over to variables, so you can
also add nodes to the graph by just doing arithmetic on variables.

```python
import tensorflow as tf

# Create a variable.
w = tf.Variable(<initial-value>, name=<optional-name>)

# Use the variable in the graph like any Tensor.
y = tf.matmul(w, ...another variable or tensor...)

# The overloaded operators are available too.
z = tf.sigmoid(w + y)

# Assign a new value to the variable with `assign()` or a related method.
w.assign(w + 1.0)
w.assign_add(1.0)
```

When you launch the graph, variables have to be explicitly initialized before
you can run Ops that use their value. You can initialize a variable by
running its *initializer op*, restoring the variable from a save file, or
simply running an `assign` Op that assigns a value to the variable. In fact,
the variable *initializer op* is just an `assign` Op that assigns the
variable's initial value to the variable itself.

```python
# Launch the graph in a session.
with tf.Session() as sess:
    # Run the variable initializer.
    sess.run(w.initializer)
    # ...you now can run ops that use the value of 'w'...
```

The most common initialization pattern is to use the convenience function
`initialize_all_variables()` to add an Op to the graph that initializes
all the variables. You then run that Op after launching the graph.

```python
# Add an Op to initialize all variables.
init_op = tf.initialize_all_variables()

# Launch the graph in a session.
with tf.Session() as sess:
    # Run the Op that initializes all variables.
    sess.run(init_op)
    # ...you can now run any Op that uses variable values...
```

If you need to create a variable with an initial value dependent on another
variable, use the other variable's `initialized_value()`. This ensures that
variables are initialized in the right order.

All variables are automatically collected in the graph where they are
created. By default, the constructor adds the new variable to the graph
collection `GraphKeys.VARIABLES`. The convenience function
`all_variables()` returns the contents of that collection.

When building a machine learning model it is often convenient to distinguish
betwen variables holding the trainable model parameters and other variables
such as a `global step` variable used to count training steps. To make this
easier, the variable constructor supports a `trainable=<bool>` parameter. If
`True`, the new variable is also added to the graph collection
`GraphKeys.TRAINABLE_VARIABLES`. The convenience function
`trainable_variables()` returns the contents of this collection. The
various `Optimizer` classes use this collection as the default list of
variables to optimize.


Creating a variable.

- - -

#### `tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None)` {#Variable.__init__}

Creates a new variable with value `initial_value`.

The new variable is added to the graph collections listed in `collections`,
which defaults to `[GraphKeys.VARIABLES]`.

If `trainable` is `True` the variable is also added to the graph collection
`GraphKeys.TRAINABLE_VARIABLES`.

This constructor creates both a `variable` Op and an `assign` Op to set the
variable to its initial value.

##### Args:


*  <b>`initial_value`</b>: A `Tensor`, or Python object convertible to a `Tensor`,
    which is the initial value for the Variable. The initial value must have
    a shape specified unless `validate_shape` is set to False. Can also be a
    callable with no argument that returns the initial value when called. In
    that case, `dtype` must be specified. (Note that initializer functions
    from init_ops.py must first be bound to a shape before being used here.)
*  <b>`trainable`</b>: If `True`, the default, also adds the variable to the graph
    collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
    the default list of variables to use by the `Optimizer` classes.
*  <b>`collections`</b>: List of graph collections keys. The new variable is added to
    these collections. Defaults to `[GraphKeys.VARIABLES]`.
*  <b>`validate_shape`</b>: If `False`, allows the variable to be initialized with a
    value of unknown shape. If `True`, the default, the shape of
    `initial_value` must be known.
*  <b>`caching_device`</b>: Optional device string describing where the Variable
    should be cached for reading.  Defaults to the Variable's device.
    If not `None`, caches on another device.  Typical use is to cache
    on the device where the Ops using the Variable reside, to deduplicate
    copying through `Switch` and other conditional statements.
*  <b>`name`</b>: Optional name for the variable. Defaults to `'Variable'` and gets
    uniquified automatically.
*  <b>`variable_def`</b>: `VariableDef` protocol buffer. If not `None`, recreates
    the Variable object with its contents. `variable_def` and the other
    arguments are mutually exclusive.
*  <b>`dtype`</b>: If set, initial_value will be converted to the given type.
    If `None`, either the datatype will be kept (if `initial_value` is
    a Tensor), or `convert_to_tensor` will decide.

##### Returns:

  A Variable.

##### Raises:


*  <b>`ValueError`</b>: If both `variable_def` and initial_value are specified.
*  <b>`ValueError`</b>: If the initial value is not specified, or does not have a
    shape and `validate_shape` is `True`.


- - -

#### `tf.Variable.initialized_value()` {#Variable.initialized_value}

Returns the value of the initialized variable.

You should use this instead of the variable itself to initialize another
variable with a value that depends on the value of this variable.

```python
# Initialize 'v' with a random tensor.
v = tf.Variable(tf.truncated_normal([10, 40]))
# Use `initialized_value` to guarantee that `v` has been
# initialized before its value is used to initialize `w`.
# The random values are picked only once.
w = tf.Variable(v.initialized_value() * 2.0)
```

##### Returns:

  A `Tensor` holding the value of this variable after its initializer
  has run.



Changing a variable value.

- - -

#### `tf.Variable.assign(value, use_locking=False)` {#Variable.assign}

Assigns a new value to the variable.

This is essentially a shortcut for `assign(self, value)`.

##### Args:


*  <b>`value`</b>: A `Tensor`. The new value for this variable.
*  <b>`use_locking`</b>: If `True`, use locking during the assignment.

##### Returns:

  A `Tensor` that will hold the new value of this variable after
  the assignment has completed.


- - -

#### `tf.Variable.assign_add(delta, use_locking=False)` {#Variable.assign_add}

Adds a value to this variable.

 This is essentially a shortcut for `assign_add(self, delta)`.

##### Args:


*  <b>`delta`</b>: A `Tensor`. The value to add to this variable.
*  <b>`use_locking`</b>: If `True`, use locking during the operation.

##### Returns:

  A `Tensor` that will hold the new value of this variable after
  the addition has completed.


- - -

#### `tf.Variable.assign_sub(delta, use_locking=False)` {#Variable.assign_sub}

Subtracts a value from this variable.

This is essentially a shortcut for `assign_sub(self, delta)`.

##### Args:


*  <b>`delta`</b>: A `Tensor`. The value to subtract from this variable.
*  <b>`use_locking`</b>: If `True`, use locking during the operation.

##### Returns:

  A `Tensor` that will hold the new value of this variable after
  the subtraction has completed.


- - -

#### `tf.Variable.scatter_sub(sparse_delta, use_locking=False)` {#Variable.scatter_sub}

Subtracts `IndexedSlices` from this variable.

This is essentially a shortcut for `scatter_sub(self, sparse_delta.indices,
sparse_delta.values)`.

##### Args:


*  <b>`sparse_delta`</b>: `IndexedSlices` to be subtracted from this variable.
*  <b>`use_locking`</b>: If `True`, use locking during the operation.

##### Returns:

  A `Tensor` that will hold the new value of this variable after
  the scattered subtraction has completed.

##### Raises:


*  <b>`ValueError`</b>: if `sparse_delta` is not an `IndexedSlices`.


- - -

#### `tf.Variable.count_up_to(limit)` {#Variable.count_up_to}

Increments this variable until it reaches `limit`.

When that Op is run it tries to increment the variable by `1`. If
incrementing the variable would bring it above `limit` then the Op raises
the exception `OutOfRangeError`.

If no error is raised, the Op outputs the value of the variable before
the increment.

This is essentially a shortcut for `count_up_to(self, limit)`.

##### Args:


*  <b>`limit`</b>: value at which incrementing the variable raises an error.

##### Returns:

  A `Tensor` that will hold the variable value before the increment. If no
  other Op modifies this variable, the values produced will all be
  distinct.



- - -

#### `tf.Variable.eval(session=None)` {#Variable.eval}

In a session, computes and returns the value of this variable.

This is not a graph construction method, it does not add ops to the graph.

This convenience method requires a session where the graph containing this
variable has been launched. If no session is passed, the default session is
used.  See the [Session class](../../api_docs/python/client.md#Session) for
more information on launching a graph and on sessions.

```python
v = tf.Variable([1, 2])
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # Usage passing the session explicitly.
    print(v.eval(sess))
    # Usage with the default session.  The 'with' block
    # above makes 'sess' the default session.
    print(v.eval())
```

##### Args:


*  <b>`session`</b>: The session to use to evaluate this variable. If
    none, the default session is used.

##### Returns:

  A numpy `ndarray` with a copy of the value of this variable.



Properties.

- - -

#### `tf.Variable.name` {#Variable.name}

The name of this variable.


- - -

#### `tf.Variable.dtype` {#Variable.dtype}

The `DType` of this variable.


- - -

#### `tf.Variable.get_shape()` {#Variable.get_shape}

The `TensorShape` of this variable.

##### Returns:

  A `TensorShape`.


- - -

#### `tf.Variable.device` {#Variable.device}

The device of this variable.


- - -

#### `tf.Variable.initializer` {#Variable.initializer}

The initializer operation for this variable.


- - -

#### `tf.Variable.graph` {#Variable.graph}

The `Graph` of this variable.


- - -

#### `tf.Variable.op` {#Variable.op}

The `Operation` of this variable.



#### Other Methods
- - -

#### `tf.Variable.from_proto(variable_def)` {#Variable.from_proto}

Returns a `Variable` object created from `variable_def`.


- - -

#### `tf.Variable.initial_value` {#Variable.initial_value}

Returns the Tensor used as the initial value for the variable.

Note that this is different from `initialized_value()` which runs
the op that initializes the variable before returning its value.
This method returns the tensor that is used by the op that initializes
the variable.

##### Returns:

  A `Tensor`.


- - -

#### `tf.Variable.ref()` {#Variable.ref}

Returns a reference to this variable.

You usually do not need to call this method as all ops that need a reference
to the variable call it automatically.

Returns is a `Tensor` which holds a reference to the variable.  You can
assign a new value to the variable by passing the tensor to an assign op.
See [`value()`](#Variable.value) if you want to get the value of the
variable.

##### Returns:

  A `Tensor` that is a reference to the variable.


- - -

#### `tf.Variable.to_proto()` {#Variable.to_proto}

Converts a `Variable` to a `VariableDef` protocol buffer.

##### Returns:

  A `VariableDef` protocol buffer.


- - -

#### `tf.Variable.value()` {#Variable.value}

Returns the last snapshot of this variable.

You usually do not need to call this method as all ops that need the value
of the variable call it automatically through a `convert_to_tensor()` call.

Returns a `Tensor` which holds the value of the variable.  You can not
assign a new value to this tensor as it is not a reference to the variable.
See [`ref()`](#Variable.ref) if you want to get a reference to the
variable.

To avoid copies, if the consumer of the returned value is on the same device
as the variable, this actually returns the live value of the variable, not
a copy.  Updates to the variable are seen by the consumer.  If the consumer
is on a different device it will get a copy of the variable.

##### Returns:

  A `Tensor` containing the value of the variable.


