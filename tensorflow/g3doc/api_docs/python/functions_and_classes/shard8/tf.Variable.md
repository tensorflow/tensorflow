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
`global_variables_initializer()` to add an Op to the graph that initializes
all the variables. You then run that Op after launching the graph.

```python
# Add an Op to initialize global variables.
init_op = tf.global_variables_initializer()

# Launch the graph in a session.
with tf.Session() as sess:
    # Run the Op that initializes global variables.
    sess.run(init_op)
    # ...you can now run any Op that uses variable values...
```

If you need to create a variable with an initial value dependent on another
variable, use the other variable's `initialized_value()`. This ensures that
variables are initialized in the right order.

All variables are automatically collected in the graph where they are
created. By default, the constructor adds the new variable to the graph
collection `GraphKeys.GLOBAL_VARIABLES`. The convenience function
`global_variables()` returns the contents of that collection.

When building a machine learning model it is often convenient to distinguish
between variables holding the trainable model parameters and other variables
such as a `global step` variable used to count training steps. To make this
easier, the variable constructor supports a `trainable=<bool>` parameter. If
`True`, the new variable is also added to the graph collection
`GraphKeys.TRAINABLE_VARIABLES`. The convenience function
`trainable_variables()` returns the contents of this collection. The
various `Optimizer` classes use this collection as the default list of
variables to optimize.


Creating a variable.

- - -

#### `tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)` {#Variable.__init__}

Creates a new variable with value `initial_value`.

The new variable is added to the graph collections listed in `collections`,
which defaults to `[GraphKeys.GLOBAL_VARIABLES]`.

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
    these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
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
*  <b>`expected_shape`</b>: A TensorShape. If set, initial_value is expected
    to have this shape.
*  <b>`import_scope`</b>: Optional `string`. Name scope to add to the
    `Variable.` Only used when initializing from protocol buffer.

##### Raises:


*  <b>`ValueError`</b>: If both `variable_def` and initial_value are specified.
*  <b>`ValueError`</b>: If the initial value is not specified, or does not have a
    shape and `validate_shape` is `True`.


- - -

#### `tf.Variable.initialized_value()` {#Variable.initialized_value}

Returns the value of the initialized variable.

You should use this instead of the variable itself to initialize another
variable with a value that depends on the value of this variable.

Beware of using initialized_value except during initialization:
initialized_value causes the Variable's initializer op to be run, so running
this op resets the variable to the initial value.

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
init = tf.global_variables_initializer()

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

#### `tf.Variable.__abs__(a, *args)` {#Variable.__abs__}

Computes the absolute value of a tensor.

Given a tensor of real numbers `x`, this operation returns a tensor
containing the absolute value of each element in `x`. For example, if x is
an input element and y is an output element, this operation computes
\\(y = |x|\\).

See [`tf.complex_abs()`](#tf_complex_abs) to compute the absolute value of a
complex
number.

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor` of type `float32`, `float64`, `int32`, or
    `int64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` the same size and type as `x` with absolute
    values.


- - -

#### `tf.Variable.__add__(a, *args)` {#Variable.__add__}

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

#### `tf.Variable.__and__(a, *args)` {#Variable.__and__}

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

#### `tf.Variable.__div__(a, *args)` {#Variable.__div__}

Divide two values using Python 2 semantics. Used for Tensor.__div__.

##### Args:


*  <b>`x`</b>: `Tensor` numerator of real numeric type.
*  <b>`y`</b>: `Tensor` denominator of real numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `x / y` returns the quotient of x and y.


- - -

#### `tf.Variable.__floordiv__(a, *args)` {#Variable.__floordiv__}

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

#### `tf.Variable.__ge__(a, *args)` {#Variable.__ge__}

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

#### `tf.Variable.__getitem__(var, slice_spec)` {#Variable.__getitem__}

Creates a slice helper object given a variable.

This allows creating a sub-tensor from part of the current contents
of a variable.
See
[`Tensor.__getitem__`](../../api_docs/python/framework.md#Tensor.__getitem__)
for detailed examples of slicing.

This function in addition also allows assignment to a sliced range.
This is similar to `__setitem__` functionality in Python. However,
the syntax is different so that the user can capture the assignment
operation for grouping or passing to `sess.run()`.
For example,

```prettyprint
import tensorflow as tf
A = tf.Variable([[1,2,3], [4,5,6], [7,8,9]], dtype=tf.float32)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print sess.run(A[:2, :2]) # => [[1,2], [4,5]]

  op = A[:2,:2].assign(22. * tf.ones((2, 2)))
  print sess.run(op) # => [[22, 22, 3], [22, 22, 6], [7,8,9]]
```

Note that assignments currently do not support NumPy broadcasting
semantics.

##### Args:


*  <b>`var`</b>: An `ops.Variable` object.
*  <b>`slice_spec`</b>: The arguments to `Tensor.__getitem__`.

##### Returns:

  The appropriate slice of "tensor", based on "slice_spec".
  As an operator. The operator also has a `assign()` method
  that can be used to generate an assignment operator.

##### Raises:


*  <b>`ValueError`</b>: If a slice range is negative size.
*  <b>`TypeError`</b>: If the slice indices aren't int, slice, or Ellipsis.


- - -

#### `tf.Variable.__gt__(a, *args)` {#Variable.__gt__}

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

#### `tf.Variable.__invert__(a, *args)` {#Variable.__invert__}

Returns the truth value of NOT x element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

#### `tf.Variable.__iter__()` {#Variable.__iter__}

Dummy method to prevent iteration. Do not call.

NOTE(mrry): If we register __getitem__ as an overloaded operator,
Python will valiantly attempt to iterate over the variable's Tensor from 0
to infinity.  Declaring this method prevents this unintended behavior.

##### Raises:


*  <b>`TypeError`</b>: when invoked.


- - -

#### `tf.Variable.__le__(a, *args)` {#Variable.__le__}

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

#### `tf.Variable.__lt__(a, *args)` {#Variable.__lt__}

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

#### `tf.Variable.__mod__(a, *args)` {#Variable.__mod__}

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

#### `tf.Variable.__mul__(a, *args)` {#Variable.__mul__}

Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse".


- - -

#### `tf.Variable.__neg__(a, *args)` {#Variable.__neg__}

Computes numerical negative value element-wise.

I.e., \\(y = -x\\).

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.


- - -

#### `tf.Variable.__or__(a, *args)` {#Variable.__or__}

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

#### `tf.Variable.__pow__(a, *args)` {#Variable.__pow__}

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

#### `tf.Variable.__radd__(a, *args)` {#Variable.__radd__}

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

#### `tf.Variable.__rand__(a, *args)` {#Variable.__rand__}

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

#### `tf.Variable.__rdiv__(a, *args)` {#Variable.__rdiv__}

Divide two values using Python 2 semantics. Used for Tensor.__div__.

##### Args:


*  <b>`x`</b>: `Tensor` numerator of real numeric type.
*  <b>`y`</b>: `Tensor` denominator of real numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `x / y` returns the quotient of x and y.


- - -

#### `tf.Variable.__rfloordiv__(a, *args)` {#Variable.__rfloordiv__}

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

#### `tf.Variable.__rmod__(a, *args)` {#Variable.__rmod__}

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

#### `tf.Variable.__rmul__(a, *args)` {#Variable.__rmul__}

Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse".


- - -

#### `tf.Variable.__ror__(a, *args)` {#Variable.__ror__}

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

#### `tf.Variable.__rpow__(a, *args)` {#Variable.__rpow__}

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

#### `tf.Variable.__rsub__(a, *args)` {#Variable.__rsub__}

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

#### `tf.Variable.__rtruediv__(a, *args)` {#Variable.__rtruediv__}




- - -

#### `tf.Variable.__rxor__(a, *args)` {#Variable.__rxor__}

x ^ y = (x | y) & ~(x & y).


- - -

#### `tf.Variable.__str__()` {#Variable.__str__}




- - -

#### `tf.Variable.__sub__(a, *args)` {#Variable.__sub__}

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

#### `tf.Variable.__truediv__(a, *args)` {#Variable.__truediv__}




- - -

#### `tf.Variable.__xor__(a, *args)` {#Variable.__xor__}

x ^ y = (x | y) & ~(x & y).


- - -

#### `tf.Variable.from_proto(variable_def, import_scope=None)` {#Variable.from_proto}

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

#### `tf.Variable.read_value()` {#Variable.read_value}

Returns the value of this variable, read in the current context.

Can be different from value() if it's on another device, with control
dependencies, etc.

##### Returns:

  A `Tensor` containing the value of the variable.


- - -

#### `tf.Variable.set_shape(shape)` {#Variable.set_shape}

Overrides the shape for this variable.

##### Args:


*  <b>`shape`</b>: the `TensorShape` representing the overridden shape.


- - -

#### `tf.Variable.to_proto(export_scope=None)` {#Variable.to_proto}

Converts a `Variable` to a `VariableDef` protocol buffer.

##### Args:


*  <b>`export_scope`</b>: Optional `string`. Name scope to remove.

##### Returns:

  A `VariableDef` protocol buffer, or `None` if the `Variable` is not
  in the specified name scope.


- - -

#### `tf.Variable.value()` {#Variable.value}

Returns the last snapshot of this variable.

You usually do not need to call this method as all ops that need the value
of the variable call it automatically through a `convert_to_tensor()` call.

Returns a `Tensor` which holds the value of the variable.  You can not
assign a new value to this tensor as it is not a reference to the variable.

To avoid copies, if the consumer of the returned value is on the same device
as the variable, this actually returns the live value of the variable, not
a copy.  Updates to the variable are seen by the consumer.  If the consumer
is on a different device it will get a copy of the variable.

##### Returns:

  A `Tensor` containing the value of the variable.


