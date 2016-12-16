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


