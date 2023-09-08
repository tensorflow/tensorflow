# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Operator overloads for `RaggedTensor`."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator


# =============================================================================
# Equality Docstring
# =============================================================================
def ragged_eq(self, other):  # pylint: disable=g-doc-args
  """Returns result of elementwise `==` or False if not broadcast-compatible.

  Compares two ragged tensors elemewise for equality if they are
  broadcast-compatible; or returns False if they are not
  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

  Note that this behavior differs from `tf.math.equal`, which raises an
  exception if the two ragged tensors are not broadcast-compatible.

  For example:

  >>> rt1 = tf.ragged.constant([[1, 2], [3]])
  >>> rt1 == rt1
  <tf.RaggedTensor [[True, True], [True]]>

  >>> rt2 = tf.ragged.constant([[1, 2], [4]])
  >>> rt1 == rt2
  <tf.RaggedTensor [[True, True], [False]]>

  >>> rt3 = tf.ragged.constant([[1, 2], [3, 4]])
  >>> # rt1 and rt3 are not broadcast-compatible.
  >>> rt1 == rt3
  False

  >>> # You can also compare a `tf.RaggedTensor` to a `tf.Tensor`.
  >>> t = tf.constant([[1, 2], [3, 4]])
  >>> rt1 == t
  False
  >>> t == rt1
  False
  >>> rt4 = tf.ragged.constant([[1, 2], [3, 4]])
  >>> rt4 == t
  <tf.RaggedTensor [[True, True], [True, True]]>
  >>> t == rt4
  <tf.RaggedTensor [[True, True], [True, True]]>

  Args:
    other: The right-hand side of the `==` operator.

  Returns:
    The ragged tensor result of the elementwise `==` operation, or `False` if
    the arguments are not broadcast-compatible.
  """
  return math_ops.tensor_equals(self, other)


# =============================================================================
# Ordering Docstring
# =============================================================================
def ragged_ge(self, other):  # pylint: disable=g-doc-args
  """Elementwise `>=` comparison of two convertible-to-ragged-tensor values.

  Computes the elemewise `>=` comparison of two values that are convertible to
  ragged tenors, with [broadcasting]
  (http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) support.
  Raises an exception if two values are not broadcast-compatible.

  For example:

  >>> rt1 = tf.ragged.constant([[1, 2], [3]])
  >>> rt1 >= rt1
  <tf.RaggedTensor [[True, True], [True]]>

  >>> rt2 = tf.ragged.constant([[2, 1], [3]])
  >>> rt1 >= rt2
  <tf.RaggedTensor [[False, True], [True]]>

  >>> rt3 = tf.ragged.constant([[1, 2], [3, 4]])
  >>> # rt1 and rt3 are not broadcast-compatible.
  >>> rt1 >= rt3
  Traceback (most recent call last):
  ...
  InvalidArgumentError: ...

  >>> # You can also compare a `tf.RaggedTensor` to a `tf.Tensor`.
  >>> rt4 = tf.ragged.constant([[1, 2],[3, 4]])
  >>> t1 = tf.constant([[2, 1], [4, 3]])
  >>> rt4 >= t1
  <tf.RaggedTensor [[False, True],
   [False, True]]>
  >>> t1 >= rt4
  <tf.RaggedTensor [[True, False],
   [True, False]]>

  >>> # Compares a `tf.RaggedTensor` to a `tf.Tensor` with broadcasting.
  >>> t2 = tf.constant([[2]])
  >>> rt4 >= t2
  <tf.RaggedTensor [[False, True],
   [True, True]]>
  >>> t2 >= rt4
  <tf.RaggedTensor [[True, True],
   [False, False]]>

  Args:
    other: The right-hand side of the `>=` operator.

  Returns:
    A `tf.RaggedTensor` of dtype `tf.bool` with the shape that `self` and
    `other` broadcast to.

  Raises:
    InvalidArgumentError: If `self` and `other` are not broadcast-compatible.
  """
  return math_ops.greater_equal(self, other)


# =============================================================================
# Logical Docstring
# =============================================================================


# =============================================================================
# Arithmetic Docstring
# =============================================================================
def ragged_abs(self, name=None):  # pylint: disable=g-doc-args
  r"""Computes the absolute value of a ragged tensor.

  Given a ragged tensor of integer or floating-point values, this operation
  returns a ragged tensor of the same type, where each element contains the
  absolute value of the corresponding element in the input.

  Given a ragged tensor `x` of complex numbers, this operation returns a tensor
  of type `float32` or `float64` that is the absolute value of each element in
  `x`. For a complex number \\(a + bj\\), its absolute value is computed as
  \\(\sqrt{a^2 + b^2}\\).

  For example:

  >>> # real number
  >>> x = tf.ragged.constant([[-2.2, 3.2], [-4.2]])
  >>> tf.abs(x)
  <tf.RaggedTensor [[2.2, 3.2], [4.2]]>

  >>> # complex number
  >>> x = tf.ragged.constant([[-2.2 + 4.7j], [-3.2 + 5.7j], [-4.2 + 6.7j]])
  >>> tf.abs(x)
  <tf.RaggedTensor [[5.189412298131649],
   [6.536818798161687],
   [7.907591289387685]]>

  Args:
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor` of the same size and type as `x`, with absolute values.
    Note, for `complex64` or `complex128` input, the returned `RaggedTensor`
    will be of type `float32` or `float64`, respectively.
  """
  return math_ops.abs(self, name=name)


# ===========================================================================
def ragged_and(self, y, name=None):  # pylint: disable=g-doc-args
  r"""Returns the truth value of elementwise `x & y`.

  Logical AND function.

  Requires that `x` and `y` have the same shape or have
  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  shapes. For example, `y` can be:

    - A single Python boolean, where the result will be calculated by applying
      logical AND with the single element to each element in `x`.
    - A `tf.Tensor` object of dtype `tf.bool` of the same shape or
      [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
      shape. In this case, the result will be the element-wise logical AND of
      `x` and `y`.
    - A `tf.RaggedTensor` object of dtype `tf.bool` of the same shape or
      [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
      shape. In this case, the result will be the element-wise logical AND of
      `x` and `y`.

  For example:

  >>> # `y` is a Python boolean
  >>> x = tf.ragged.constant([[True, False], [True]])
  >>> y = True
  >>> x & y
  <tf.RaggedTensor [[True, False], [True]]>
  >>> tf.math.logical_and(x, y)  # Equivalent of x & y
  <tf.RaggedTensor [[True, False], [True]]>
  >>> y & x
  <tf.RaggedTensor [[True, False], [True]]>
  >>> tf.math.reduce_all(x & y)  # Reduce to a scalar bool Tensor.
  <tf.Tensor: shape=(), dtype=bool, numpy=False>

  >>> # `y` is a tf.Tensor of the same shape.
  >>> x = tf.ragged.constant([[True, False], [True, False]])
  >>> y = tf.constant([[True, False], [False, True]])
  >>> x & y
  <tf.RaggedTensor [[True, False], [False, False]]>

  >>> # `y` is a tf.Tensor of a broadcast-compatible shape.
  >>> x = tf.ragged.constant([[True, False], [True]])
  >>> y = tf.constant([[True], [False]])
  >>> x & y
  <tf.RaggedTensor [[True, False], [False]]>

  >>> # `y` is a `tf.RaggedTensor` of the same shape.
  >>> x = tf.ragged.constant([[True, False], [True]])
  >>> y = tf.ragged.constant([[False, True], [True]])
  >>> x & y
  <tf.RaggedTensor [[False, False], [True]]>

  >>> # `y` is a `tf.RaggedTensor` of a broadcast-compatible shape.
  >>> x = tf.ragged.constant([[[True, True, False]], [[]], [[True, False]]])
  >>> y = tf.ragged.constant([[[True]], [[True]], [[False]]], ragged_rank=1)
  >>> x & y
  <tf.RaggedTensor [[[True, True, False]], [[]], [[False, False]]]>

  Args:
    y: A Python boolean or a `tf.Tensor` or `tf.RaggedTensor` of dtype
      `tf.bool`.
    name: A name for the operation (optional).

  Returns:
    A `tf.RaggedTensor` of dtype `tf.bool` with the shape that `x` and `y`
    broadcast to.
  """
  return math_ops.logical_and(self, y, name)


# Helper Methods.
def _right(operator):
  """Right-handed version of an operator: swap args x and y."""
  return tf_decorator.make_decorator(operator, lambda y, x: operator(x, y))


def ragged_hash(self):
  """The operation invoked by the `RaggedTensor.__hash__` operator."""
  g = getattr(self.row_splits, "graph", None)
  # pylint: disable=protected-access
  if (
      tensor.Tensor._USE_EQUALITY
      and ops.executing_eagerly_outside_functions()
      and (g is None or g.building_function)
  ):
    raise TypeError("RaggedTensor is unhashable.")
  else:
    return id(self)


# Indexing
ragged_tensor.RaggedTensor.__getitem__ = ragged_getitem.ragged_tensor_getitem

# Equality
ragged_tensor.RaggedTensor.__eq__ = ragged_eq
ragged_tensor.RaggedTensor.__ne__ = math_ops.tensor_not_equals
ragged_tensor.RaggedTensor.__hash__ = ragged_hash

# Ordering operators
ragged_tensor.RaggedTensor.__ge__ = ragged_ge
ragged_tensor.RaggedTensor.__gt__ = math_ops.greater
ragged_tensor.RaggedTensor.__le__ = math_ops.less_equal
ragged_tensor.RaggedTensor.__lt__ = math_ops.less

# Logical operators
ragged_tensor.RaggedTensor.__and__ = ragged_and
ragged_tensor.RaggedTensor.__rand__ = _right(ragged_and)

ragged_tensor.RaggedTensor.__invert__ = math_ops.logical_not
ragged_tensor.RaggedTensor.__ror__ = _right(math_ops.logical_or)
ragged_tensor.RaggedTensor.__or__ = math_ops.logical_or
ragged_tensor.RaggedTensor.__xor__ = math_ops.logical_xor
ragged_tensor.RaggedTensor.__rxor__ = _right(math_ops.logical_xor)

# Arithmetic operators
ragged_tensor.RaggedTensor.__abs__ = ragged_abs
ragged_tensor.RaggedTensor.__add__ = math_ops.add
ragged_tensor.RaggedTensor.__radd__ = _right(math_ops.add)
ragged_tensor.RaggedTensor.__div__ = math_ops.div
ragged_tensor.RaggedTensor.__rdiv__ = _right(math_ops.div)
ragged_tensor.RaggedTensor.__floordiv__ = math_ops.floordiv
ragged_tensor.RaggedTensor.__rfloordiv__ = _right(math_ops.floordiv)
ragged_tensor.RaggedTensor.__mod__ = math_ops.floormod
ragged_tensor.RaggedTensor.__rmod__ = _right(math_ops.floormod)
ragged_tensor.RaggedTensor.__mul__ = math_ops.multiply
ragged_tensor.RaggedTensor.__rmul__ = _right(math_ops.multiply)
ragged_tensor.RaggedTensor.__neg__ = math_ops.negative
ragged_tensor.RaggedTensor.__pow__ = math_ops.pow
ragged_tensor.RaggedTensor.__rpow__ = _right(math_ops.pow)
ragged_tensor.RaggedTensor.__sub__ = math_ops.subtract
ragged_tensor.RaggedTensor.__rsub__ = _right(math_ops.subtract)
ragged_tensor.RaggedTensor.__truediv__ = math_ops.truediv
ragged_tensor.RaggedTensor.__rtruediv__ = _right(math_ops.truediv)


def ragged_bool(self):  # pylint: disable=g-doc-args
  """Raises TypeError when a RaggedTensor is used as a Python bool.

  To prevent RaggedTensor from being used as a bool, this function always raise
  TypeError when being called.

  For example:

  >>> x = tf.ragged.constant([[1, 2], [3]])
  >>> result = True if x else False  # Evaluate x as a bool value.
  Traceback (most recent call last):
  ...
  TypeError: RaggedTensor may not be used as a boolean.

  >>> x = tf.ragged.constant([[1]])
  >>> r = (x == 1)  # tf.RaggedTensor [[True]]
  >>> if r:  # Evaluate r as a bool value.
  ...   pass
  Traceback (most recent call last):
  ...
  TypeError: RaggedTensor may not be used as a boolean.
  """
  raise TypeError("RaggedTensor may not be used as a boolean.")


ragged_tensor.RaggedTensor.__bool__ = ragged_bool  # Python3 bool conversion.
ragged_tensor.RaggedTensor.__nonzero__ = ragged_bool  # Python2 bool conversion.
