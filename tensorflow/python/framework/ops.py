# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Classes and functions used to construct graphs."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import linecache
import re
import sys
import threading

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import tf_contextlib


# Temporary global switch determining if we should enable the work-in-progress
# calls to the C API. Currently disabled by default but can be manually enabled
# e.g. in tests. This will be removed once all functionality is supported and
# there's no performance penalty with it enabled.
#
# TODO(skyewm) before we can remove this:
# - functions
# - import_graph_def() incrementally adds inputs to ops (i.e. creates an
#   Operation and then calls _add_input()). The current code requires that all
#   inputs be specified when creating the Operation (since we call
#   TF_FinishOperation()).
# - ops_test.py (and others?) create unregistered op types
# - while loop
# - performance (e.g. delete/refactor redundant Python functionality, switch to
#   new session API)
_USE_C_API = False


def _override_helper(clazz_object, operator, func):
  """Overrides (string) operator on Tensors to call func.

  Args:
    clazz_object: the class to override for; either Tensor or SparseTensor.
    operator: the string name of the operator to override.
    func: the function that replaces the overridden operator.

  Raises:
    ValueError: If operator has already been overwritten,
      or if operator is not allowed to be overwritten.
  """
  existing = getattr(clazz_object, operator, None)
  if existing is not None:
    # Check to see if this is a default method-wrapper or slot wrapper which
    # will be true for the comparison operators.
    if not isinstance(existing, type(object.__lt__)):
      raise ValueError("operator %s cannot be overwritten again on class %s." %
                       (operator, clazz_object))
  if operator not in Tensor.OVERLOADABLE_OPERATORS:
    raise ValueError("Overriding %s is disallowed" % operator)
  setattr(clazz_object, operator, func)


def _as_graph_element(obj):
  """Convert `obj` to a graph element if possible, otherwise return `None`.

  Args:
    obj: Object to convert.

  Returns:
    The result of `obj._as_graph_element()` if that method is available;
        otherwise `None`.
  """
  conv_fn = getattr(obj, "_as_graph_element", None)
  if conv_fn and callable(conv_fn):
    return conv_fn()
  return None


_TENSOR_LIKE_TYPES = tuple()


def is_dense_tensor_like(t):
  """EXPERIMENTAL: Returns true if `t` implements the tensor interface.

  See `register_dense_tensor_like_type()` for the current definition of a
  "tensor-like type".

  Args:
    t: An object.

  Returns:
    True iff `t` is an instance of one of the registered "tensor-like" types.
  """
  return isinstance(t, _TENSOR_LIKE_TYPES)


def register_dense_tensor_like_type(tensor_type):
  """EXPERIMENTAL: Registers `tensor_type` as implementing the tensor interface.

  A "tensor-like type" can represent a single dense tensor, and implements
  the `name` and `dtype` properties.

  Args:
    tensor_type: A type implementing the tensor interface.

  Raises:
    TypeError: If `tensor_type` does not implement the tensor interface.
  """
  try:
    if not isinstance(tensor_type.name, property):
      raise TypeError("Type %s does not define a `name` property")
  except AttributeError:
    raise TypeError("Type %s does not define a `name` property")
  try:
    if not isinstance(tensor_type.dtype, property):
      raise TypeError("Type %s does not define a `dtype` property")
  except AttributeError:
    raise TypeError("Type %s does not define a `dtype` property")
  # We expect this list to be small, so choose quadratic complexity
  # for registration, so that we have a tuple that can be used for
  # more efficient `isinstance` checks later.
  global _TENSOR_LIKE_TYPES
  _TENSOR_LIKE_TYPES = tuple(list(_TENSOR_LIKE_TYPES) + [tensor_type])


# NOTE(ebrevdo): Do not subclass this.  If you do, I will break you on purpose.
class _TensorLike(object):
  """Internal cls for grouping Tensor, SparseTensor, ..., for is_instance."""
  pass


class Tensor(_TensorLike):
  """Represents one of the outputs of an `Operation`.

  A `Tensor` is a symbolic handle to one of the outputs of an
  `Operation`. It does not hold the values of that operation's output,
  but instead provides a means of computing those values in a
  TensorFlow @{tf.Session}.

  This class has two primary purposes:

  1. A `Tensor` can be passed as an input to another `Operation`.
     This builds a dataflow connection between operations, which
     enables TensorFlow to execute an entire `Graph` that represents a
     large, multi-step computation.

  2. After the graph has been launched in a session, the value of the
     `Tensor` can be computed by passing it to
     @{tf.Session.run}.
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
  """

  # List of Python operators that we allow to override.
  OVERLOADABLE_OPERATORS = {
      # Binary.
      "__add__",
      "__radd__",
      "__sub__",
      "__rsub__",
      "__mul__",
      "__rmul__",
      "__div__",
      "__rdiv__",
      "__truediv__",
      "__rtruediv__",
      "__floordiv__",
      "__rfloordiv__",
      "__mod__",
      "__rmod__",
      "__lt__",
      "__le__",
      "__gt__",
      "__ge__",
      "__and__",
      "__rand__",
      "__or__",
      "__ror__",
      "__xor__",
      "__rxor__",
      "__getitem__",
      "__pow__",
      "__rpow__",
      # Unary.
      "__invert__",
      "__neg__",
      "__abs__",
      "__matmul__",
      "__rmatmul__"
  }

  def __init__(self, op, value_index, dtype):
    """Creates a new `Tensor`.

    Args:
      op: An `Operation`. `Operation` that computes this tensor.
      value_index: An `int`. Index of the operation's endpoint that produces
        this tensor.
      dtype: A `DType`. Type of elements stored in this tensor.

    Raises:
      TypeError: If the op is not an `Operation`.
    """
    if not isinstance(op, Operation):
      raise TypeError("op needs to be an Operation: %s" % op)
    self._op = op
    self._value_index = value_index
    self._dtype = dtypes.as_dtype(dtype)
    self._shape = tensor_shape.unknown_shape()
    # List of operations that use this Tensor as input.  We maintain this list
    # to easily navigate a computation graph.
    self._consumers = []

    # Attributes used for C++ shape inference. Not inspected, only forwarded.
    # If set, will be a HandleData object from cpp_shape_inference.proto.
    self._handle_data = None

  @property
  def op(self):
    """The `Operation` that produces this tensor as an output."""
    return self._op

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self._dtype

  @property
  def graph(self):
    """The `Graph` that contains this tensor."""
    return self._op.graph

  @property
  def name(self):
    """The string name of this tensor."""
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    return "%s:%d" % (self._op.name, self._value_index)

  @property
  def device(self):
    """The name of the device on which this tensor will be produced, or None."""
    return self._op.device

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of this tensor.

    The shape is computed using shape inference functions that are
    registered in the Op for each `Operation`.  See
    @{tf.TensorShape}
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

    Returns:
      A `TensorShape` representing the shape of this tensor.

    """
    return self._shape

  def _shape_as_list(self):
    if self._shape.ndims is not None:
      return [dim.value for dim in self._shape.dims]
    else:
      return None

  def get_shape(self):
    """Alias of Tensor.shape."""
    return self.shape

  def set_shape(self, shape):
    """Updates the shape of this tensor.

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

    Args:
      shape: A `TensorShape` representing the shape of this tensor.

    Raises:
      ValueError: If `shape` is not compatible with the current shape of
        this tensor.
    """
    self._shape = self._shape.merge_with(shape)

  @property
  def value_index(self):
    """The index of this tensor in the outputs of its `Operation`."""
    return self._value_index

  def consumers(self):
    """Returns a list of `Operation`s that consume this tensor.

    Returns:
      A list of `Operation`s.
    """
    return self._consumers

  def _add_consumer(self, consumer):
    """Add a consumer to this tensor.

    Args:
      consumer: an Operation.

    Raises:
      TypeError: if the consumer is not an Operation.
    """
    if not isinstance(consumer, Operation):
      raise TypeError("Consumer must be an Operation: %s" % consumer)
    self._consumers.append(consumer)

  def _as_node_def_input(self):
    """Return a value to use for the NodeDef "input" attribute.

    The returned string can be used in a NodeDef "input" attribute
    to indicate that the NodeDef uses this Tensor as input.

    Raises:
      ValueError: if this Tensor's Operation does not have a name.

    Returns:
      a string.
    """
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    if self._value_index == 0:
      return self._op.name
    else:
      return "%s:%d" % (self._op.name, self._value_index)

  def _as_tf_output(self):
    assert self.op._c_op  # pylint: disable=protected-access
    tf_output = c_api.TF_Output()
    tf_output.oper = self.op._c_op  # pylint: disable=protected-access
    tf_output.index = self.value_index
    return tf_output

  def __str__(self):
    return "Tensor(\"%s\"%s%s%s)" % (
        self.name,
        (", shape=%s" % self.get_shape())
        if self.get_shape().ndims is not None else "",
        (", dtype=%s" % self._dtype.name) if self._dtype else "",
        (", device=%s" % self.device) if self.device else "")

  def __repr__(self):
    return "<tf.Tensor '%s' shape=%s dtype=%s>" % (
        self.name, self.get_shape(), self._dtype.name)

  def __hash__(self):
    # Necessary to support Python's collection membership operators
    return id(self)

  def __eq__(self, other):
    # Necessary to support Python's collection membership operators
    return id(self) == id(other)

  # NOTE(mrry): This enables the Tensor's overloaded "right" binary
  # operators to run when the left operand is an ndarray, because it
  # accords the Tensor class higher priority than an ndarray, or a
  # numpy matrix.
  # TODO(mrry): Convert this to using numpy's __numpy_ufunc__
  # mechanism, which allows more control over how Tensors interact
  # with ndarrays.
  __array_priority__ = 100

  @staticmethod
  def _override_operator(operator, func):
    _override_helper(Tensor, operator, func)

  def __iter__(self):
    """Dummy method to prevent iteration. Do not call.

    NOTE(mrry): If we register __getitem__ as an overloaded operator,
    Python will valiantly attempt to iterate over the Tensor from 0 to
    infinity.  Declaring this method prevents this unintended
    behavior.

    Raises:
      TypeError: when invoked.
    """
    raise TypeError("'Tensor' object is not iterable.")

  def __bool__(self):
    """Dummy method to prevent a tensor from being used as a Python `bool`.

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

    Raises:
      `TypeError`.
    """
    raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
                    "Use `if t is not None:` instead of `if t:` to test if a "
                    "tensor is defined, and use TensorFlow ops such as "
                    "tf.cond to execute subgraphs conditioned on the value of "
                    "a tensor.")

  def __nonzero__(self):
    """Dummy method to prevent a tensor from being used as a Python `bool`.

    This is the Python 2.x counterpart to `__bool__()` above.

    Raises:
      `TypeError`.
    """
    raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
                    "Use `if t is not None:` instead of `if t:` to test if a "
                    "tensor is defined, and use TensorFlow ops such as "
                    "tf.cond to execute subgraphs conditioned on the value of "
                    "a tensor.")

  def eval(self, feed_dict=None, session=None):
    """Evaluates this tensor in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `Tensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values.
        See @{tf.Session.run} for a
        description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this tensor. If
        none, the default session will be used.

    Returns:
      A numpy array corresponding to the value of this tensor.

    """
    return _eval_using_default_session(self, feed_dict, self.graph, session)


def _TensorTensorConversionFunction(t, dtype=None, name=None, as_ref=False):
  _ = name, as_ref
  if dtype and not dtype.is_compatible_with(t.dtype):
    raise ValueError(
        "Tensor conversion requested dtype %s for Tensor with dtype %s: %r"
        % (dtype.name, t.dtype.name, str(t)))
  return t


_tensor_conversion_func_registry = {
    0: [(Tensor, _TensorTensorConversionFunction)]}
register_dense_tensor_like_type(Tensor)


def convert_to_tensor(value,
                      dtype=None,
                      name=None,
                      preferred_dtype=None):
  """Converts the given `value` to a `Tensor`.

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

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    preferred_dtype: Optional element type for the returned tensor,
      used when dtype is None. In some cases, a caller may not have a
      dtype in mind when converting to a tensor, so preferred_dtype
      can be used as a soft preference.  If the conversion to
      `preferred_dtype` is not possible, this argument has no effect.

  Returns:
    An `Output` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value`.
    RuntimeError: If a registered conversion function returns an invalid value.

  """
  return internal_convert_to_tensor(
      value=value,
      dtype=dtype,
      name=name,
      preferred_dtype=preferred_dtype,
      as_ref=False)


def internal_convert_to_tensor(value,
                               dtype=None,
                               name=None,
                               as_ref=False,
                               preferred_dtype=None):
  """Converts the given `value` to an `Tensor`.

  This function converts Python objects of various types to `Tensor`
  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
  and Python scalars. For example:

  This function can be useful when composing a new operation in Python
  All standard Python op constructors apply this function to each of their
  Tensor-valued inputs, which allows those ops to accept numpy arrays, Python
  lists, and scalars in addition to `Tensor` objects.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    as_ref: True if we want the mutable view of Variables, if applicable.
    preferred_dtype: Optional element type for the returned tensor,
      used when dtype is None. In some cases, a caller may not have a
      dtype in mind when converting to a tensor, so preferred_dtype
      can be used as a soft preference.  If the conversion to
      `preferred_dtype` is not possible, this argument has no effect.

  Returns:
    A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value`.
    RuntimeError: If a registered conversion function returns an invalid value.

  """
  error_prefix = "" if name is None else "%s: " % name
  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  for _, funcs_at_priority in sorted(_tensor_conversion_func_registry.items()):
    for base_type, conversion_func in funcs_at_priority:
      if isinstance(value, base_type):
        # If dtype is None but preferred_dtype is not None, we try to
        # cast to preferred_dtype first.
        ret = None
        if dtype is None and preferred_dtype is not None:
          try:
            ret = conversion_func(
                value, dtype=preferred_dtype, name=name, as_ref=as_ref)
          except (TypeError, ValueError):
            # Could not coerce the conversion to use the preferred dtype.
            ret = None

          if ret is not None and ret is not NotImplemented:
            if (ret.dtype.base_dtype !=
                dtypes.as_dtype(preferred_dtype).base_dtype):
              raise TypeError("convert_to_tensor did not convert to "
                              "the preferred dtype: %s vs %s " %
                              (ret.dtype.base_dtype,
                               dtypes.as_dtype(preferred_dtype).base_dtype))

        if ret is None:
          ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)

        if ret is NotImplemented:
          continue

        if not isinstance(ret, Tensor):
          raise RuntimeError(
              "%sConversion function %r for type %s returned non-Tensor: %r"
              % (error_prefix, conversion_func, base_type, ret))
        if dtype and not dtype.is_compatible_with(ret.dtype):
          raise RuntimeError(
              "%sConversion function %r for type %s returned incompatible "
              "dtype: requested = %s, actual = %s"
              % (error_prefix, conversion_func, base_type,
                 dtype.name, ret.dtype.name))
        return ret
  raise TypeError("%sCannot convert %r with type %s to Tensor: "
                  "no conversion function registered."
                  % (error_prefix, value, type(value)))


def internal_convert_n_to_tensor(values,
                                 dtype=None,
                                 name=None,
                                 as_ref=False,
                                 preferred_dtype=None):
  """Converts `values` to a list of `Tensor` objects.

  Args:
    values: A list of objects that can be consumed by `tf.convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` objects.
    name: (Optional.) A name prefix to used when a new `Tensor` is
      created, in which case element `i` will be given the name `name
      + '_' + i`.
    as_ref: True if the caller wants the results as ref tensors.
    preferred_dtype: Optional element type for the returned tensors,
      used when dtype is None. In some cases, a caller may not have a
      dtype in mind when converting to a tensor, so preferred_dtype
      can be used as a soft preference.  If the conversion to
      `preferred_dtype` is not possible, this argument has no effect.

  Returns:
    A list of `Tensor` and/or `IndexedSlices` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  if not isinstance(values, collections.Sequence):
    raise TypeError("values must be a list.")
  ret = []
  for i, value in enumerate(values):
    n = None if name is None else "%s_%d" % (name, i)
    ret.append(
        internal_convert_to_tensor(
            value,
            dtype=dtype,
            name=n,
            as_ref=as_ref,
            preferred_dtype=preferred_dtype))
  return ret


def convert_n_to_tensor(values,
                        dtype=None,
                        name=None,
                        preferred_dtype=None):
  """Converts `values` to a list of `Tensor` objects.

  Args:
    values: A list of objects that can be consumed by `tf.convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` objects.
    name: (Optional.) A name prefix to used when a new `Tensor` is
      created, in which case element `i` will be given the name `name
      + '_' + i`.
    preferred_dtype: Optional element type for the returned tensors,
      used when dtype is None. In some cases, a caller may not have a
      dtype in mind when converting to a tensor, so preferred_dtype
      can be used as a soft preference.  If the conversion to
      `preferred_dtype` is not possible, this argument has no effect.

  Returns:
    A list of `Tensor` and/or `IndexedSlices` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  return internal_convert_n_to_tensor(values=values,
                                      dtype=dtype,
                                      name=name,
                                      preferred_dtype=preferred_dtype,
                                      as_ref=False)


def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None):
  """Converts the given object to a `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` or `SparseTensor` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.

  Returns:
    An `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  return internal_convert_to_tensor_or_indexed_slices(
      value=value, dtype=dtype, name=name, as_ref=False)


def internal_convert_to_tensor_or_indexed_slices(value, dtype=None, name=None,
                                                 as_ref=False):
  """Converts the given object to an `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` or `SparseTensor` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    An `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  if isinstance(value, _TensorLike):
    if dtype and not dtypes.as_dtype(dtype).is_compatible_with(value.dtype):
      raise ValueError(
          "Tensor conversion requested dtype %s for Tensor with dtype %s: %r"
          % (dtypes.as_dtype(dtype).name, value.dtype.name, str(value)))
    return value
  else:
    return internal_convert_to_tensor(value,
                                      dtype=dtype,
                                      name=name,
                                      as_ref=as_ref)


def internal_convert_n_to_tensor_or_indexed_slices(values, dtype=None,
                                                   name=None, as_ref=False):
  """Converts `values` to a list of `Tensor` or `IndexedSlices` objects.

  Any `IndexedSlices` or `SparseTensor` objects in `values` are returned
  unmodified.

  Args:
    values: A list of `None`, `IndexedSlices`, `SparseTensor`, or objects that
      can be consumed by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor`
      `IndexedSlices`.
    name: (Optional.) A name prefix to used when a new `Tensor` is
      created, in which case element `i` will be given the name `name
      + '_' + i`.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A list of `Tensor`, `IndexedSlices`, and/or `SparseTensor` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  if not isinstance(values, collections.Sequence):
    raise TypeError("values must be a list.")
  ret = []
  for i, value in enumerate(values):
    if value is None:
      ret.append(value)
    else:
      n = None if name is None else "%s_%d" % (name, i)
      ret.append(
          internal_convert_to_tensor_or_indexed_slices(
              value, dtype=dtype, name=n, as_ref=as_ref))
  return ret


def convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None):
  """Converts `values` to a list of `Output` or `IndexedSlices` objects.

  Any `IndexedSlices` or `SparseTensor` objects in `values` are returned
  unmodified.

  Args:
    values: A list of `None`, `IndexedSlices`, `SparseTensor`, or objects that
      can be consumed by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor`
      `IndexedSlices`.
    name: (Optional.) A name prefix to used when a new `Tensor` is
      created, in which case element `i` will be given the name `name
      + '_' + i`.

  Returns:
    A list of `Tensor`, `IndexedSlices`, and/or `SparseTensor` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  return internal_convert_n_to_tensor_or_indexed_slices(
      values=values, dtype=dtype, name=name, as_ref=False)


def register_tensor_conversion_function(base_type, conversion_func,
                                        priority=100):
  """Registers a function for converting objects of `base_type` to `Tensor`.

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

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values
      run earlier than conversion functions with larger priority values.
      Defaults to 100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.

  """
  if not (isinstance(base_type, type) or
          (isinstance(base_type, tuple)
           and all(isinstance(x, type) for x in base_type))):
    raise TypeError("base_type must be a type or a tuple of types.")
  if not callable(conversion_func):
    raise TypeError("conversion_func must be callable.")

  try:
    funcs_at_priority = _tensor_conversion_func_registry[priority]
  except KeyError:
    funcs_at_priority = []
    _tensor_conversion_func_registry[priority] = funcs_at_priority
  funcs_at_priority.append((base_type, conversion_func))


class IndexedSlices(_TensorLike):
  """A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. @{tf.gather}).

  Contrast this representation with
  @{tf.SparseTensor},
  which uses multi-dimensional indices and scalar values.
  """

  def __init__(self, values, indices, dense_shape=None):
    """Creates an `IndexedSlices`."""
    _get_graph_from_inputs([values, indices, dense_shape])
    self._values = values
    self._indices = indices
    self._dense_shape = dense_shape

  @property
  def values(self):
    """A `Tensor` containing the values of the slices."""
    return self._values

  @property
  def indices(self):
    """A 1-D `Tensor` containing the indices of the slices."""
    return self._indices

  @property
  def dense_shape(self):
    """A 1-D `Tensor` containing the shape of the corresponding dense tensor."""
    return self._dense_shape

  @property
  def name(self):
    """The name of this `IndexedSlices`."""
    return self.values.name

  @property
  def device(self):
    """The name of the device on which `values` will be produced, or `None`."""
    return self.values.device

  @property
  def op(self):
    """The `Operation` that produces `values` as an output."""
    return self.values.op

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self.values.dtype

  @property
  def graph(self):
    """The `Graph` that contains the values, indices, and shape tensors."""
    return self._values.graph

  def __str__(self):
    return "IndexedSlices(indices=%s, values=%s%s)" % (
        self._indices, self._values,
        (", dense_shape=%s" % self._dense_shape)
        if self._dense_shape is not None else "")

  def __neg__(self):
    return IndexedSlices(-self.values, self.indices, self.dense_shape)


IndexedSlicesValue = collections.namedtuple(
    "IndexedSlicesValue", ["values", "indices", "dense_shape"])


def _device_string(dev_spec):
  if isinstance(dev_spec, pydev.DeviceSpec):
    return dev_spec.to_string()
  else:
    return dev_spec


def _NodeDef(op_type, name, device=None, attrs=None):
  """Create a NodeDef proto.

  Args:
    op_type: Value for the "op" attribute of the NodeDef proto.
    name: Value for the "name" attribute of the NodeDef proto.
    device: string, device, or function from NodeDef to string.
      Value for the "device" attribute of the NodeDef proto.
    attrs: Optional dictionary where the key is the attribute name (a string)
      and the value is the respective "attr" attribute of the NodeDef proto (an
      AttrValue).

  Returns:
    A node_def_pb2.NodeDef protocol buffer.
  """
  node_def = node_def_pb2.NodeDef()
  node_def.op = compat.as_bytes(op_type)
  node_def.name = compat.as_bytes(name)
  if attrs is not None:
    for k, v in six.iteritems(attrs):
      node_def.attr[k].CopyFrom(v)
  if device is not None:
    if callable(device):
      node_def.device = device(node_def)
    else:
      node_def.device = _device_string(device)
  return node_def


# Copied from core/framework/node_def_util.cc
# TODO(mrry,josh11b): Consolidate this validation in C++ code.
_VALID_OP_NAME_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$")
_VALID_SCOPE_NAME_REGEX = re.compile("^[A-Za-z0-9_.\\-/]*$")


class Operation(object):
  """Represents a graph node that performs computation on tensors.

  An `Operation` is a node in a TensorFlow `Graph` that takes zero or
  more `Tensor` objects as input, and produces zero or more `Tensor`
  objects as output. Objects of type `Operation` are created by
  calling a Python op constructor (such as
  @{tf.matmul})
  or @{tf.Graph.create_op}.

  For example `c = tf.matmul(a, b)` creates an `Operation` of type
  "MatMul" that takes tensors `a` and `b` as input, and produces `c`
  as output.

  After the graph has been launched in a session, an `Operation` can
  be executed by passing it to
  @{tf.Session.run}.
  `op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.
  """

  def __init__(self, node_def, g, inputs=None, output_types=None,
               control_inputs=None, input_types=None, original_op=None,
               op_def=None):
    r"""Creates an `Operation`.

    NOTE: This constructor validates the name of the `Operation` (passed
    as `node_def.name`). Valid `Operation` names match the following
    regular expression:

        [A-Za-z0-9.][A-Za-z0-9_.\\-/]*

    Args:
      node_def: `node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`.
        Used for attributes of `node_def_pb2.NodeDef`, typically `name`,
        `op`, and `device`.  The `input` attribute is irrelevant here
        as it will be computed when generating the model.
      g: `Graph`. The parent graph.
      inputs: list of `Tensor` objects. The inputs to this `Operation`.
      output_types: list of `DType` objects.  List of the types of the
        `Tensors` computed by this operation.  The length of this list indicates
        the number of output endpoints of the `Operation`.
      control_inputs: list of operations or tensors from which to have a
        control dependency.
      input_types: List of `DType` objects representing the
        types of the tensors accepted by the `Operation`.  By default
        uses `[x.dtype.base_dtype for x in inputs]`.  Operations that expect
        reference-typed inputs must specify these explicitly.
      original_op: Optional. Used to associate the new `Operation` with an
        existing `Operation` (for example, a replica with the op that was
        replicated).
      op_def: Optional. The `op_def_pb2.OpDef` proto that describes the
        op type that this `Operation` represents.

    Raises:
      TypeError: if control inputs are not Operations or Tensors,
        or if `node_def` is not a `NodeDef`,
        or if `g` is not a `Graph`,
        or if `inputs` are not tensors,
        or if `inputs` and `input_types` are incompatible.
      ValueError: if the `node_def` name is not valid.
    """
    if not isinstance(node_def, node_def_pb2.NodeDef):
      raise TypeError("node_def needs to be a NodeDef: %s" % node_def)
    if node_def.ByteSize() >= (1 << 31) or node_def.ByteSize() < 0:
      raise ValueError(
          "Cannot create a tensor proto whose content is larger than 2GB.")
    if not _VALID_OP_NAME_REGEX.match(node_def.name):
      raise ValueError("'%s' is not a valid node name" % node_def.name)
    if not isinstance(g, Graph):
      raise TypeError("g needs to be a Graph: %s" % g)
    self._node_def = copy.deepcopy(node_def)
    self._graph = g
    if inputs is None:
      inputs = []
    elif not isinstance(inputs, list):
      raise TypeError("inputs needs to be a list of Tensors: %s" % inputs)
    self._inputs = list(inputs)  # Defensive copy.
    for a in self._inputs:
      if not isinstance(a, Tensor):
        raise TypeError("input needs to be a Tensor: %s" % a)
      # Mark that we consume the inputs.
      a._add_consumer(self)  # pylint: disable=protected-access
    if output_types is None:
      output_types = []
    self._output_types_val = output_types
    self._outputs = [Tensor(self, i, output_type)
                     for i, output_type in enumerate(output_types)]
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in self._inputs]
    else:
      if not all(x.is_compatible_with(i.dtype)
                 for i, x in zip(self._inputs, input_types)):
        raise TypeError("In op '%s', input types (%s) are not compatible "
                        "with expected types (%s)" % (
                            self.node_def.name,
                            [i.dtype for i in self._inputs],
                            input_types))
    self._input_types = input_types

    # Build the list of control inputs.
    self._control_inputs = []
    if control_inputs:
      for c in control_inputs:
        c_op = None
        if isinstance(c, Operation):
          c_op = c
        elif isinstance(c, (Tensor, IndexedSlices)):
          c_op = c.op
        else:
          raise TypeError("Control input must be an Operation, "
                          "a Tensor, or IndexedSlices: %s" % c)
        self._control_inputs.append(c_op)

    self._original_op = original_op
    self._op_def = op_def
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access
    # Add this op to the current control flow context:
    self._control_flow_context = g._get_control_flow_context()
    if self._control_flow_context is not None:
      self._control_flow_context.AddOp(self)
    # NOTE(keveman): Control flow context's AddOp could be creating new ops and
    # setting op.inputs[index] = new_op. Thus the new ops' id could be larger
    # than this op's id even though this op depend on them. Therefore, delaying
    # assigning id to this op until all ops this could be dependent on are
    # created.
    self._id_value = self._graph._next_id()  # pylint: disable=protected-access
    self._recompute_node_def()

    if _USE_C_API:
      assert self._graph._c_graph, (  # pylint: disable=protected-access
          "_USE_C_API set to False when creating Graph, you may need to "
          "manually set 'ops._USE_C_API = True' before creating the Graph")
      if self._op_def:
        # TODO(skyewm): op_def_library.apply_op() flattens the incoming
        # inputs. Refactor so we don't have to do this here.
        grouped_inputs = self._reconstruct_sequence_inputs(
            self._op_def, self._inputs, self._node_def.attr)
      else:
        # If no OpDef is specified, assume all inputs are scalar.
        grouped_inputs = self._inputs

      self._c_op = self._create_c_op(self._graph, self._node_def,
                                     grouped_inputs, self._control_inputs)
    else:
      self._c_op = None

  def _create_c_op(self, graph, node_def, inputs, control_inputs):
    """Creates a TF_Operation.

    Arguments:
      graph: a `Graph`.
      node_def: `node_def_pb2.NodeDef` for the operation to create.
      inputs: A list of `Tensor`s (corresponding to scalar inputs) and lists of
        `Tensor`s (corresponding to sequence inputs, e.g. "int64 * N",
        "list(int64)"). The length of the list should be equal to the number of
        inputs specified by this operation's op def.
      control_inputs: A list of `Operation`s to set as control dependencies.

    Returns:
      A wrapped TF_Operation*.
    """
    # pylint: disable=protected-access
    op_desc = c_api.TF_NewOperation(graph._c_graph, compat.as_str(node_def.op),
                                    compat.as_str(node_def.name))
    # Add inputs
    for op_input in inputs:
      if isinstance(op_input, (list, tuple)):
        c_api.TF_AddInputList(op_desc, [t._as_tf_output() for t in op_input])
      else:
        c_api.TF_AddInput(op_desc, op_input._as_tf_output())

    # Add control inputs
    for control_input in control_inputs:
      c_api.TF_AddControlInput(op_desc, control_input._c_op)
    # pylint: enable=protected-access

    # Add attrs
    for name, attr_value in node_def.attr.items():
      serialized = attr_value.SerializeToString()
      # TODO(skyewm): this creates and deletes a new TF_Status for every attr.
      # It might be worth creating a convenient way to re-use the same status.
      with errors.raise_exception_on_not_ok_status() as status:
        c_api.TF_SetAttrValueProto(op_desc, compat.as_str(name), serialized,
                                   status)

    with errors.raise_exception_on_not_ok_status() as status:
      c_op = c_api.TF_FinishOperation(op_desc, status)

    return c_op

  def _reconstruct_sequence_inputs(self, op_def, inputs, attrs):
    """Regroups a flat list of input tensors into scalar and sequence inputs.

    Arguments:
      op_def: The `op_def_pb2.OpDef` (for knowing the input types)
      inputs: a list of input `Tensor`s to the op.
      attrs: mapping from attr name to `attr_value_pb2.AttrValue` (these define
        how long each sequence is)

    Returns:
      A list of `Tensor`s (corresponding to scalar inputs) and lists of
      `Tensor`s (corresponding to sequence inputs).
    """
    grouped_inputs = []
    i = 0
    for input_arg in op_def.input_arg:
      if input_arg.number_attr:
        input_len = attrs[input_arg.number_attr].i
        is_sequence = True
      elif input_arg.type_list_attr:
        input_len = len(attrs[input_arg.type_list_attr].list.type)
        is_sequence = True
      else:
        input_len = 1
        is_sequence = False

      if is_sequence:
        grouped_inputs.append(inputs[i:i + input_len])
      else:
        grouped_inputs.append(inputs[i])
      i += input_len

    assert i == len(inputs)
    return grouped_inputs

  def colocation_groups(self):
    """Returns the list of colocation groups of the op."""
    default_colocation_group = [compat.as_bytes("loc:@%s" %
                                                self._node_def.name)]
    if "_class" not in self._node_def.attr:
      # This op has no explicit colocation group, so it is itself its
      # own root of a colocation group.
      return default_colocation_group

    attr_groups = [class_name
                   for class_name in self.get_attr("_class")
                   if class_name.startswith(b"loc:@")]

    # If there are no colocation groups in the explicit _class field,
    # return the default colocation group.
    return attr_groups if attr_groups else default_colocation_group

  def values(self):
    """DEPRECATED: Use outputs."""
    return tuple(self.outputs)

  def _get_control_flow_context(self):
    """Returns the control flow context of this op.

    Returns:
      A context object.
    """
    return self._control_flow_context

  def _set_control_flow_context(self, context):
    """Sets the current control flow context of this op.

    Args:
      context: a context object.
    """
    self._control_flow_context = context

  @property
  def name(self):
    """The full name of this operation."""
    if _USE_C_API:
      # TODO(iga): Remove this assert after converting to C API by default.
      # Just being a bit paranoid here.
      assert self._node_def.name == c_api.TF_OperationName(self._c_op)
      return c_api.TF_OperationName(self._c_op)
    else:
      return self._node_def.name

  @property
  def _id(self):
    """The unique integer id of this operation."""
    return self._id_value

  @property
  def device(self):
    """The name of the device to which this op has been assigned, if any.

    Returns:
      The string name of the device to which this op has been
      assigned, or an empty string if it has not been assigned to a
      device.
    """
    if _USE_C_API:
      # TODO(iga): Remove this assert after converting to C API by default.
      # Just being a bit paranoid here.
      assert self._node_def.device == c_api.TF_OperationDevice(self._c_op)
      return c_api.TF_OperationDevice(self._c_op)
    else:
      return self._node_def.device

  @property
  def _output_types(self):
    """List this operation's output types.

    Returns:
      List of the types of the Tensors computed by this operation.
      Each element in the list is an integer whose value is one of
      the TF_DataType enums defined in c_api.h
      The length of this list indicates the number of output endpoints
      of the operation.
    """
    if _USE_C_API:
      num_outputs = c_api.TF_OperationNumOutputs(self._c_op)
      output_types = [c_api.TF_OperationOutputType(self._tf_output(i)) for
                      i in xrange(num_outputs)]
      # TODO(iga): Remove this assert after converting to C API by default.
      # Just being a bit paranoid here.
      assert self._output_types_val == output_types
      # In all the tests we have output_types that are passed into
      # Operation.__init__ are a list of ints (which is illegal according
      # to the docstring), but input_types are instances of DType.
      # This extra assert is to catch if we ever use DType for output_types.
      if output_types:
        assert isinstance(output_types[0], int)
      return output_types
    else:
      return self._output_types_val

  def _tf_output(self, output_idx):
    """Create and return a new TF_Output for output_idx'th output of this op."""
    tf_output = c_api.TF_Output()
    tf_output.oper = self._c_op
    tf_output.index = output_idx
    return tf_output

  def _set_device(self, device):
    """Set the device of this operation.

    Args:
      device: string or device..  The device to set.
    """
    assert not _USE_C_API, "Operation._set_device doesn't work with C API"
    self._node_def.device = _device_string(device)

  def _add_input(self, tensor, dtype=None):
    """Add a new input to this operation.

    Args:
      tensor: the Tensor to add as an input.
      dtype: tf.DType: type of the input; defaults to
        the tensor's dtype.

    Raises:
      TypeError: if tensor is not a Tensor,
        or if input tensor type is not convertible to dtype.
      ValueError: if the Tensor is from a different graph.
    """
    if not isinstance(tensor, Tensor):
      raise TypeError("tensor must be a Tensor: %s" % tensor)
    _assert_same_graph(self, tensor)
    if dtype is None:
      dtype = tensor.dtype
    else:
      dtype = dtypes.as_dtype(dtype)
      if not dtype.is_compatible_with(tensor.dtype):
        raise TypeError(
            "Cannot convert a tensor of type %s to an input of type %s"
            % (tensor.dtype.name, dtype.name))
    self._inputs.append(tensor)
    self._input_types.append(dtype)
    tensor._add_consumer(self)  # pylint: disable=protected-access
    self._recompute_node_def()

  def _update_input(self, index, tensor, dtype=None):
    """Update the input to this operation at the given index.

    NOTE: This is for TF internal use only. Please don't use it.

    Args:
      index: the index of the input to update.
      tensor: the Tensor to be used as the input at the given index.
      dtype: tf.DType: type of the input; defaults to
        the tensor's dtype.

    Raises:
      TypeError: if tensor is not a Tensor,
        or if input tensor type is not convertible to dtype.
      ValueError: if the Tensor is from a different graph.
    """
    if not isinstance(tensor, Tensor):
      raise TypeError("tensor must be a Tensor: %s" % tensor)
    _assert_same_graph(self, tensor)
    if dtype is None:
      dtype = tensor.dtype
    else:
      dtype = dtypes.as_dtype(dtype)
      if not dtype.is_compatible_with(tensor.dtype):
        raise TypeError(
            "Cannot convert a tensor of type %s to an input of type %s"
            % (tensor.dtype.name, dtype.name))

    self._inputs[index].consumers().remove(self)
    self._inputs[index] = tensor
    self._input_types[index] = dtype
    tensor._add_consumer(self)  # pylint: disable=protected-access
    self._recompute_node_def()

  def _add_control_inputs(self, ops):
    """Add a list of new control inputs to this operation.

    Args:
      ops: the list of Operations to add as control input.

    Raises:
      TypeError: if ops is not a list of Operations.
      ValueError: if any op in ops is from a different graph.
    """
    if ops:
      for op in ops:
        if not isinstance(op, Operation):
          raise TypeError("op must be an Operation: %s" % op)
        _assert_same_graph(self, op)
        self._control_inputs.append(op)
      self._recompute_node_def()

  def _add_control_input(self, op):
    """Add a new control input to this operation.

    Args:
      op: the Operation to add as control input.

    Raises:
      TypeError: if op is not an Operation.
      ValueError: if op is from a different graph.
    """
    self._add_control_inputs([op])

  # Methods below are used when building the NodeDef and Graph proto.
  def _recompute_node_def(self):
    del self._node_def.input[:]
    self._node_def.input.extend([t._as_node_def_input() for t in self._inputs])
    if self._control_inputs:
      self._node_def.input.extend(["^%s" % op.name for op in
                                   self._control_inputs])

  def __str__(self):
    return str(self._node_def)

  def __repr__(self):
    return "<tf.Operation '%s' type=%s>" % (self.name, self.type)

  @property
  def outputs(self):
    """The list of `Tensor` objects representing the outputs of this op."""
    return self._outputs

# pylint: disable=protected-access
  class _InputList(object):
    """Immutable input list wrapper."""

    def __init__(self, op):
      self._op = op

    def __iter__(self):
      return iter(self._op._inputs)

    def __len__(self):
      return len(self._op._inputs)

    def __bool__(self):
      return bool(self._op._inputs)

    # Python 3 wants __bool__, Python 2.7 wants __nonzero__
    __nonzero__ = __bool__

    def __getitem__(self, i):
      return self._op._inputs[i]
# pylint: enable=protected-access

  @property
  def inputs(self):
    """The list of `Tensor` objects representing the data inputs of this op."""
    return Operation._InputList(self)

  @property
  def _input_dtypes(self):
    return self._input_types

  @property
  def control_inputs(self):
    """The `Operation` objects on which this op has a control dependency.

    Before this op is executed, TensorFlow will ensure that the
    operations in `self.control_inputs` have finished executing. This
    mechanism can be used to run ops sequentially for performance
    reasons, or to ensure that the side effects of an op are observed
    in the correct order.

    Returns:
      A list of `Operation` objects.

    """
    if _USE_C_API:
      control_c_ops = c_api.TF_OperationGetControlInputs_wrapper(self._c_op)
      # pylint: disable=protected-access
      return [self.graph._get_operation_by_name_unsafe(
          c_api.TF_OperationName(c_op)) for c_op in control_c_ops]
      # pylint: enable=protected-access
    else:
      return self._control_inputs

  @property
  def type(self):
    """The type of the op (e.g. `"MatMul"`)."""
    return self._node_def.op

  @property
  def graph(self):
    """The `Graph` that contains this operation."""
    return self._graph

  @property
  def node_def(self):
    # pylint: disable=line-too-long
    """Returns a serialized `NodeDef` representation of this operation.

    Returns:
      A
      [`NodeDef`](https://www.tensorflow.org/code/tensorflow/core/framework/node_def.proto)
      protocol buffer.
    """
    # pylint: enable=line-too-long
    return self._node_def

  @property
  def op_def(self):
    # pylint: disable=line-too-long
    """Returns the `OpDef` proto that represents the type of this op.

    Returns:
      An
      [`OpDef`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto)
      protocol buffer.
    """
    # pylint: enable=line-too-long
    return self._op_def

  @property
  def traceback(self):
    """Returns the call stack from when this operation was constructed."""
    return self._graph._convert_stack(self._traceback)  # pylint: disable=protected-access

  @property
  def traceback_with_start_lines(self):
    """Same as traceback but includes start line of function definition.

    Returns:
      A list of 5-tuples (filename, lineno, name, code, func_start_lineno).
    """
    return self._graph._convert_stack(  # pylint: disable=protected-access
        self._traceback, include_func_start_lineno=True)

  def get_attr(self, name):
    """Returns the value of the attr of this op with the given `name`.

    Args:
      name: The name of the attr to fetch.

    Returns:
      The value of the attr, as a Python object.

    Raises:
      ValueError: If this op does not have an attr with the given `name`.
    """
    fields = ["s", "i", "f", "b", "type", "shape", "tensor"]
    if name not in self._node_def.attr:
      raise ValueError("No attr named '" + name + "' in " +
                       str(self._node_def))
    x = self._node_def.attr[name]
    # Treat an empty oneof value as an empty list.
    if not x.WhichOneof("value"):
      return []
    if x.HasField("list"):
      for f in fields:
        if getattr(x.list, f):
          if f == "type":
            return [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
          else:
            return list(getattr(x.list, f))
      return []
    else:
      for f in fields:
        if x.HasField(f):
          if f == "type":
            return dtypes.as_dtype(getattr(x, f))
          else:
            return getattr(x, f)
      assert False, "Unsupported field type in " + str(x)

  def run(self, feed_dict=None, session=None):
    """Runs this operation in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for this operation.

    *N.B.* Before invoking `Operation.run()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values.
        See @{tf.Session.run}
        for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to run to this operation. If
        none, the default session will be used.
    """
    _run_using_default_session(self, feed_dict, self.graph, session)


_gradient_registry = registry.Registry("gradient")


class RegisterGradient(object):
  """A decorator for registering the gradient function for an op type.

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
  """

  def __init__(self, op_type):
    """Creates a new decorator with `op_type` as the Operation type.

    Args:
      op_type: The string type of an operation. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
    """
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f):
    """Registers the function `f` as gradient function for `op_type`."""
    _gradient_registry.register(f, self._op_type)
    return f


def NotDifferentiable(op_type):
  """Specifies that ops of type `op_type` is not differentiable.

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

  Args:
    op_type: The string type of an operation. This corresponds to the
      `OpDef.name` field for the proto that defines the operation.

  Raises:
    TypeError: If `op_type` is not a string.

  """
  if not isinstance(op_type, six.string_types):
    raise TypeError("op_type must be a string")
  _gradient_registry.register(None, op_type)


# Alias for the old name, will be eventually removed.
NoGradient = NotDifferentiable


def get_gradient_function(op):
  """Returns the function that computes gradients for "op"."""
  if not op.inputs: return None
  try:
    op_type = op.get_attr("_gradient_op_type")
  except ValueError:
    op_type = op.type
  return _gradient_registry.lookup(op_type)


_shape_registry = registry.Registry("shape functions")
_default_shape_function_registry = registry.Registry("default shape functions")

# These are set to common_shapes.call_cpp_shape_fn by op generated code
# (generated by python_op_gen.cc).
# It is set outside ops.py to avoid a circular dependency.
_call_cpp_shape_fn = None
_call_cpp_shape_fn_and_require_op = None


def _set_call_cpp_shape_fn(call_cpp_shape_fn):
  """Sets default shape fns from passed common_shapes.call_cpp_shape_fn."""
  global _call_cpp_shape_fn, _call_cpp_shape_fn_and_require_op
  if _call_cpp_shape_fn:
    return  # already registered

  def call_without_requiring(op):
    return call_cpp_shape_fn(op, require_shape_fn=False)

  _call_cpp_shape_fn = call_without_requiring

  def call_with_requiring(op):
    return call_cpp_shape_fn(op, require_shape_fn=True)

  _call_cpp_shape_fn_and_require_op = call_with_requiring


class RegisterShape(object):
  """No longer used.  Was: A decorator for registering a shape function.

  Shape functions must now be registered via the SetShapeFn on the
  original Op specification in C++.

  """

  def __init__(self, op_type):
    """Saves the `op_type` as the `Operation` type."""
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f):
    """Registers "f" as the shape function for "op_type"."""
    if f is None:
      assert _call_cpp_shape_fn

      # None is a special "weak" value that provides a default shape function,
      # and can be overridden by a non-None registration.
      try:
        _default_shape_function_registry.register(_call_cpp_shape_fn,
                                                  self._op_type)
      except KeyError:
        # Ignore duplicate registrations of the weak value. This can
        # occur if the op library input to wrapper generation
        # inadvertently links in one or more of the standard op
        # libraries.
        pass
    else:
      _shape_registry.register(f, self._op_type)
    return f


def set_shapes_for_outputs(op):
  """Uses the registered shape functions to set the shapes for op's outputs."""
  try:
    shape_func = _shape_registry.lookup(op.type)
  except LookupError:
    try:
      shape_func = _default_shape_function_registry.lookup(op.type)
    except LookupError:
      shape_func = _call_cpp_shape_fn_and_require_op

  shapes = shape_func(op)
  if shapes is None:
    raise RuntimeError(
        "Shape function for op %s did not return any shapes" % op)
  elif isinstance(shapes, dict):
    # Returned by call_cpp_shape_fn
    shapes_dict = shapes
    shapes = shapes_dict["shapes"]
    handle_datas = shapes_dict["handle_data"]
    for output, handle_data in zip(op.outputs, handle_datas):
      # pylint: disable=protected-access
      output._handle_data = handle_data
      # pylint: enable=protected-access

  if len(op.outputs) != len(shapes):
    raise RuntimeError(
        "Shape function for op %s returned %d shapes but expected %d %s %s" %
        (op, len(shapes), len(op.outputs), shape_func.__name__, str(shapes)))
  for output, s in zip(op.outputs, shapes):
    output.set_shape(s)


class OpStats(object):
  """A holder for statistics about an operator.

  This class holds information about the resource requirements for an op,
  including the size of its weight parameters on-disk and how many FLOPS it
  requires to execute forward inference.

  If you define a new operation, you can create a function that will return a
  set of information about its usage of the CPU and disk space when serialized.
  The function itself takes a Graph object that's been set up so you can call
  methods like get_tensor_by_name to help calculate the results, and a NodeDef
  argument.

  """

  def __init__(self, statistic_type, value=None):
    """Sets up the initial placeholders for the statistics."""
    self.statistic_type = statistic_type
    self.value = value

  @property
  def statistic_type(self):
    return self._statistic_type

  @statistic_type.setter
  def statistic_type(self, statistic_type):
    self._statistic_type = statistic_type

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value

  def __iadd__(self, other):
    if other.statistic_type != self.statistic_type:
      raise ValueError("Can't add an OpStat of type %s to one of %s.",
                       self.statistic_type, other.statistic_type)
    if self.value is None:
      self.value = other.value
    elif other.value is not None:
      self._value += other.value
    return self

_stats_registry = registry.Registry("statistical functions")


class RegisterStatistics(object):
  """A decorator for registering the statistics function for an op type.

  This decorator can be defined for an op type so that it gives a
  report on the resources used by an instance of an operator, in the
  form of an OpStats object.

  Well-known types of statistics include these so far:

  - flops: When running a graph, the bulk of the computation happens doing
    numerical calculations like matrix multiplications. This type allows a node
    to return how many floating-point operations it takes to complete. The
    total number of FLOPs for a graph is a good guide to its expected latency.

  You can add your own statistics just by picking a new type string, registering
  functions for the ops you care about, and then calling get_stats_for_node_def.

  If a statistic for an op is registered multiple times, a KeyError will be
  raised.

  Since the statistics is counted on a per-op basis. It is not suitable for
  model parameters (capacity), which is expected to be counted only once, even
  if it is shared by multiple ops. (e.g. RNN)

  For example, you can define a new metric called doohickey for a Foo operation
  by placing this in your code:

  ```python
  @ops.RegisterStatistics("Foo", "doohickey")
  def _calc_foo_bojangles(unused_graph, unused_node_def):
    return ops.OpStats("doohickey", 20)
  ```

  Then in client code you can retrieve the value by making this call:

  ```python
  doohickey = ops.get_stats_for_node_def(graph, node_def, "doohickey")
  ```

  If the NodeDef is for an op with a registered doohickey function, you'll get
  back the calculated amount in doohickey.value, or None if it's not defined.

  """

  def __init__(self, op_type, statistic_type):
    """Saves the `op_type` as the `Operation` type."""
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string.")
    if "," in op_type:
      raise TypeError("op_type must not contain a comma.")
    self._op_type = op_type
    if not isinstance(statistic_type, six.string_types):
      raise TypeError("statistic_type must be a string.")
    if "," in statistic_type:
      raise TypeError("statistic_type must not contain a comma.")
    self._statistic_type = statistic_type

  def __call__(self, f):
    """Registers "f" as the statistics function for "op_type"."""
    _stats_registry.register(f, self._op_type + "," + self._statistic_type)
    return f


def get_stats_for_node_def(graph, node, statistic_type):
  """Looks up the node's statistics function in the registry and calls it.

  This function takes a Graph object and a NodeDef from a GraphDef, and if
  there's an associated statistics method, calls it and returns a result. If no
  function has been registered for the particular node type, it returns an empty
  statistics object.

  Args:
    graph: A Graph object that's been set up with the node's graph.
    node: A NodeDef describing the operator.
    statistic_type: A string identifying the statistic we're interested in.
  Returns:
    An OpStats object containing information about resource usage.
  """

  try:
    stats_func = _stats_registry.lookup(node.op + "," + statistic_type)
    result = stats_func(graph, node)
  except LookupError:
    result = OpStats(statistic_type)
  return result


def _name_from_scope_name(name):
  """Returns the name of an op given the name of its scope.

  Args:
    name: the name of the scope.

  Returns:
    the name of the op (equal to scope name minus any trailing slash).
  """
  return name[:-1] if name[-1] == "/" else name


class _ScopedTF_Graph(object):

  def __init__(self):
    self.graph = c_api.TF_NewGraph()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api.TF_DeleteGraph is not None:
      c_api.TF_DeleteGraph(self.graph)


class Graph(object):
  """A TensorFlow computation, represented as a dataflow graph.

  A `Graph` contains a set of
  @{tf.Operation} objects,
  which represent units of computation; and
  @{tf.Tensor} objects, which represent
  the units of data that flow between operations.

  A default `Graph` is always registered, and accessible by calling
  @{tf.get_default_graph}.
  To add an operation to the default graph, simply call one of the functions
  that defines a new `Operation`:

  ```python
  c = tf.constant(4.0)
  assert c.graph is tf.get_default_graph()
  ```

  Another typical usage involves the
  @{tf.Graph.as_default}
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

  A `Graph` instance supports an arbitrary number of "collections"
  that are identified by name. For convenience when building a large
  graph, collections can store groups of related objects: for
  example, the `tf.Variable` uses a collection (named
  @{tf.GraphKeys.GLOBAL_VARIABLES}) for
  all variables that are created during the construction of a graph. The caller
  may define additional collections by specifying a new name.
  """

  def __init__(self):
    """Creates a new, empty Graph."""
    # Protects the core state that may be accessed by multiple readers.
    # Only state that can be returned via public accessors (`as_graph_def()`,
    # `get_operations()`, `as_graph_element()`, `get_collection()`, and
    # `get_collection_ref()`) is by the lock. Thread-safety is provided on a
    # best-effort basis to support buggy programs, and is not guaranteed by the
    # public `tf.Graph` API.
    # NOTE(mrry): This does not protect the various stacks. A warning will
    # be reported if these are used from multiple threads
    self._lock = threading.Lock()
    self._nodes_by_id = dict()  # GUARDED_BY(self._lock)
    self._next_id_counter = 0  # GUARDED_BY(self._lock)
    self._nodes_by_name = dict()  # GUARDED_BY(self._lock)
    self._version = 0  # GUARDED_BY(self._lock)
    # Current name stack: uniquified names
    self._name_stack = ""
    # Maps a name used in the graph to the next id to use for that name.
    self._names_in_use = {}
    # Functions that will be applied to choose a device if none is specified.
    self._device_function_stack = []
    # Default original_op applied to new ops.
    self._default_original_op = None
    # Current control flow context. It could be either CondContext or
    # WhileContext defined in ops/control_flow_ops.py
    self._control_flow_context = None
    # A new node will depend of the union of all of the nodes in the stack.
    self._control_dependencies_stack = []
    # Arbritrary collections of objects.
    self._collections = {}
    # The graph-level random seed
    self._seed = None
    # A dictionary of attributes that should be applied to all ops.
    self._attr_scope_map = {}
    # A map from op type to the kernel label that should be used.
    self._op_to_kernel_label_map = {}
    # A map from op type to an alternative op type that should be used when
    # computing gradients.
    self._gradient_override_map = {}
    # True if the graph is considered "finalized".  In that case no
    # new operations can be added.
    self._finalized = False
    # Functions defined in the graph
    self._functions = collections.OrderedDict()
    # Default GraphDef versions
    self._graph_def_versions = versions_pb2.VersionDef(
        producer=versions.GRAPH_DEF_VERSION,
        min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER)
    self._building_function = False
    # Stack of colocate_with ops
    self._colocation_stack = []
    # Set of tensors that are dangerous to feed!
    self._unfeedable_tensors = set()
    # Set of operations that are dangerous to fetch!
    self._unfetchable_ops = set()
    # A map of tensor handle placeholder to tensor dtype.
    self._handle_feeders = {}
    # A map from tensor handle to its read op.
    self._handle_readers = {}
    # A map from tensor handle to its move op.
    self._handle_movers = {}
    # A map from tensor handle to its delete op.
    self._handle_deleters = {}
    # Resource container.
    self._container = ""
    self._registered_ops = op_def_registry.get_registered_ops()

    # TODO(skyewm): fold as much of the above as possible into the C
    # implementation
    if _USE_C_API:
      self._scoped_c_graph = _ScopedTF_Graph()
    else:
      self._scoped_c_graph = None

  def _convert_stack(self, stack, include_func_start_lineno=False):
    """Converts a stack extracted using _extract_stack() to a traceback stack.

    Args:
      stack: A list of n 5-tuples,
        (filename, lineno, name, frame_globals, func_start_lineno).
      include_func_start_lineno: True if function start line number should be
        included as the 5th entry in return tuples.

    Returns:
      A list of n 4-tuples or 5-tuples
      (filename, lineno, name, code, [optional: func_start_lineno]), where the
      code tuple element is calculated from the corresponding elements of the
      input tuple.
    """
    ret = []
    for (filename, lineno, name, frame_globals, func_start_lineno,
         unused_frame_info) in stack:
      linecache.checkcache(filename)
      line = linecache.getline(filename, lineno, frame_globals)
      if line:
        line = line.strip()
      else:
        line = None
      if include_func_start_lineno:
        ret.append((filename, lineno, name, line, func_start_lineno))
      else:
        ret.append((filename, lineno, name, line))
    return ret

  def _extract_stack(self):
    """A lightweight, extensible re-implementation of traceback.extract_stack.

    NOTE(mrry): traceback.extract_stack eagerly retrieves the line of code for
      each stack frame using linecache, which results in an abundance of stat()
      calls. This implementation does not retrieve the code, and any consumer
      should apply _convert_stack to the result to obtain a traceback that can
      be formatted etc. using traceback methods.

    Derived classes can implement _extract_frame_info() to add extra information
    to the traceback.

    Returns:
      A list of 6-tuples
      (filename, lineno, name, frame_globals, func_start_lineno, custom_info)
      corresponding to the call stack of the current thread.
    """
    try:
      raise ZeroDivisionError
    except ZeroDivisionError:
      f = sys.exc_info()[2].tb_frame.f_back
    ret = []
    while f is not None:
      lineno = f.f_lineno
      co = f.f_code
      filename = co.co_filename
      name = co.co_name
      frame_globals = f.f_globals
      func_start_lineno = co.co_firstlineno
      frame_info = self._extract_frame_info(f)
      ret.append((filename, lineno, name, frame_globals, func_start_lineno,
                  frame_info))
      f = f.f_back
    ret.reverse()
    return ret

  def _extract_frame_info(self, frame):  # pylint: disable=unused-argument
    """Extracts custom information from a frame in an op traceback."""
    return None

  def _check_not_finalized(self):
    """Check if the graph is finalized.

    Raises:
      RuntimeError: If the graph finalized.
    """
    if self._finalized:
      raise RuntimeError("Graph is finalized and cannot be modified.")

  def _add_op(self, op):
    """Adds 'op' to the graph.

    Args:
      op: the Operator or Tensor to add.

    Raises:
      TypeError: if op is not an Operation or Tensor.
      ValueError: if the op.name or op._id are already used.
    """
    self._check_not_finalized()
    if not isinstance(op, (Tensor, Operation)):
      raise TypeError("op must be a Tensor or Operation: %s" % op)
    with self._lock:
      # pylint: disable=protected-access
      if op._id in self._nodes_by_id:
        raise ValueError("cannot add an op with id %d as it already "
                         "exists in the graph" % op._id)
      if op.name in self._nodes_by_name:
        raise ValueError("cannot add op with name %s as that name "
                         "is already used" % op.name)
      self._nodes_by_id[op._id] = op
      self._nodes_by_name[op.name] = op
      self._version = max(self._version, op._id)
      # pylint: enable=protected-access

  @property
  def _c_graph(self):
    if self._scoped_c_graph:
      return self._scoped_c_graph.graph
    return None

  @property
  def version(self):
    """Returns a version number that increases as ops are added to the graph.

    Note that this is unrelated to the
    @{tf.Graph.graph_def_versions}.
    """
    if self._finalized:
      return self._version

    with self._lock:
      return self._version

  @property
  def graph_def_versions(self):
    # pylint: disable=line-too-long
    """The GraphDef version information of this graph.

    For details on the meaning of each version, see
    [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto).

    Returns:
      A `VersionDef`.
    """
    # pylint: enable=line-too-long
    return self._graph_def_versions

  @property
  def seed(self):
    """The graph-level random seed of this graph."""
    return self._seed

  @seed.setter
  def seed(self, seed):
    self._seed = seed

  @property
  def finalized(self):
    """True if this graph has been finalized."""
    return self._finalized

  def finalize(self):
    """Finalizes this graph, making it read-only.

    After calling `g.finalize()`, no new operations can be added to
    `g`.  This method is used to ensure that no operations are added
    to a graph when it is shared between multiple threads, for example
    when using a @{tf.train.QueueRunner}.
    """
    self._finalized = True

  def _unsafe_unfinalize(self):
    """Opposite of `finalize`. Internal interface.

    NOTE: Unfinalizing a graph could have negative impact on performance,
    especially in a multi-threaded environment.  Unfinalizing a graph
    when it is in use by a Session may lead to undefined behavior. Ensure
    that all sessions using a graph are closed before calling this method.
    """
    self._finalized = False

  def _get_control_flow_context(self):
    """Returns the current control flow context.

    Returns:
      A context object.
    """
    return self._control_flow_context

  def _set_control_flow_context(self, context):
    """Sets the current control flow context.

    Args:
      context: a context object.
    """
    self._control_flow_context = context

  def _as_graph_def(self, from_version=None, add_shapes=False):
    # pylint: disable=line-too-long
    """Returns a serialized `GraphDef` representation of this graph.

    The serialized `GraphDef` can be imported into another `Graph`
    (using @{tf.import_graph_def}) or used with the
    [C++ Session API](../../../../api_docs/cc/index.md).

    This method is thread-safe.

    Args:
      from_version: Optional.  If this is set, returns a `GraphDef`
        containing only the nodes that were added to this graph since
        its `version` property had the given value.
      add_shapes: If true, adds an "_output_shapes" list attr to each
        node with the inferred shapes of each of its outputs.

    Returns:
      A tuple containing a
      [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer, and the version of the graph to which that
      `GraphDef` corresponds.

    Raises:
      ValueError: If the `graph_def` would be too large.

    """
    # pylint: enable=line-too-long
    with self._lock:
      graph = graph_pb2.GraphDef()
      graph.versions.CopyFrom(self._graph_def_versions)
      bytesize = 0
      for op_id in sorted(self._nodes_by_id):
        op = self._nodes_by_id[op_id]
        if from_version is None or op_id > from_version:
          graph.node.extend([op.node_def])
          if op.outputs and add_shapes:
            assert "_output_shapes" not in graph.node[-1].attr
            graph.node[-1].attr["_output_shapes"].list.shape.extend([
                output.get_shape().as_proto() for output in op.outputs])
          bytesize += op.node_def.ByteSize()
          if bytesize >= (1 << 31) or bytesize < 0:
            raise ValueError("GraphDef cannot be larger than 2GB.")
      if self._functions:
        for f in self._functions.values():
          bytesize += f.definition.ByteSize()
          if bytesize >= (1 << 31) or bytesize < 0:
            raise ValueError("GraphDef cannot be larger than 2GB.")
          graph.library.function.extend([f.definition])
          if f.grad_func_name:
            grad_def = function_pb2.GradientDef()
            grad_def.function_name = f.name
            grad_def.gradient_func = f.grad_func_name
            graph.library.gradient.extend([grad_def])
      return graph, self._version

  def as_graph_def(self, from_version=None, add_shapes=False):
    # pylint: disable=line-too-long
    """Returns a serialized `GraphDef` representation of this graph.

    The serialized `GraphDef` can be imported into another `Graph`
    (using @{tf.import_graph_def}) or used with the
    [C++ Session API](../../api_docs/cc/index.md).

    This method is thread-safe.

    Args:
      from_version: Optional.  If this is set, returns a `GraphDef`
        containing only the nodes that were added to this graph since
        its `version` property had the given value.
      add_shapes: If true, adds an "_output_shapes" list attr to each
        node with the inferred shapes of each of its outputs.

    Returns:
      A [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer.

    Raises:
      ValueError: If the `graph_def` would be too large.
    """
    # pylint: enable=line-too-long
    result, _ = self._as_graph_def(from_version, add_shapes)
    return result

  def _is_function(self, name):
    """Tests whether 'name' is registered in this graph's function library.

    Args:
      name: string op name.
    Returns:
      bool indicating whether or not 'name' is registered in function library.
    """
    return name in self._functions

  def _get_function(self, name):
    """Returns the function definition for 'name'.

    Args:
      name: string function name.
    Returns:
      The function def proto.
    """
    return self._functions.get(name, None)

  def _add_function(self, function):
    """Adds a function to the graph.

    After the function has been added, you can call to the function by
    passing the function name in place of an op name to
    `Graph.create_op()`.

    Args:
      function: A `_DefinedFunction` object.


    Raises:
      ValueError: if another function is defined with the same name.
    """
    name = function.name
    previous = self._functions.get(name, None)
    if previous:
      raise ValueError("Another function is already defined with that name")
    # Sanity checks on gradient definition.
    if (function.grad_func_name is not None) and (
        function.python_grad_func is not None):
      raise ValueError("Gradient defined twice for function %s" % name)
    # Need a new-enough consumer to support the functions we add to the graph.
    if self._graph_def_versions.min_consumer < 12:
      self._graph_def_versions.min_consumer = 12
    self._functions[name] = function

  @property
  def building_function(self):
    """Returns True iff this graph represents a function."""
    return self._building_function

  # Helper functions to create operations.
  def create_op(self, op_type, inputs, dtypes,
                input_types=None, name=None, attrs=None, op_def=None,
                compute_shapes=True, compute_device=True):
    """Creates an `Operation` in this graph.

    This is a low-level interface for creating an `Operation`. Most
    programs will not call this method directly, and instead use the
    Python op constructors, such as `tf.constant()`, which add ops to
    the default graph.

    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: A list of `DType` objects that will be the types of the tensors
        that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of
        the tensors that the operation consumes. By default, uses the base
        `DType` of each input in `inputs`. Operations that expect
        reference-typed inputs must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_shapes: (Optional.) If True, shape inference will be performed
        to compute the shapes of the outputs.
      compute_device: (Optional.) If True, device functions will be executed
        to compute the device property of the Operation.

    Raises:
      TypeError: if any of the inputs is not a `Tensor`.
      ValueError: if colocation conflicts with existing device assignment.

    Returns:
      An `Operation` object.

    """
    self._check_not_finalized()
    for idx, a in enumerate(inputs):
      if not isinstance(a, Tensor):
        raise TypeError("Input #%d is not a tensor: %s" % (idx, a))
    if name is None:
      name = op_type
    # If a names ends with a '/' it is a "name scope" and we use it as-is,
    # after removing the trailing '/'.
    if name and name[-1] == "/":
      name = _name_from_scope_name(name)
    else:
      name = self.unique_name(name)

    node_def = _NodeDef(op_type, name, device=None, attrs=attrs)

    # Apply any additional attributes requested. Do not overwrite any existing
    # attributes.
    for key, value in self._attr_scope_map.items():
      if key not in node_def.attr:
        if callable(value):
          value = value(node_def)
          if not isinstance(value, (type(None), attr_value_pb2.AttrValue)):
            raise TypeError(
                "Callable for scope map key '%s' must return either None or "
                "an AttrValue protocol buffer; but it returned: %s" %
                (key, value))
        node_def.attr[key].CopyFrom(value)

    # Apply a kernel label if one has been specified for this op_type.
    try:
      kernel_label = self._op_to_kernel_label_map[op_type]
      node_def.attr["_kernel"].CopyFrom(
          attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:
      pass

    # Apply the overriding op_type for gradients if one has been
    # specified for this op_type.
    try:
      mapped_op_type = self._gradient_override_map[op_type]
      node_def.attr["_gradient_op_type"].CopyFrom(
          attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
      pass

    control_inputs = self._control_dependencies_for_inputs(inputs)
    ret = Operation(node_def, self, inputs=inputs, output_types=dtypes,
                    control_inputs=control_inputs, input_types=input_types,
                    original_op=self._default_original_op, op_def=op_def)
    if compute_shapes:
      set_shapes_for_outputs(ret)
    self._add_op(ret)
    self._record_op_seen_by_control_dependencies(ret)

    if compute_device:
      self._apply_device_functions(ret)

    if self._colocation_stack:
      all_colocation_groups = []
      for colocation_op in self._colocation_stack:
        all_colocation_groups.extend(colocation_op.colocation_groups())
        if colocation_op.device:
          # Make this device match the device of the colocated op, to
          # provide consistency between the device and the colocation
          # property.
          if ret.device and ret.device != colocation_op.device:
            logging.warning("Tried to colocate %s with an op %s that had "
                            "a different device: %s vs %s. "
                            "Ignoring colocation property.",
                            name, colocation_op.name,
                            ret.device, colocation_op.device)
          else:
            ret._set_device(colocation_op.device)

      all_colocation_groups = sorted(set(all_colocation_groups))
      ret.node_def.attr["_class"].CopyFrom(attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))

    # Sets "container" attribute if
    # (1) self._container is not None
    # (2) "is_stateful" is set in OpDef
    # (3) "container" attribute is in OpDef
    # (4) "container" attribute is None
    if (self._container and
        op_type in self._registered_ops and
        self._registered_ops[op_type].is_stateful and
        "container" in ret.node_def.attr and
        not ret.node_def.attr["container"].s):
      ret.node_def.attr["container"].CopyFrom(
          attr_value_pb2.AttrValue(s=compat.as_bytes(self._container)))

    return ret

  def as_graph_element(self, obj, allow_tensor=True, allow_operation=True):
    """Returns the object referred to by `obj`, as an `Operation` or `Tensor`.

    This function validates that `obj` represents an element of this
    graph, and gives an informative error message if it is not.

    This function is the canonical way to get/validate an object of
    one of the allowed types from an external argument reference in the
    Session API.

    This method may be called concurrently from multiple threads.

    Args:
      obj: A `Tensor`, an `Operation`, or the name of a tensor or operation.
        Can also be any object with an `_as_graph_element()` method that returns
        a value of one of these types.
      allow_tensor: If true, `obj` may refer to a `Tensor`.
      allow_operation: If true, `obj` may refer to an `Operation`.

    Returns:
      The `Tensor` or `Operation` in the Graph corresponding to `obj`.

    Raises:
      TypeError: If `obj` is not a type we support attempting to convert
        to types.
      ValueError: If `obj` is of an appropriate type but invalid. For
        example, an invalid string.
      KeyError: If `obj` is not an object in the graph.
    """
    if self._finalized:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

    with self._lock:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

  def _as_graph_element_locked(self, obj, allow_tensor, allow_operation):
    """See `Graph.as_graph_element()` for details."""
    # The vast majority of this function is figuring
    # out what an API user might be doing wrong, so
    # that we can give helpful error messages.
    #
    # Ideally, it would be nice to split it up, but we
    # need context to generate nice error messages.

    if allow_tensor and allow_operation:
      types_str = "Tensor or Operation"
    elif allow_tensor:
      types_str = "Tensor"
    elif allow_operation:
      types_str = "Operation"
    else:
      raise ValueError("allow_tensor and allow_operation can't both be False.")

    temp_obj = _as_graph_element(obj)
    if temp_obj is not None:
      obj = temp_obj

    # If obj appears to be a name...
    if isinstance(obj, compat.bytes_or_text_types):
      name = compat.as_str(obj)

      if ":" in name and allow_tensor:
        # Looks like a Tensor name and can be a Tensor.
        try:
          op_name, out_n = name.split(":")
          out_n = int(out_n)
        except:
          raise ValueError("The name %s looks a like a Tensor name, but is "
                           "not a valid one. Tensor names must be of the "
                           "form \"<op_name>:<output_index>\"." % repr(name))
        if op_name in self._nodes_by_name:
          op = self._nodes_by_name[op_name]
        else:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, does not exist in the "
                         "graph." % (repr(name), repr(op_name)))
        try:
          return op.outputs[out_n]
        except:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, exists but only has "
                         "%s outputs."
                         % (repr(name), repr(op_name), len(op.outputs)))

      elif ":" in name and not allow_tensor:
        # Looks like a Tensor name but can't be a Tensor.
        raise ValueError("Name %s appears to refer to a Tensor, not a %s."
                         % (repr(name), types_str))

      elif ":" not in name and allow_operation:
        # Looks like an Operation name and can be an Operation.
        if name not in self._nodes_by_name:
          raise KeyError("The name %s refers to an Operation not in the "
                         "graph." % repr(name))
        return self._nodes_by_name[name]

      elif ":" not in name and not allow_operation:
        # Looks like an Operation name but can't be an Operation.
        if name in self._nodes_by_name:
          # Yep, it's an Operation name
          err_msg = ("The name %s refers to an Operation, not a %s."
                     % (repr(name), types_str))
        else:
          err_msg = ("The name %s looks like an (invalid) Operation name, "
                     "not a %s." % (repr(name), types_str))
        err_msg += (" Tensor names must be of the form "
                    "\"<op_name>:<output_index>\".")
        raise ValueError(err_msg)

    elif isinstance(obj, Tensor) and allow_tensor:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Tensor %s is not an element of this graph." % obj)
      return obj
    elif isinstance(obj, Operation) and allow_operation:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Operation %s is not an element of this graph." % obj)
      return obj
    else:
      # We give up!
      raise TypeError("Can not convert a %s into a %s."
                      % (type(obj).__name__, types_str))

  def get_operations(self):
    """Return the list of operations in the graph.

    You can modify the operations in place, but modifications
    to the list such as inserts/delete have no effect on the
    list of operations known to the graph.

    This method may be called concurrently from multiple threads.

    Returns:
      A list of Operations.
    """
    if self._finalized:
      return list(self._nodes_by_id.values())

    with self._lock:
      return list(self._nodes_by_id.values())

  def get_operation_by_name(self, name):
    """Returns the `Operation` with the given `name`.

    This method may be called concurrently from multiple threads.

    Args:
      name: The name of the `Operation` to return.

    Returns:
      The `Operation` with the given `name`.

    Raises:
      TypeError: If `name` is not a string.
      KeyError: If `name` does not correspond to an operation in this graph.
    """

    if not isinstance(name, six.string_types):
      raise TypeError("Operation names are strings (or similar), not %s."
                      % type(name).__name__)
    return self.as_graph_element(name, allow_tensor=False, allow_operation=True)

  def _get_operation_by_name_unsafe(self, name):
    """Returns the `Operation` with the given `name`.

    This is a internal unsafe version of get_operation_by_name. It skips many
    checks and does not have user friedly error messages but runs considerably
    faster. This method may be called concurrently from multiple threads.

    Args:
      name: The name of the `Operation` to return.

    Returns:
      The `Operation` with the given `name`.

    Raises:
      KeyError: If `name` does not correspond to an operation in this graph.
    """

    if self._finalized:
      return self._nodes_by_name[name]

    with self._lock:
      return self._nodes_by_name[name]

  def get_tensor_by_name(self, name):
    """Returns the `Tensor` with the given `name`.

    This method may be called concurrently from multiple threads.

    Args:
      name: The name of the `Tensor` to return.

    Returns:
      The `Tensor` with the given `name`.

    Raises:
      TypeError: If `name` is not a string.
      KeyError: If `name` does not correspond to a tensor in this graph.
    """
    # Names should be strings.
    if not isinstance(name, six.string_types):
      raise TypeError("Tensor names are strings (or similar), not %s."
                      % type(name).__name__)
    return self.as_graph_element(name, allow_tensor=True, allow_operation=False)

  def _next_id(self):
    """Id for next Operation instance. Also increments the internal id."""
    self._check_not_finalized()
    with self._lock:
      self._next_id_counter += 1
      return self._next_id_counter

  @property
  def _last_id(self):
    return self._next_id_counter

  def as_default(self):
    """Returns a context manager that makes this `Graph` the default graph.

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

    Returns:
      A context manager for using this graph as the default graph.
    """
    return _default_graph_stack.get_controller(self)

  @property
  def collections(self):
    """Returns the names of the collections known to this graph."""
    return list(self._collections)

  def add_to_collection(self, name, value):
    """Stores `value` in the collection with the given `name`.

    Note that collections are not sets, so it is possible to add a value to
    a collection several times.

    Args:
      name: The key for the collection. The `GraphKeys` class
        contains many standard names for collections.
      value: The value to add to the collection.
    """
    self._check_not_finalized()
    with self._lock:
      if name not in self._collections:
        self._collections[name] = [value]
      else:
        self._collections[name].append(value)

  def add_to_collections(self, names, value):
    """Stores `value` in the collections given by `names`.

    Note that collections are not sets, so it is possible to add a value to
    a collection several times. This function makes sure that duplicates in
    `names` are ignored, but it will not check for pre-existing membership of
    `value` in any of the collections in `names`.

    `names` can be any iterable, but if `names` is a string, it is treated as a
    single collection name.

    Args:
      names: The keys for the collections to add to. The `GraphKeys` class
        contains many standard names for collections.
      value: The value to add to the collections.
    """
    # Make sure names are unique, but treat strings as a single collection name
    names = (names,) if isinstance(names, six.string_types) else set(names)
    for name in names:
      self.add_to_collection(name, value)

  def get_collection_ref(self, name):
    """Returns a list of values in the collection with the given `name`.

    If the collection exists, this returns the list itself, which can
    be modified in place to change the collection.  If the collection does
    not exist, it is created as an empty list and the list is returned.

    This is different from `get_collection()` which always returns a copy of
    the collection list if it exists and never creates an empty collection.

    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.

    Returns:
      The list of values in the collection with the given `name`, or an empty
      list if no value has been added to that collection.
    """
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        coll_list = []
        self._collections[name] = coll_list
      return coll_list

  def get_collection(self, name, scope=None):
    """Returns a list of values in the collection with the given `name`.

    This is different from `get_collection_ref()` which always returns the
    actual collection list if it exists in that it returns a new list each time
    it is called.

    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.
      scope: (Optional.) A string. If supplied, the resulting list is filtered
        to include only items whose `name` attribute matches `scope` using
        `re.match`. Items without a `name` attribute are never returned if a
        scope is supplied. The choice of `re.match` means that a `scope` without
        special tokens filters by prefix.

    Returns:
      The list of values in the collection with the given `name`, or
      an empty list if no value has been added to that collection. The
      list contains the values in the order under which they were
      collected.
    """
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        return []
      if scope is None:
        return list(coll_list)
      else:
        c = []
        regex = re.compile(scope)
        for item in coll_list:
          if hasattr(item, "name") and regex.match(item.name):
            c.append(item)
        return c

  def get_all_collection_keys(self):
    """Returns a list of collections used in this graph."""
    with self._lock:
      return [x for x in self._collections if isinstance(x, six.string_types)]

  def clear_collection(self, name):
    """Clears all values in a collection.

    Args:
      name: The key for the collection. The `GraphKeys` class contains many
        standard names for collections.
    """
    self._check_not_finalized()
    with self._lock:
      if name in self._collections:
        del self._collections[name]

  @tf_contextlib.contextmanager
  def _original_op(self, op):
    """Python 'with' handler to help annotate ops with their originator.

    An op may have an 'original_op' property that indicates the op on which
    it was based. For example a replica op is based on the op that was
    replicated and a gradient op is based on the op that was differentiated.

    All ops created in the scope of this 'with' handler will have
    the given 'op' as their original op.

    Args:
      op: The Operation that all ops created in this scope will have as their
        original op.

    Yields:
      Nothing.
    """
    old_original_op = self._default_original_op
    try:
      self._default_original_op = op
      yield
    finally:
      self._default_original_op = old_original_op

  # pylint: disable=g-doc-return-or-yield,line-too-long
  @tf_contextlib.contextmanager
  def name_scope(self, name):
    r"""Returns a context manager that creates hierarchical names for operations.

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

    Args:
      name: A name for the scope.

    Returns:
      A context manager that installs `name` as a new name scope.

    Raises:
      ValueError: If `name` is not a valid scope name, according to the rules
        above.
    """
    if name:
      if self._name_stack:
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        if not _VALID_SCOPE_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
      else:
        # Scopes created in the root must match the more restrictive
        # op name regex, which constrains the initial character.
        if not _VALID_OP_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
    try:
      old_stack = self._name_stack
      if not name:  # Both for name=None and name="" we re-set to empty scope.
        new_stack = None
      elif name and name[-1] == "/":
        new_stack = _name_from_scope_name(name)
      else:
        new_stack = self.unique_name(name)
      self._name_stack = new_stack
      yield "" if new_stack is None else new_stack + "/"
    finally:
      self._name_stack = old_stack
  # pylint: enable=g-doc-return-or-yield,line-too-long

  def unique_name(self, name, mark_as_used=True):
    """Return a unique operation name for `name`.

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

    Args:
      name: The name for an operation.
      mark_as_used: Whether to mark this name as being used.

    Returns:
      A string to be passed to `create_op()` that will be used
      to name the operation being created.
    """
    if self._name_stack:
      name = self._name_stack + "/" + name
    i = self._names_in_use.get(name, 0)
    # Increment the number for "name".
    if mark_as_used:
      self._names_in_use[name] = i + 1
    if i > 0:
      base_name = name
      # Make sure the composed name is not already used.
      while name in self._names_in_use:
        name = "%s_%d" % (base_name, i)
        i += 1
      # Mark the composed name as used in case someone wants
      # to call unique_name("name_1").
      if mark_as_used:
        self._names_in_use[name] = 1
    return name

  def get_name_scope(self):
    """Returns the current name scope.

    For example:

    ```python
    with tf.name_scope('scope1'):
      with tf.name_scope('scope2'):
        print(tf.get_default_graph().get_name_scope())
    ```
    would print the string `scope1/scope2`.

    Returns:
      A string representing the current name scope.
    """
    return self._name_stack

  @tf_contextlib.contextmanager
  def colocate_with(self, op, ignore_existing=False):
    """Returns a context manager that specifies an op to colocate with.

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

    Args:
      op: The op to colocate all created ops with, or `None`.
      ignore_existing: If true, only applies colocation of this op within
        the context, rather than applying all colocation properties
        on the stack.  If `op` is `None`, this value must be `True`.

    Raises:
      ValueError: if op is None but ignore_existing is False.

    Yields:
      A context manager that specifies the op with which to colocate
      newly created ops.

    """
    if op is None and not ignore_existing:
      raise ValueError(
          "Trying to reset colocation (op is None) but "
          "ignore_existing is not True")

    if op is not None and not isinstance(op, Operation):
      # We always want to colocate with the reference op.
      op = internal_convert_to_tensor_or_indexed_slices(op, as_ref=True).op

    # By default, colocate_with resets the device function stack,
    # since colocate_with is typically used in specific internal
    # library functions where colocation is intended to be "stronger"
    # than device functions.
    #
    # In the future, a caller may specify that device_functions win
    # over colocation, in which case we can add support.
    device_fn_tmp = self._device_function_stack
    self._device_function_stack = []

    if ignore_existing:
      current_stack = self._colocation_stack
      self._colocation_stack = []

    if op is not None:
      self._colocation_stack.append(op)

    try:
      yield
    finally:
      # Restore device function stack
      self._device_function_stack = device_fn_tmp
      if op is not None:
        self._colocation_stack.pop()

      # Reset the colocation stack if requested.
      if ignore_existing:
        self._colocation_stack = current_stack

  @tf_contextlib.contextmanager
  def device(self, device_name_or_function):
    # pylint: disable=line-too-long
    """Returns a context manager that specifies the default device to use.

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

    Args:
      device_name_or_function: The device name or function to use in
        the context.

    Returns:
      A context manager that specifies the default device to use for newly
      created ops.

    """
    # pylint: enable=line-too-long
    if (device_name_or_function is not None
        and not callable(device_name_or_function)):
      device_function = pydev.merge_device(device_name_or_function)
    else:
      device_function = device_name_or_function

    try:
      self._device_function_stack.append(device_function)
      yield
    finally:
      self._device_function_stack.pop()

  def _apply_device_functions(self, op):
    """Applies the current device function stack to the given operation."""
    # Apply any device functions in reverse order, so that the most recently
    # pushed function has the first chance to apply a device to the op.
    # We apply here because the result can depend on the Operation's
    # signature, which is computed in the Operation constructor.
    for device_function in reversed(self._device_function_stack):
      if device_function is None:
        break
      op._set_device(device_function(op))

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def container(self, container_name):
    """Returns a context manager that specifies the resource container to use.

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

    Args:
      container_name: container name string.

    Returns:
      A context manager for defining resource containers for stateful ops,
        yields the container name.
    """
    original_container = self._container
    try:
      self._container = container_name
      yield self._container
    finally:
      self._container = original_container
  # pylint: enable=g-doc-return-or-yield

  class _ControlDependenciesController(object):
    """Context manager for `control_dependencies()`."""

    def __init__(self, graph, control_inputs):
      """Create a new `_ControlDependenciesController`.

      A `_ControlDependenciesController` is the context manager for
      `with tf.control_dependencies()` blocks.  These normally nest,
      as described in the documentation for `control_dependencies()`.

      The `control_inputs` argument list control dependencies that must be
      added to the current set of control dependencies.  Because of
      uniquification the set can be empty even if the caller passed a list of
      ops.  The special value `None` indicates that we want to start a new
      empty set of control dependencies instead of extending the current set.

      In that case we also clear the current control flow context, which is an
      additional mechanism to add control dependencies.

      Args:
        graph: The graph that this controller is  managing.
        control_inputs: List of ops to use as control inputs in addition
          to the current control dependencies.  None to indicate that
          the dependencies should be cleared.
      """
      self._graph = graph
      if control_inputs is None:
        self._control_inputs = []
        self._new_stack = True
      else:
        self._control_inputs = control_inputs
        self._new_stack = False
      self._seen_nodes = set()
      self._old_stack = None
      self._old_control_flow_context = None

# pylint: disable=protected-access
    def __enter__(self):
      if self._new_stack:
        # Clear the control_dependencies graph.
        self._old_stack = self._graph._control_dependencies_stack
        self._graph._control_dependencies_stack = []
        # Clear the control_flow_context too.
        self._old_control_flow_context = self._graph._get_control_flow_context()
        self._graph._set_control_flow_context(None)
      self._graph._push_control_dependencies_controller(self)

    def __exit__(self, unused_type, unused_value, unused_traceback):
      self._graph._pop_control_dependencies_controller(self)
      if self._new_stack:
        self._graph._control_dependencies_stack = self._old_stack
        self._graph._set_control_flow_context(self._old_control_flow_context)
# pylint: enable=protected-access

    @property
    def control_inputs(self):
      return self._control_inputs

    def add_op(self, op):
      self._seen_nodes.add(op)

    def op_in_group(self, op):
      return op in self._seen_nodes

  def _push_control_dependencies_controller(self, controller):
    self._control_dependencies_stack.append(controller)

  def _pop_control_dependencies_controller(self, controller):
    assert self._control_dependencies_stack[-1] is controller
    self._control_dependencies_stack.pop()

  def _current_control_dependencies(self):
    ret = set()
    for controller in self._control_dependencies_stack:
      for op in controller.control_inputs:
        ret.add(op)
    return ret

  def _control_dependencies_for_inputs(self, input_tensors):
    """For an op that takes `input_tensors` as inputs, compute control inputs.

    The returned control dependencies should yield an execution that
    is equivalent to adding all control inputs in
    self._control_dependencies_stack to a newly created op. However,
    this function attempts to prune the returned control dependencies
    by observing that nodes created within the same `with
    control_dependencies(...):` block may have data dependencies that make
    the explicit approach redundant.

    Args:
      input_tensors: The direct data dependencies for an op to be created.

    Returns:
      A list of control inputs for the op to be created.
    """
    ret = []
    input_ops = set([t.op for t in input_tensors])
    for controller in self._control_dependencies_stack:
      # If any of the input_ops already depends on the inputs from controller,
      # we say that the new op is dominated (by that input), and we therefore
      # do not need to add control dependencies for this controller's inputs.
      dominated = False
      for op in input_ops:
        if controller.op_in_group(op):
          dominated = True
          break
      if not dominated:
        # Don't add a control input if we already have a data dependency on i.
        # NOTE(mrry): We do not currently track transitive data dependencies,
        #   so we may add redundant control inputs.
        ret.extend([c for c in controller.control_inputs if c not in input_ops])
    return ret

  def _record_op_seen_by_control_dependencies(self, op):
    """Record that the given op depends on all registered control dependencies.

    Args:
      op: An Operation.
    """
    for controller in self._control_dependencies_stack:
      controller.add_op(op)

  def control_dependencies(self, control_inputs):
    """Returns a context manager that specifies control dependencies.

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

    Args:
      control_inputs: A list of `Operation` or `Tensor` objects which
        must be executed or computed before running the operations
        defined in the context.  Can also be `None` to clear the control
        dependencies.

    Returns:
     A context manager that specifies control dependencies for all
     operations constructed within the context.

    Raises:
      TypeError: If `control_inputs` is not a list of `Operation` or
        `Tensor` objects.
    """
    if control_inputs is None:
      return self._ControlDependenciesController(self, None)
    # First convert the inputs to ops, and deduplicate them.
    # NOTE(mrry): Other than deduplication, we do not currently track direct
    #   or indirect dependencies between control_inputs, which may result in
    #   redundant control inputs.
    control_ops = []
    current = self._current_control_dependencies()
    for c in control_inputs:
      c = self.as_graph_element(c)
      if isinstance(c, Tensor):
        c = c.op
      elif not isinstance(c, Operation):
        raise TypeError("Control input must be Operation or Tensor: %s" % c)
      if c not in current:
        control_ops.append(c)
        current.add(c)
    return self._ControlDependenciesController(self, control_ops)

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _attr_scope(self, attr_map):
    """EXPERIMENTAL: A context manager for setting attributes on operators.

    This context manager can be used to add additional
    attributes to operators within the scope of the context.

    For example:

       with ops.Graph().as_default() as g:
         f_1 = Foo()  # No extra attributes
         with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=False)}):
           f_2 = Foo()  # Additional attribute _a=False
           with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=True)}):
             f_3 = Foo()  # Additional attribute _a=False
             with g._attr_scope({"_a": None}):
               f_4 = Foo()  # No additional attributes.

    Args:
      attr_map: A dictionary mapping attr name strings to
        AttrValue protocol buffers or None.

    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.

    Raises:
      TypeError: If attr_map is not a dictionary mapping
        strings to AttrValue protobufs.
    """
    if not isinstance(attr_map, dict):
      raise TypeError("attr_map must be a dictionary mapping "
                      "strings to AttrValue protocol buffers")
    # The saved_attrs dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_attrs = {}
    # Install the given attribute
    for name, attr in attr_map.items():
      if not (isinstance(name, six.string_types) and
              (isinstance(attr, (type(None), attr_value_pb2.AttrValue)) or
               callable(attr))):
        raise TypeError("attr_map must be a dictionary mapping "
                        "strings to AttrValue protocol buffers or "
                        "callables that emit AttrValue protocol buffers")
      try:
        saved_attrs[name] = self._attr_scope_map[name]
      except KeyError:
        pass
      if attr is None:
        del self._attr_scope_map[name]
      else:
        self._attr_scope_map[name] = attr
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the attributes set for this context, and restore any saved
      # attributes.
      for name, attr in attr_map.items():
        try:
          self._attr_scope_map[name] = saved_attrs[name]
        except KeyError:
          del self._attr_scope_map[name]
  # pylint: enable=g-doc-return-or-yield

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _kernel_label_map(self, op_to_kernel_label_map):
    """EXPERIMENTAL: A context manager for setting kernel labels.

    This context manager can be used to select particular
    implementations of kernels within the scope of the context.

    For example:

        with ops.Graph().as_default() as g:
          f_1 = Foo()  # Uses the default registered kernel for the Foo op.
          with g.kernel_label_map({"Foo": "v_2"}):
            f_2 = Foo()  # Uses the registered kernel with label "v_2"
                         # for the Foo op.
            with g.kernel_label_map({"Foo": "v_3"}):
              f_3 = Foo()  # Uses the registered kernel with label "v_3"
                           # for the Foo op.
              with g.kernel_label_map({"Foo": ""}):
                f_4 = Foo()  # Uses the default registered kernel
                             # for the Foo op.

    Args:
      op_to_kernel_label_map: A dictionary mapping op type strings to
        kernel label strings.

    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.

    Raises:
      TypeError: If op_to_kernel_label_map is not a dictionary mapping
        strings to strings.
    """
    if not isinstance(op_to_kernel_label_map, dict):
      raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_labels dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_labels = {}
    # Install the given label
    for op_type, label in op_to_kernel_label_map.items():
      if not (isinstance(op_type, six.string_types)
              and isinstance(label, six.string_types)):
        raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_labels[op_type] = self._op_to_kernel_label_map[op_type]
      except KeyError:
        pass
      self._op_to_kernel_label_map[op_type] = label
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, label in op_to_kernel_label_map.items():
        try:
          self._op_to_kernel_label_map[op_type] = saved_labels[op_type]
        except KeyError:
          del self._op_to_kernel_label_map[op_type]
  # pylint: enable=g-doc-return-or-yield

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def gradient_override_map(self, op_type_map):
    """EXPERIMENTAL: A context manager for overriding gradient functions.

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

    Args:
      op_type_map: A dictionary mapping op type strings to alternative op
        type strings.

    Returns:
      A context manager that sets the alternative op type to be used for one
      or more ops created in that context.

    Raises:
      TypeError: If `op_type_map` is not a dictionary mapping strings to
        strings.
    """
    if not isinstance(op_type_map, dict):
      raise TypeError("op_type_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_mappings dictionary stores any currently-set mappings that
    # will be overridden by this context manager.
    saved_mappings = {}
    # Install the given label
    for op_type, mapped_op_type in op_type_map.items():
      if not (isinstance(op_type, six.string_types)
              and isinstance(mapped_op_type, six.string_types)):
        raise TypeError("op_type_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_mappings[op_type] = self._gradient_override_map[op_type]
      except KeyError:
        pass
      self._gradient_override_map[op_type] = mapped_op_type
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, mapped_op_type in op_type_map.items():
        try:
          self._gradient_override_map[op_type] = saved_mappings[op_type]
        except KeyError:
          del self._gradient_override_map[op_type]
  # pylint: enable=g-doc-return-or-yield

  def prevent_feeding(self, tensor):
    """Marks the given `tensor` as unfeedable in this graph."""
    self._unfeedable_tensors.add(tensor)

  def is_feedable(self, tensor):
    """Returns `True` if and only if `tensor` is feedable."""
    return tensor not in self._unfeedable_tensors

  def prevent_fetching(self, op):
    """Marks the given `op` as unfetchable in this graph."""
    self._unfetchable_ops.add(op)

  def is_fetchable(self, tensor_or_op):
    """Returns `True` if and only if `tensor_or_op` is fetchable."""
    if isinstance(tensor_or_op, Tensor):
      return tensor_or_op.op not in self._unfetchable_ops
    else:
      return tensor_or_op not in self._unfetchable_ops


def device(device_name_or_function):
  """Wrapper for `Graph.device()` using the default graph.

  See
  @{tf.Graph.device}
  for more details.

  Args:
    device_name_or_function: The device name or function to use in
      the context.

  Returns:
    A context manager that specifies the default device to use for newly
    created ops.
  """
  return get_default_graph().device(device_name_or_function)


def container(container_name):
  """Wrapper for `Graph.container()` using the default graph.

  Args:
    container_name: The container string to use in the context.

  Returns:
    A context manager that specifies the default container to use for newly
    created stateful ops.
  """
  return get_default_graph().container(container_name)


def colocate_with(op, ignore_existing=False):
  return get_default_graph().colocate_with(op, ignore_existing)


def control_dependencies(control_inputs):
  """Wrapper for `Graph.control_dependencies()` using the default graph.

  See @{tf.Graph.control_dependencies}
  for more details.

  Args:
    control_inputs: A list of `Operation` or `Tensor` objects which
      must be executed or computed before running the operations
      defined in the context.  Can also be `None` to clear the control
      dependencies.

  Returns:
   A context manager that specifies control dependencies for all
   operations constructed within the context.
  """
  return get_default_graph().control_dependencies(control_inputs)


class _DefaultStack(threading.local):
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    super(_DefaultStack, self).__init__()
    self._enforce_nesting = True
    self.stack = []

  def get_default(self):
    return self.stack[-1] if len(self.stack) >= 1 else None

  def reset(self):
    self.stack = []

  def is_cleared(self):
    return self.stack == []

  @property
  def enforce_nesting(self):
    return self._enforce_nesting

  @enforce_nesting.setter
  def enforce_nesting(self, value):
    self._enforce_nesting = value

  @tf_contextlib.contextmanager
  def get_controller(self, default):
    """A context manager for manipulating a default stack."""
    try:
      self.stack.append(default)
      yield default
    finally:
      if self._enforce_nesting:
        if self.stack[-1] is not default:
          raise AssertionError(
              "Nesting violated for default stack of %s objects"
              % type(default))
        self.stack.pop()
      else:
        self.stack.remove(default)

_default_session_stack = _DefaultStack()


def default_session(session):
  """Python "with" handler for defining a default session.

  This function provides a means of registering a session for handling
  Tensor.eval() and Operation.run() calls. It is primarily intended for use
  by session.Session, but can be used with any object that implements
  the Session.run() interface.

  Use with the "with" keyword to specify that Tensor.eval() and Operation.run()
  invocations within the scope of a block should be executed by a particular
  session.

  The default session applies to the current thread only, so it is always
  possible to inspect the call stack and determine the scope of a default
  session. If you create a new thread, and wish to use the default session
  in that thread, you must explicitly add a "with ops.default_session(sess):"
  block in that thread's function.

  Example:
    The following code examples are equivalent:

    # 1. Using the Session object directly:
    sess = ...
    c = tf.constant(5.0)
    sess.run(c)

    # 2. Using default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      result = c.eval()

    # 3. Overriding default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      with ops.default_session(...):
        c.eval(session=sess)

  Args:
    session: The session to be installed as the default session.

  Returns:
    A context manager for the default session.
  """
  return _default_session_stack.get_controller(session)


def get_default_session():
  """Returns the default session for the current thread.

  The returned `Session` will be the innermost session on which a
  `Session` or `Session.as_default()` context has been entered.

  NOTE: The default session is a property of the current thread. If you
  create a new thread, and wish to use the default session in that
  thread, you must explicitly add a `with sess.as_default():` in that
  thread's function.

  Returns:
    The default `Session` being used in the current thread.
  """
  return _default_session_stack.get_default()


def _eval_using_default_session(tensors, feed_dict, graph, session=None):
  """Uses the default session to evaluate one or more tensors.

  Args:
    tensors: A single Tensor, or a list of Tensor objects.
    feed_dict: A dictionary that maps Tensor objects (or tensor names) to lists,
      numpy ndarrays, TensorProtos, or strings.
    graph: The graph in which the tensors are defined.
    session: (Optional) A different session to use to evaluate "tensors".

  Returns:
    Either a single numpy ndarray if "tensors" is a single tensor; or a list
    of numpy ndarrays that each correspond to the respective element in
    "tensors".

  Raises:
    ValueError: If no default session is available; the default session
      does not have "graph" as its graph; or if "session" is specified,
      and it does not have "graph" as its graph.
  """
  if session is None:
    session = get_default_session()
    if session is None:
      raise ValueError("Cannot evaluate tensor using `eval()`: No default "
                       "session is registered. Use `with "
                       "sess.as_default()` or pass an explicit session to "
                       "`eval(session=sess)`")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph. Pass an explicit session to "
                       "`eval(session=sess)`.")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph.")
  return session.run(tensors, feed_dict)


def _run_using_default_session(operation, feed_dict, graph, session=None):
  """Uses the default session to run "operation".

  Args:
    operation: The Operation to be run.
    feed_dict: A dictionary that maps Tensor objects (or tensor names) to lists,
      numpy ndarrays, TensorProtos, or strings.
    graph: The graph in which "operation" is defined.
    session: (Optional) A different session to use to run "operation".

  Raises:
    ValueError: If no default session is available; the default session
      does not have "graph" as its graph; or if "session" is specified,
      and it does not have "graph" as its graph.
  """
  if session is None:
    session = get_default_session()
    if session is None:
      raise ValueError("Cannot execute operation using `run()`: No default "
                       "session is registered. Use `with "
                       "sess.as_default():` or pass an explicit session to "
                       "`run(session=sess)`")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to execute operation: "
                       "the operation's graph is different from the "
                       "session's graph. Pass an explicit session to "
                       "run(session=sess).")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to execute operation: "
                       "the operation's graph is different from the session's "
                       "graph.")
  session.run(operation, feed_dict)


class _DefaultGraphStack(_DefaultStack):
  """A thread-local stack of objects for providing an implicit default graph."""

  def __init__(self):
    super(_DefaultGraphStack, self).__init__()
    self._global_default_graph = None

  def get_default(self):
    """Override that returns a global default if the stack is empty."""
    ret = super(_DefaultGraphStack, self).get_default()
    if ret is None:
      ret = self._GetGlobalDefaultGraph()
    return ret

  def _GetGlobalDefaultGraph(self):
    if self._global_default_graph is None:
      # TODO(mrry): Perhaps log that the default graph is being used, or set
      #   provide some other feedback to prevent confusion when a mixture of
      #   the global default graph and an explicit graph are combined in the
      #   same process.
      self._global_default_graph = Graph()
    return self._global_default_graph

  def reset(self):
    super(_DefaultGraphStack, self).reset()
    self._global_default_graph = None

_default_graph_stack = _DefaultGraphStack()


def reset_default_graph():
  """Clears the default graph stack and resets the global default graph.

  NOTE: The default graph is a property of the current thread. This
  function applies only to the current thread.  Calling this function while
  a `tf.Session` or `tf.InteractiveSession` is active will result in undefined
  behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
  after calling this function will result in undefined behavior.
  Raises:
    AssertionError: If this function is called within a nested graph.
  """
  if not _default_graph_stack.is_cleared():
    raise AssertionError("Do not use tf.reset_default_graph() to clear "
                         "nested graphs. If you need a cleared graph, "
                         "exit the nesting and create a new graph.")
  _default_graph_stack.reset()


def get_default_graph():
  """Returns the default graph for the current thread.

  The returned graph will be the innermost graph on which a
  `Graph.as_default()` context has been entered, or a global default
  graph if none has been explicitly created.

  NOTE: The default graph is a property of the current thread. If you
  create a new thread, and wish to use the default graph in that
  thread, you must explicitly add a `with g.as_default():` in that
  thread's function.

  Returns:
    The default `Graph` being used in the current thread.
  """
  return _default_graph_stack.get_default()


def _assert_same_graph(original_item, item):
  """Fail if the 2 items are from different graphs.

  Args:
    original_item: Original item to check against.
    item: Item to check.

  Raises:
    ValueError: if graphs do not match.
  """
  if original_item.graph is not item.graph:
    raise ValueError(
        "%s must be from the same graph as %s." % (item, original_item))


def _get_graph_from_inputs(op_input_list, graph=None):
  """Returns the appropriate graph to use for the given inputs.

  This library method provides a consistent algorithm for choosing the graph
  in which an Operation should be constructed:

  1. If the default graph is being used to construct a function, we
     use the default graph.
  2. If the "graph" is specified explicitly, we validate that all of the inputs
     in "op_input_list" are compatible with that graph.
  3. Otherwise, we attempt to select a graph from the first Operation-
     or Tensor-valued input in "op_input_list", and validate that all other
     such inputs are in the same graph.
  4. If the graph was not specified and it could not be inferred from
     "op_input_list", we attempt to use the default graph.

  Args:
    op_input_list: A list of inputs to an operation, which may include `Tensor`,
      `Operation`, and other objects that may be converted to a graph element.
    graph: (Optional) The explicit graph to use.

  Raises:
    TypeError: If op_input_list is not a list or tuple, or if graph is not a
      Graph.
    ValueError: If a graph is explicitly passed and not all inputs are from it,
      or if the inputs are from multiple graphs, or we could not find a graph
      and there was no default graph.

  Returns:
    The appropriate graph to use for the given inputs.

  """
  if get_default_graph().building_function:
    return get_default_graph()

  op_input_list = tuple(op_input_list)  # Handle generators correctly
  if graph and not isinstance(graph, Graph):
    raise TypeError("Input graph needs to be a Graph: %s" % graph)

  # 1. We validate that all of the inputs are from the same graph. This is
  #    either the supplied graph parameter, or the first one selected from one
  #    the graph-element-valued inputs. In the latter case, we hold onto
  #    that input in original_graph_element so we can provide a more
  #    informative error if a mismatch is found.
  original_graph_element = None
  for op_input in op_input_list:
    # Determine if this is a valid graph_element.
    graph_element = None
    if isinstance(op_input, (Operation, _TensorLike)):
      graph_element = op_input
    else:
      graph_element = _as_graph_element(op_input)

    if graph_element is not None:
      if not graph:
        original_graph_element = graph_element
        graph = graph_element.graph
      elif original_graph_element is not None:
        _assert_same_graph(original_graph_element, graph_element)
      elif graph_element.graph is not graph:
        raise ValueError(
            "%s is not from the passed-in graph." % graph_element)

  # 2. If all else fails, we use the default graph, which is always there.
  return graph or get_default_graph()


class GraphKeys(object):
  """Standard names to use for graph collections.

  The standard library uses various well-known names to collect and
  retrieve values associated with a graph. For example, the
  `tf.Optimizer` subclasses default to optimizing the variables
  collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
  specified, but it is also possible to pass an explicit list of
  variables.

  The following standard keys are defined:

  * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
    across distributed environment (model variables are subset of these). See
    @{tf.global_variables}
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
    @{tf.trainable_variables}
    for more details.
  * `SUMMARIES`: the summary `Tensor` objects that have been created in the
    graph. See
    @{tf.summary.merge_all}
    for more details.
  * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
    produce input for a computation. See
    @{tf.train.start_queue_runners}
    for more details.
  * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
    keep moving averages.  See
    @{tf.moving_average_variables}
    for more details.
  * `REGULARIZATION_LOSSES`: regularization losses collected during graph
    construction.

  The following standard keys are _defined_, but their collections are **not**
  automatically populated as many of the others are:

  * `WEIGHTS`
  * `BIASES`
  * `ACTIVATIONS`
  """

  # Key to collect Variable objects that are global (shared across machines).
  # Default collection for all variables, except local ones.
  GLOBAL_VARIABLES = "variables"
  # Key to collect local variables that are local to the machine and are not
  # saved/restored.
  LOCAL_VARIABLES = "local_variables"
  # Key to collect model variables defined by layers.
  MODEL_VARIABLES = "model_variables"
  # Key to collect Variable objects that will be trained by the
  # optimizers.
  TRAINABLE_VARIABLES = "trainable_variables"
  # Key to collect summaries.
  SUMMARIES = "summaries"
  # Key to collect QueueRunners.
  QUEUE_RUNNERS = "queue_runners"
  # Key to collect table initializers.
  TABLE_INITIALIZERS = "table_initializer"
  # Key to collect asset filepaths. An asset represents an external resource
  # like a vocabulary file.
  ASSET_FILEPATHS = "asset_filepaths"
  # Key to collect Variable objects that keep moving averages.
  MOVING_AVERAGE_VARIABLES = "moving_average_variables"
  # Key to collect regularization losses at graph construction.
  REGULARIZATION_LOSSES = "regularization_losses"
  # Key to collect concatenated sharded variables.
  CONCATENATED_VARIABLES = "concatenated_variables"
  # Key to collect savers.
  SAVERS = "savers"
  # Key to collect weights
  WEIGHTS = "weights"
  # Key to collect biases
  BIASES = "biases"
  # Key to collect activations
  ACTIVATIONS = "activations"
  # Key to collect update_ops
  UPDATE_OPS = "update_ops"
  # Key to collect losses
  LOSSES = "losses"
  # Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
  SAVEABLE_OBJECTS = "saveable_objects"
  # Key to collect all shared resources used by the graph which need to be
  # initialized once per cluster.
  RESOURCES = "resources"
  # Key to collect all shared resources used in this graph which need to be
  # initialized once per session.
  LOCAL_RESOURCES = "local_resources"
  # Trainable resource-style variables.
  TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"

  # Key to indicate various ops.
  INIT_OP = "init_op"
  LOCAL_INIT_OP = "local_init_op"
  READY_OP = "ready_op"
  READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
  SUMMARY_OP = "summary_op"
  GLOBAL_STEP = "global_step"

  # Used to count the number of evaluations performed during a single evaluation
  # run.
  EVAL_STEP = "eval_step"
  TRAIN_OP = "train_op"

  # Key for control flow context.
  COND_CONTEXT = "cond_context"
  WHILE_CONTEXT = "while_context"

  # Key for streaming model ports.
  # NOTE(yuanbyu): internal and experimental.
  _STREAMING_MODEL_PORTS = "streaming_model_ports"

  @decorator_utils.classproperty
  def VARIABLES(cls):  # pylint: disable=no-self-argument
    logging.warning("VARIABLES collection name is deprecated, "
                    "please use GLOBAL_VARIABLES instead; "
                    "VARIABLES will be removed after 2017-03-02.")
    return cls.GLOBAL_VARIABLES


def add_to_collection(name, value):
  """Wrapper for `Graph.add_to_collection()` using the default graph.

  See @{tf.Graph.add_to_collection}
  for more details.

  Args:
    name: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collection.
  """
  get_default_graph().add_to_collection(name, value)


def add_to_collections(names, value):
  """Wrapper for `Graph.add_to_collections()` using the default graph.

  See @{tf.Graph.add_to_collections}
  for more details.

  Args:
    names: The key for the collections. The `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collections.
  """
  get_default_graph().add_to_collections(names, value)


def get_collection_ref(key):
  """Wrapper for `Graph.get_collection_ref()` using the default graph.

  See @{tf.Graph.get_collection_ref}
  for more details.

  Args:
    key: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.

  Returns:
    The list of values in the collection with the given `name`, or an empty
    list if no value has been added to that collection.  Note that this returns
    the collection list itself, which can be modified in place to change the
    collection.
  """
  return get_default_graph().get_collection_ref(key)


def get_collection(key, scope=None):
  """Wrapper for `Graph.get_collection()` using the default graph.

  See @{tf.Graph.get_collection}
  for more details.

  Args:
    key: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.
    scope: (Optional.) If supplied, the resulting list is filtered to include
      only items whose `name` attribute matches using `re.match`. Items
      without a `name` attribute are never returned if a scope is supplied and
      the choice or `re.match` means that a `scope` without special tokens
      filters by prefix.

  Returns:
    The list of values in the collection with the given `name`, or
    an empty list if no value has been added to that collection. The
    list contains the values in the order under which they were
    collected.
  """
  return get_default_graph().get_collection(key, scope)


def get_all_collection_keys():
  """Returns a list of collections used in the default graph."""
  return get_default_graph().get_all_collection_keys()


# pylint: disable=g-doc-return-or-yield
@tf_contextlib.contextmanager
def name_scope(name, default_name=None, values=None):
  """Returns a context manager for use when defining a Python op.

  This context manager validates that the given `values` are from the
  same graph, makes that graph the default graph, and pushes a
  name scope in that graph (see
  @{tf.Graph.name_scope}
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

  Args:
    name: The name argument that is passed to the op function.
    default_name: The default name to use if the `name` argument is `None`.
    values: The list of `Tensor` arguments that are passed to the op function.

  Returns:
    A context manager for use in defining Python ops. Yields the name scope.

  Raises:
    ValueError: if neither `name` nor `default_name` is provided
      but `values` are.
  """
  n = default_name if name is None else name
  if n is None and values is not None:
    # We only raise an error if values is not None (provided) because currently
    # tf.name_scope(None) (values=None then) is sometimes used as an idiom
    # to reset to top scope.
    raise ValueError(
        "At least one of name (%s) and default_name (%s) must be provided." % (
            name, default_name))
  if values is None:
    values = []
  g = _get_graph_from_inputs(values)
  with g.as_default(), g.name_scope(n) as scope:
    yield scope
# pylint: enable=g-doc-return-or-yield


def strip_name_scope(name, export_scope):
  """Removes name scope from a name.

  Args:
    name: A `string` name.
    export_scope: Optional `string`. Name scope to remove.

  Returns:
    Name with name scope removed, or the original name if export_scope
    is None.
  """
  if export_scope:
    try:
      # Strips export_scope/, export_scope///,
      # ^export_scope/, loc:@export_scope/.
      str_to_replace = r"([\^]|loc:@|^)" + export_scope + r"[\/]+(.*)"
      return re.sub(str_to_replace, r"\1\2", compat.as_str(name), count=1)
    except TypeError as e:
      # If the name is not of a type we can process, simply return it.
      logging.warning(e)
      return name
  else:
    return name


def prepend_name_scope(name, import_scope):
  """Prepends name scope to a name.

  Args:
    name: A `string` name.
    import_scope: Optional `string`. Name scope to add.

  Returns:
    Name with name scope added, or the original name if import_scope
    is None.
  """
  if import_scope:
    try:
      str_to_replace = r"([\^]|loc:@|^)(.*)"
      return re.sub(str_to_replace, r"\1" + import_scope + r"/\2",
                    compat.as_str(name))
    except TypeError as e:
      # If the name is not of a type we can process, simply return it.
      logging.warning(e)
      return name
  else:
    return name


# pylint: disable=g-doc-return-or-yield
@tf_contextlib.contextmanager
def op_scope(values, name, default_name=None):
  """DEPRECATED. Same as name_scope above, just different argument order."""
  logging.warn("tf.op_scope(values, name, default_name) is deprecated,"
               " use tf.name_scope(name, default_name, values)")
  with name_scope(name, default_name=default_name, values=values) as scope:
    yield scope


_proto_function_registry = registry.Registry("proto functions")


def register_proto_function(collection_name, proto_type=None, to_proto=None,
                            from_proto=None):
  """Registers `to_proto` and `from_proto` functions for collection_name.

  `to_proto` function converts a Python object to the corresponding protocol
  buffer, and returns the protocol buffer.

  `from_proto` function converts protocol buffer into a Python object, and
  returns the object..

  Args:
    collection_name: Name of the collection.
    proto_type: Protobuf type, such as `saver_pb2.SaverDef`,
      `variable_pb2.VariableDef`, `queue_runner_pb2.QueueRunnerDef`..
    to_proto: Function that implements Python object to protobuf conversion.
    from_proto: Function that implements protobuf to Python object conversion.
  """
  if to_proto and not callable(to_proto):
    raise TypeError("to_proto must be callable.")
  if from_proto and not callable(from_proto):
    raise TypeError("from_proto must be callable.")

  _proto_function_registry.register((proto_type, to_proto, from_proto),
                                    collection_name)


def get_collection_proto_type(collection_name):
  """Returns the proto_type for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[0]
  except LookupError:
    return None


def get_to_proto_function(collection_name):
  """Returns the to_proto function for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[1]
  except LookupError:
    return None


def get_from_proto_function(collection_name):
  """Returns the from_proto function for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[2]
  except LookupError:
    return None


def _operation_conversion_error(op, dtype=None, name=None, as_ref=False):
  """Produce a nice error if someone converts an Operation to a Tensor."""
  raise TypeError(
      ("Can't convert Operation '%s' to Tensor "
       "(target dtype=%r, name=%r, as_ref=%r)") %
      (op.name, dtype, name, as_ref))


register_tensor_conversion_function(Operation, _operation_conversion_error)
