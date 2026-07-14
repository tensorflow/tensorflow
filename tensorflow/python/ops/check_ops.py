# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-short-docstring-punctuation
"""Asserts and Boolean Checks."""

import collections

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

NUMERIC_TYPES = frozenset([
    dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int8, dtypes.int16,
    dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32,
    dtypes.uint64, dtypes.qint8, dtypes.qint16, dtypes.qint32, dtypes.quint8,
    dtypes.quint16, dtypes.complex64, dtypes.complex128, dtypes.bfloat16
])

__all__ = [
    'assert_negative',
    'assert_positive',
    'assert_proper_iterable',
    'assert_non_negative',
    'assert_non_positive',
    'assert_equal',
    'assert_none_equal',
    'assert_near',
    'assert_integer',
    'assert_less',
    'assert_less_equal',
    'assert_greater',
    'assert_greater_equal',
    'assert_rank',
    'assert_rank_at_least',
    'assert_rank_in',
    'assert_same_float_dtype',
    'assert_scalar',
    'assert_type',
    'assert_shapes',
    'is_non_decreasing',
    'is_numeric_tensor',
    'is_strictly_increasing',
]


def _maybe_constant_value_string(t):
  if not isinstance(t, tensor_lib.Tensor):
    return str(t)
  const_t = tensor_util.constant_value(t)
  if const_t is not None:
    return str(const_t)
  return t


def _assert_static(condition, data):
  """Raises a InvalidArgumentError with as much information as possible."""
  if not condition:
    data_static = [_maybe_constant_value_string(x) for x in data]
    raise errors.InvalidArgumentError(node_def=None, op=None,
                                      message='\n'.join(data_static))


def _shape_and_dtype_str(tensor):
  """Returns a string containing tensor's shape and dtype."""
  return 'shape=%s dtype=%s' % (tensor.shape, tensor.dtype.name)


def _unary_assert_doc(sym, sym_name):
  """Common docstring for assert_* ops that evaluate a unary predicate over every element of a tensor.

  Args:
    sym: Mathematical symbol for the check performed on each element, i.e. "> 0"
    sym_name: English-language name for the op described by sym

  Returns:
    Decorator that adds the appropriate docstring to the function for symbol
    `sym`.
  """

  def _decorator(func):
    """Generated decorator that adds the appropriate docstring to the function for symbol `sym`.

    Args:
      func: Function for a TensorFlow op

    Returns:
      Version of `func` with documentation attached.
    """
    opname = func.__name__
    cap_sym_name = sym_name.capitalize()

    func.__doc__ = """
    Assert the condition `x {sym}` holds element-wise.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.debugging.{opname}(x, y)]):
      output = tf.reduce_sum(x)
    ```

    {sym_name} means, for every element `x[i]` of `x`, we have `x[i] {sym}`.
    If `x` is empty this is trivially satisfied.

    Args:
      x:  Numeric `Tensor`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to "{opname}".

    Returns:
      Op that raises `InvalidArgumentError` if `x {sym}` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x {sym}` is False. The check can be performed immediately during
        eager execution or if `x` is statically known.
    """.format(
        sym=sym, sym_name=cap_sym_name, opname=opname)
    return func

  return _decorator


def _binary_assert_doc(sym, test_var):
  """Common docstring for most of the v1 assert_* ops that compare two tensors element-wise.

  Args:
    sym: Binary operation symbol, i.e. "=="
    test_var: a string that represents the variable in the right-hand side of
      binary operator of the test case

  Returns:
    Decorator that adds the appropriate docstring to the function for
  symbol `sym`.
  """

  def _decorator(func):
    """Generated decorator that adds the appropriate docstring to the function for symbol `sym`.

    Args:
      func: Function for a TensorFlow op

    Returns:
      A version of `func` with documentation attached.
    """
    opname = func.__name__

    func.__doc__ = """
    Assert the condition `x {sym} y` holds element-wise.

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] {sym} y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.compat.v1.{opname}(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Args:
      x:  Numeric `Tensor`.
      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`, `y`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to "{opname}".

    Returns:
      Op that raises `InvalidArgumentError` if `x {sym} y` is False.

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x {sym} y` is False. The check can be performed immediately during
        eager execution or if `x` and `y` are statically known.

    @compatibility(TF2)
    `tf.compat.v1.{opname}` is compatible with eager execution and
    `tf.function`.
    Please use `tf.debugging.{opname}` instead when migrating to TF2. Apart
    from `data`, all arguments are supported with the same argument name.

    If you want to ensure the assert statements run before the
    potentially-invalid computation, please use `tf.control_dependencies`,
    as tf.function auto-control dependencies are insufficient for assert
    statements.

    #### Structural Mapping to Native TF2

    Before:

    ```python
    tf.compat.v1.{opname}(
      x=x, y=y, data=data, summarize=summarize,
      message=message, name=name)
    ```

    After:

    ```python
    tf.debugging.{opname}(
      x=x, y=y, message=message,
      summarize=summarize, name=name)
    ```

    #### TF1 & TF2 Usage Example

    TF1:

    >>> g = tf.Graph()
    >>> with g.as_default():
    ...   a = tf.compat.v1.placeholder(tf.float32, [2])
    ...   b = tf.compat.v1.placeholder(tf.float32, [2])
    ...   result = tf.compat.v1.{opname}(a, b,
    ...     message='"a {sym} b" does not hold for the given inputs')
    ...   with tf.compat.v1.control_dependencies([result]):
    ...     sum_node = a + b
    >>> sess = tf.compat.v1.Session(graph=g)
    >>> val = sess.run(sum_node, feed_dict={{a: [1, 2], b:{test_var}}})


    TF2:

    >>> a = tf.Variable([1, 2], dtype=tf.float32)
    >>> b = tf.Variable({test_var}, dtype=tf.float32)
    >>> assert_op = tf.debugging.{opname}(a, b, message=
    ...   '"a {sym} b" does not hold for the given inputs')
    >>> # When working with tf.control_dependencies
    >>> with tf.control_dependencies([assert_op]):
    ...   val = a + b

    @end_compatibility
    """.format(
        sym=sym, opname=opname, test_var=test_var)
    return func

  return _decorator


def _binary_assert_doc_v2(sym, opname, test_var):
  """Common docstring for v2 assert_* ops that compare two tensors element-wise.

  Args:
    sym: Binary operation symbol, i.e. "=="
    opname: Name for the symbol, i.e. "assert_equal"
    test_var: A number used in the docstring example

  Returns:
    Decorator that adds the appropriate docstring to the function for
  symbol `sym`.
  """

  def _decorator(func):
    """Decorator that adds docstring to the function for symbol `sym`.

    Args:
      func: Function for a TensorFlow op

    Returns:
      A version of `func` with documentation attached.
    """

    func.__doc__ = """
    Assert the condition `x {sym} y` holds element-wise.

    This Op checks that `x[i] {sym} y[i]` holds for every pair of (possibly
    broadcast) elements of `x` and `y`. If both `x` and `y` are empty, this is
    trivially satisfied.

    If `x` {sym} `y` does not hold, `message`, as well as the first `summarize`
    entries of `x` and `y` are printed, and `InvalidArgumentError` is raised.

    When using inside `tf.function`, this API takes effects during execution.
    It's recommended to use this API with `tf.control_dependencies` to
    ensure the correct execution order.

    In the following example, without `tf.control_dependencies`, errors may
    not be raised at all.
    Check `tf.control_dependencies` for more details.

    >>> def check_size(x):
    ...   with tf.control_dependencies([
    ...       tf.debugging.{opname}(tf.size(x), {test_var},
    ...                       message='Bad tensor size')]):
    ...     return x

    >>> check_size(tf.ones([2, 3], tf.float32))
    Traceback (most recent call last):
       ...
    InvalidArgumentError: ...

    Args:
      x:  Numeric `Tensor`.
      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
      message: A string to prefix to the default message. (optional)
      summarize: Print this many entries of each tensor. (optional)
      name: A name for this operation (optional).  Defaults to "{opname}".

    Returns:
      Op that raises `InvalidArgumentError` if `x {sym} y` is False. This can
        be used with `tf.control_dependencies` inside of `tf.function`s to
        block followup computation until the check has executed.
      @compatibility(eager)
      returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x == y` is False. The check can be performed immediately during eager
        execution or if `x` and `y` are statically known.
    """.format(
        sym=sym, opname=opname, test_var=test_var)
    return func

  return _decorator


def _make_assert_msg_data(sym, x, y, summarize, test_op):
  """Subroutine of _binary_assert that generates the components of the default error message when running in eager mode.

  Args:
    sym: Mathematical symbol for the test to apply to pairs of tensor elements,
      i.e. "=="
    x: First input to the assertion after applying `convert_to_tensor()`
    y: Second input to the assertion
    summarize: Value of the "summarize" parameter to the original assert_* call;
      tells how many elements of each tensor to print.
    test_op: TensorFlow op that returns a Boolean tensor with True in each
      position where the assertion is satisfied.

  Returns:
    List of tensors and scalars that, when stringified and concatenated,
    will produce the error message string.
  """
  # Prepare a message with first elements of x and y.
  data = []

  data.append('Condition x %s y did not hold.' % sym)

  if summarize > 0:
    if x.shape == y.shape and x.shape.as_list():
      # If the shapes of x and y are the same (and not scalars),
      # Get the values that actually differed and their indices.
      # If shapes are different this information is more confusing
      # than useful.
      mask = math_ops.logical_not(test_op)
      indices = array_ops.where(mask)
      indices_np = indices.numpy()
      x_vals = array_ops.boolean_mask(x, mask)
      y_vals = array_ops.boolean_mask(y, mask)
      num_vals = min(summarize, indices_np.shape[0])
      data.append('Indices of first %d different values:' % num_vals)
      data.append(indices_np[:num_vals])
      data.append('Corresponding x values:')
      data.append(x_vals.numpy().reshape((-1,))[:num_vals])
      data.append('Corresponding y values:')
      data.append(y_vals.numpy().reshape((-1,))[:num_vals])

    # reshape((-1,)) is the fastest way to get a flat array view.
    x_np = x.numpy().reshape((-1,))
    y_np = y.numpy().reshape((-1,))
    x_sum = min(x_np.size, summarize)
    y_sum = min(y_np.size, summarize)
    data.append('First %d elements of x:' % x_sum)
    data.append(x_np[:x_sum])
    data.append('First %d elements of y:' % y_sum)
    data.append(y_np[:y_sum])

  return data


def _pretty_print(data_item, summarize):
  """Format a data item for use in an error message in eager mode.

  Args:
    data_item: One of the items in the "data" argument to an assert_* function.
      Can be a Tensor or a scalar value.
    summarize: How many elements to retain of each tensor-valued entry in data.

  Returns:
    An appropriate string representation of data_item
  """
  if isinstance(data_item, tensor_lib.Tensor):
    arr = data_item.numpy()
    if np.isscalar(arr):
      # Tensor.numpy() returns a scalar for zero-dimensional tensors
      return str(arr)
    else:
      flat = arr.reshape((-1,))
      lst = [str(x) for x in flat[:summarize]]
      if len(lst) < flat.size:
        lst.append('...')
      return str(lst)
  else:
    return str(data_item)


def _binary_assert(sym, opname, op_func, static_func, x, y, data, summarize,
                   message, name):
  """Generic binary elementwise assertion.

  Implements the behavior described in _binary_assert_doc() above.
  Args:
    sym: Mathematical symbol for the test to apply to pairs of tensor elements,
      i.e. "=="
    opname: Name of the assert op in the public API, i.e. "assert_equal"
    op_func: Function that, if passed the two Tensor inputs to the assertion (x
      and y), will return the test to be passed to reduce_all() i.e.
    static_func: Function that, if passed numpy ndarray versions of the two
      inputs to the assertion, will return a Boolean ndarray with containing
      True in all positions where the assertion PASSES.
      i.e. np.equal for assert_equal()
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to the value of
      `opname`.

  Returns:
    See docstring template in _binary_assert_doc().
  """
  with ops.name_scope(name, opname, [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')

    if context.executing_eagerly():
      test_op = op_func(x, y)
      condition = math_ops.reduce_all(test_op)
      if condition:
        return

      # If we get here, the assertion has failed.
      # Default to printing 3 elements like control_flow_ops.Assert (used
      # by graph mode) does. Also treat negative values as "print
      # everything" for consistency with Tensor::SummarizeValue().
      if summarize is None:
        summarize = 3
      elif summarize < 0:
        summarize = 1e9  # Code below will find exact size of x and y.

      if data is None:
        data = _make_assert_msg_data(sym, x, y, summarize, test_op)

      if message is not None:
        data = [message] + list(data)

      raise errors.InvalidArgumentError(
          node_def=None,
          op=None,
          message=('\n'.join(_pretty_print(d, summarize) for d in data)))

    else:  # not context.executing_eagerly()
      if data is None:
        data = [
            'Condition x %s y did not hold element-wise:' % sym,
            'x (%s) = ' % x.name, x,
            'y (%s) = ' % y.name, y
        ]
      if message is not None:
        data = [message] + list(data)
      condition = math_ops.reduce_all(op_func(x, y))
      x_static = tensor_util.constant_value(x)
      y_static = tensor_util.constant_value(y)
      if x_static is not None and y_static is not None:
        condition_static = np.all(static_func(x_static, y_static))
        _assert_static(condition_static, data)
      return control_flow_assert.Assert(condition, data, summarize=summarize)


@tf_export(
    'debugging.assert_proper_iterable',
    v1=['debugging.assert_proper_iterable', 'assert_proper_iterable'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_proper_iterable')
def assert_proper_iterable(values):
  """Static assert that values is a "proper" iterable.

  `Ops` that expect iterables of `Tensor` can call this to validate input.
  Useful since `Tensor`, `ndarray`, byte/text type are all iterables themselves.

  Args:
    values:  Object to be checked.

  Raises:
    TypeError:  If `values` is not iterable or is one of
      `Tensor`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.
  """
  unintentional_iterables = (
      (tensor_lib.Tensor, sparse_tensor.SparseTensor, np.ndarray)
      + compat.bytes_or_text_types
  )
  if isinstance(values, unintentional_iterables):
    raise TypeError(
        'Expected argument "values" to be a "proper" iterable.  Found: %s' %
        type(values))

  if not hasattr(values, '__iter__'):
    raise TypeError(
        'Expected argument "values" to be iterable.  Found: %s' % type(values))


@tf_export('debugging.assert_negative', v1=[])
@dispatch.add_dispatch_support
def assert_negative_v2(x, message=None, summarize=None, name=None):
  """Assert the condition `x < 0` holds element-wise.

  This Op checks that `x[i] < 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.

  If `x` is not negative everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.

  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all negative. This can be
      used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] < 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
  return assert_negative(x=x, message=message, summarize=summarize, name=name)


@tf_export(v1=['debugging.assert_negative', 'assert_negative'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_negative')
@_unary_assert_doc('< 0', 'negative')
def assert_negative(x, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_negative', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.executing_eagerly():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x < 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(x, zero, data=data, summarize=summarize)


@tf_export('debugging.assert_positive', v1=[])
@dispatch.add_dispatch_support
def assert_positive_v2(x, message=None, summarize=None, name=None):
  """Assert the condition `x > 0` holds element-wise.

  This Op checks that `x[i] > 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.

  If `x` is not positive everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.

  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional). Defaults to "assert_positive".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all positive. This can be
      used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] > 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
  return assert_positive(x=x, summarize=summarize, message=message, name=name)


@tf_export(v1=['debugging.assert_positive', 'assert_positive'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_positive')
@_unary_assert_doc('> 0', 'positive')
def assert_positive(x, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_positive', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.executing_eagerly():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message, 'Condition x > 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(zero, x, data=data, summarize=summarize)


@tf_export('debugging.assert_non_negative', v1=[])
@dispatch.add_dispatch_support
def assert_non_negative_v2(x, message=None, summarize=None, name=None):
  """Assert the condition `x >= 0` holds element-wise.

  This Op checks that `x[i] >= 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.

  If `x` is not >= 0 everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.

  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to
      "assert_non_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-negative. This can
      be used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] >= 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
  return assert_non_negative(x=x, summarize=summarize, message=message,
                             name=name)


@tf_export(v1=['debugging.assert_non_negative', 'assert_non_negative'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_non_negative')
@_unary_assert_doc('>= 0', 'non-negative')
def assert_non_negative(x, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_non_negative', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.executing_eagerly():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x >= 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(zero, x, data=data, summarize=summarize)


@tf_export('debugging.assert_non_positive', v1=[])
@dispatch.add_dispatch_support
def assert_non_positive_v2(x, message=None, summarize=None, name=None):
  """Assert the condition `x <= 0` holds element-wise.

  This Op checks that `x[i] <= 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.

  If `x` is not <= 0 everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.

  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to
      "assert_non_positive".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-positive. This can
      be used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] <= 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
  return assert_non_positive(x=x, summarize=summarize, message=message,
                             name=name)


@tf_export(v1=['debugging.assert_non_positive', 'assert_non_positive'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_non_positive')
@_unary_assert_doc('<= 0', 'non-positive')
def assert_non_positive(x, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_non_positive', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.executing_eagerly():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x <= 0 did not hold element-wise:'
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(x, zero, data=data, summarize=summarize)


@tf_export('debugging.assert_equal', 'assert_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('==', 'assert_equal', 3)
def assert_equal_v2(x, y, message=None, summarize=None, name=None):
  return assert_equal(x=x, y=y, summarize=summarize, message=message, name=name)


@tf_export(v1=['debugging.assert_equal', 'assert_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('==', '[1, 2]')
def assert_equal(x, y, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  with ops.name_scope(name, 'assert_equal', [x, y, data]):
    # Short-circuit if x and y are the same tensor.
    if x is y:
      return None if context.executing_eagerly() else control_flow_ops.no_op()
  return _binary_assert('==', 'assert_equal', math_ops.equal, np.equal, x, y,
                        data, summarize, message, name)


@tf_export('debugging.assert_none_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('!=', 'assert_none_equal', 6)
def assert_none_equal_v2(x, y, summarize=None, message=None, name=None):
  return assert_none_equal(x=x, y=y, summarize=summarize, message=message,
                           name=name)


@tf_export(v1=['debugging.assert_none_equal', 'assert_none_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_none_equal')
@_binary_assert_doc('!=', '[2, 1]')
def assert_none_equal(
    x, y, data=None, summarize=None, message=None, name=None):
  return _binary_assert('!=', 'assert_none_equal', math_ops.not_equal,
                        np.not_equal, x, y, data, summarize, message, name)


@tf_export('debugging.assert_near', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
def assert_near_v2(x, y, rtol=None, atol=None, message=None, summarize=None,
                   name=None):
  """Assert the condition `x` and `y` are close element-wise.

  This Op checks that `x[i] - y[i] < atol + rtol * tf.abs(y[i])` holds for every
  pair of (possibly broadcast) elements of `x` and `y`. If both `x` and `y` are
  empty, this is trivially satisfied.

  If any elements of `x` and `y` are not close, `message`, as well as the first
  `summarize` entries of `x` and `y` are printed, and `InvalidArgumentError`
  is raised.

  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest
  representable positive number such that `1 + eps != 1`.  This is about
  `1.2e-6` in `32bit`, `2.22e-15` in `64bit`, and `0.00977` in `16bit`.
  See `numpy.finfo`.

  Args:
    x: Float or complex `Tensor`.
    y: Float or complex `Tensor`, same dtype as and broadcastable to `x`.
    rtol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The relative tolerance.  Default is `10 * eps`.
    atol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The absolute tolerance.  Default is `10 * eps`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_near".

  Returns:
    Op that raises `InvalidArgumentError` if `x` and `y` are not close enough.
      This can be used with `tf.control_dependencies` inside of `tf.function`s
      to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x != y` is False for any pair of elements in `x` and `y`. The check can
      be performed immediately during eager execution or if `x` and `y` are
      statically known.

  @compatibility(numpy)
  Similar to `numpy.testing.assert_allclose`, except tolerance depends on data
  type. This is due to the fact that `TensorFlow` is often used with `32bit`,
  `64bit`, and even `16bit` data.
  @end_compatibility
  """
  return assert_near(x=x, y=y, rtol=rtol, atol=atol, summarize=summarize,
                     message=message, name=name)


@tf_export(v1=['debugging.assert_near', 'assert_near'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_near')
def assert_near(
    x, y, rtol=None, atol=None, data=None, summarize=None, message=None,
    name=None):
  """Assert the condition `x` and `y` are close element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_near(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have

  ```tf.abs(x[i] - y[i]) <= atol + rtol * tf.abs(y[i])```.

  If both `x` and `y` are empty, this is trivially satisfied.

  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest
  representable positive number such that `1 + eps != 1`.  This is about
  `1.2e-6` in `32bit`, `2.22e-15` in `64bit`, and `0.00977` in `16bit`.
  See `numpy.finfo`.

  Args:
    x:  Float or complex `Tensor`.
    y:  Float or complex `Tensor`, same `dtype` as, and broadcastable to, `x`.
    rtol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The relative tolerance.  Default is `10 * eps`.
    atol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The absolute tolerance.  Default is `10 * eps`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_near".

  Returns:
    Op that raises `InvalidArgumentError` if `x` and `y` are not close enough.

  @compatibility(numpy)
  Similar to `numpy.testing.assert_allclose`, except tolerance depends on data
  type. This is due to the fact that `TensorFlow` is often used with `32bit`,
  `64bit`, and even `16bit` data.
  @end_compatibility
  """
  message = _message_prefix(message)
  with ops.name_scope(name, 'assert_near', [x, y, rtol, atol, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y', dtype=x.dtype)

    dtype = x.dtype
    if dtype.is_complex:
      dtype = dtype.real_dtype
    eps = np.finfo(dtype.as_numpy_dtype).eps
    rtol = 10 * eps if rtol is None else rtol
    atol = 10 * eps if atol is None else atol

    rtol = ops.convert_to_tensor(rtol, name='rtol', dtype=dtype)
    atol = ops.convert_to_tensor(atol, name='atol', dtype=dtype)

    if context.executing_eagerly():
      x_name = _shape_and_dtype_str(x)
      y_name = _shape_and_dtype_str(y)
    else:
      x_name = x.name
      y_name = y.name

    if data is None:
      data = [
          message,
          'x and y not equal to tolerance rtol = %s, atol = %s' % (rtol, atol),
          'x (%s) = ' % x_name, x, 'y (%s) = ' % y_name, y
      ]
    tol = atol + rtol * math_ops.abs(y)
    diff = math_ops.abs(x - y)
    condition = math_ops.reduce_all(math_ops.less(diff, tol))
    return control_flow_assert.Assert(condition, data, summarize=summarize)


@tf_export('debugging.assert_less', 'assert_less', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('<', 'assert_less', 3)
def assert_less_v2(x, y, message=None, summarize=None, name=None):
  return assert_less(x=x, y=y, summarize=summarize, message=message, name=name)


@tf_export(v1=['debugging.assert_less', 'assert_less'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('<', '[2, 3]')
def assert_less(x, y, data=None, summarize=None, message=None, name=None):
  return _binary_assert('<', 'assert_less', math_ops.less, np.less, x, y, data,
                        summarize, message, name)


@tf_export('debugging.assert_less_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('<=', 'assert_less_equal', 3)
def assert_less_equal_v2(x, y, message=None, summarize=None, name=None):
  return assert_less_equal(x=x, y=y,
                           summarize=summarize, message=message, name=name)


@tf_export(v1=['debugging.assert_less_equal', 'assert_less_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_less_equal')
@_binary_assert_doc('<=', '[1, 3]')
def assert_less_equal(x, y, data=None, summarize=None, message=None, name=None):
  return _binary_assert('<=', 'assert_less_equal', math_ops.less_equal,
                        np.less_equal, x, y, data, summarize, message, name)


@tf_export('debugging.assert_greater', 'assert_greater', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('>', 'assert_greater', 9)
def assert_greater_v2(x, y, message=None, summarize=None, name=None):
  return assert_greater(x=x, y=y, summarize=summarize, message=message,
                        name=name)


@tf_export(v1=['debugging.assert_greater', 'assert_greater'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('>', '[0, 1]')
def assert_greater(x, y, data=None, summarize=None, message=None, name=None):  # pylint: disable=missing-docstring
  return _binary_assert('>', 'assert_greater', math_ops.greater, np.greater, x,
                        y, data, summarize, message, name)


@tf_export('debugging.assert_greater_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('>=', 'assert_greater_equal', 9)
def assert_greater_equal_v2(x, y, message=None, summarize=None, name=None):
  return assert_greater_equal(x=x, y=y, summarize=summarize, message=message,
                              name=name)


@tf_export(v1=['debugging.assert_greater_equal', 'assert_greater_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_greater_equal')
@_binary_assert_doc('>=', '[1, 0]')
def assert_greater_equal(x, y, data=None, summarize=None, message=None,
                         name=None):
  return _binary_assert('>=', 'assert_greater_equal', math_ops.greater_equal,
                        np.greater_equal, x, y, data, summarize, message, name)


def _assert_rank_condition(
    x, rank, static_condition, dynamic_condition, data, summarize):
  """Assert `x` has a rank that satisfies a given condition.

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    static_condition:   A python function that takes `[actual_rank, given_rank]`
      and returns `True` if the condition is satisfied, `False` otherwise.
    dynamic_condition:  An `op` that takes [actual_rank, given_rank] and return
      `True` if the condition is satisfied, `False` otherwise.
    data:  The tensors to print out if the condition is false.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.

  Returns:
    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.

  Raises:
    ValueError:  If static checks determine `x` fails static_condition.
  """
  assert_type(rank, dtypes.int32)

  # Attempt to statically defined rank.
  rank_static = tensor_util.constant_value(rank)
  if rank_static is not None:
    if rank_static.ndim != 0:
      raise ValueError('Rank must be a scalar.')

    x_rank_static = x.get_shape().ndims
    if x_rank_static is not None:
      if not static_condition(x_rank_static, rank_static):
        raise ValueError(
            'Static rank condition failed', x_rank_static, rank_static)
      return control_flow_ops.no_op(name='static_checks_determined_all_ok')

  condition = dynamic_condition(array_ops.rank(x), rank)

  # Add the condition that `rank` must have rank zero.  Prevents the bug where
  # someone does assert_rank(x, [n]), rather than assert_rank(x, n).
  if rank_static is None:
    this_data = ['Rank must be a scalar. Received rank: ', rank]
    rank_check = assert_rank(rank, 0, data=this_data)
    condition = control_flow_ops.with_dependencies([rank_check], condition)

  return control_flow_assert.Assert(condition, data, summarize=summarize)


@tf_export('debugging.assert_rank', 'assert_rank', v1=[])
@dispatch.add_dispatch_support
def assert_rank_v2(x, rank, message=None, name=None):
  """Assert that `x` has rank equal to `rank`.

  This Op checks that the rank of `x` is equal to `rank`.

  If `x` has a different rank, `message`, as well as the shape of `x` are
  printed, and `InvalidArgumentError` is raised.

  Args:
    x: `Tensor`.
    rank: Scalar integer `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to
      "assert_rank".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.
    If static checks determine `x` has correct rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x` does not have rank `rank`. The check can be performed immediately
      during eager execution or if the shape of `x` is statically known.
  """
  return assert_rank(x=x, rank=rank, message=message, name=name)


@tf_export(v1=['debugging.assert_rank', 'assert_rank'])
@dispatch.add_dispatch_support
def assert_rank(x, rank, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar integer `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and the shape of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_rank".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
  with ops.name_scope(name, 'assert_rank', (x, rank) + tuple(data or [])):
    if not isinstance(x, sparse_tensor.SparseTensor):
      x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')
    message = _message_prefix(message)

    static_condition = lambda actual_rank, given_rank: actual_rank == given_rank
    dynamic_condition = math_ops.equal

    if context.executing_eagerly() or isinstance(x, sparse_tensor.SparseTensor):
      name = ''
    else:
      name = x.name

    if data is None:
      data = [
          message,
          'Tensor %s must have rank' % name, rank, 'Received shape: ',
          array_ops.shape(x)
      ]

    try:
      assert_op = _assert_rank_condition(x, rank, static_condition,
                                         dynamic_condition, data, summarize)

    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%sTensor %s must have rank %d.  Received rank %d, shape %s' %
            (message, name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise ValueError(e.args[0])

  return assert_op


@tf_export('debugging.assert_rank_at_least', v1=[])
@dispatch.add_dispatch_support
def assert_rank_at_least_v2(x, rank, message=None, name=None):
  """Assert that `x` has rank of at least `rank`.

  This Op checks that the rank of `x` is greater or equal to `rank`.

  If `x` has a rank lower than `rank`, `message`, as well as the shape of `x`
  are printed, and `InvalidArgumentError` is raised.

  Args:
    x: `Tensor`.
    rank: Scalar integer `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to
      "assert_rank_at_least".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.
    If static checks determine `x` has correct rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: `x` does not have rank at least `rank`, but the rank
      cannot be statically determined.
    ValueError: If static checks determine `x` has mismatched rank.
  """
  return assert_rank_at_least(x=x, rank=rank, message=message, name=name)


@tf_export(v1=['debugging.assert_rank_at_least', 'assert_rank_at_least'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_rank_at_least')
def assert_rank_at_least(
    x, rank, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank` or higher.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank_at_least(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_at_least".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
  with ops.name_scope(
      name, 'assert_rank_at_least', (x, rank) + tuple(data or [])):
    x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')
    message = _message_prefix(message)

    static_condition = lambda actual_rank, given_rank: actual_rank >= given_rank
    dynamic_condition = math_ops.greater_equal

    if context.executing_eagerly():
      name = ''
    else:
      name = x.name

    if data is None:
      data = [
          message,
          'Tensor %s must have rank at least' % name, rank,
          'Received shape: ', array_ops.shape(x)
      ]

    try:
      assert_op = _assert_rank_condition(x, rank, static_condition,
                                         dynamic_condition, data, summarize)

    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%sTensor %s must have rank at least %d.  Received rank %d, '
            'shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise

  return assert_op


def _static_rank_in(actual_rank, given_ranks):
  return actual_rank in given_ranks


def _dynamic_rank_in(actual_rank, given_ranks):
  if len(given_ranks) < 1:
    return ops.convert_to_tensor(False)
  result = math_ops.equal(given_ranks[0], actual_rank)
  for given_rank in given_ranks[1:]:
    result = math_ops.logical_or(
        result, math_ops.equal(given_rank, actual_rank))
  return result


def _assert_ranks_condition(
    x, ranks, static_condition, dynamic_condition, data, summarize):
  """Assert `x` has a rank that satisfies a given condition.

  Args:
    x:  Numeric `Tensor`.
    ranks:  Scalar `Tensor`.
    static_condition:   A python function that takes
      `[actual_rank, given_ranks]` and returns `True` if the condition is
      satisfied, `False` otherwise.
    dynamic_condition:  An `op` that takes [actual_rank, given_ranks]
      and return `True` if the condition is satisfied, `False` otherwise.
    data:  The tensors to print out if the condition is false.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.

  Returns:
    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.

  Raises:
    ValueError:  If static checks determine `x` fails static_condition.
  """
  for rank in ranks:
    assert_type(rank, dtypes.int32)

  # Attempt to statically defined rank.
  ranks_static = tuple([tensor_util.constant_value(rank) for rank in ranks])
  if not any(r is None for r in ranks_static):
    for rank_static in ranks_static:
      if rank_static.ndim != 0:
        raise ValueError('Rank must be a scalar.')

    x_rank_static = x.get_shape().ndims
    if x_rank_static is not None:
      if not static_condition(x_rank_static, ranks_static):
        raise ValueError(
            'Static rank condition failed', x_rank_static, ranks_static)
      return control_flow_ops.no_op(name='static_checks_determined_all_ok')

  condition = dynamic_condition(array_ops.rank(x), ranks)

  # Add the condition that `rank` must have rank zero.  Prevents the bug where
  # someone does assert_rank(x, [n]), rather than assert_rank(x, n).
  for rank, rank_static in zip(ranks, ranks_static):
    if rank_static is None:
      this_data = ['Rank must be a scalar. Received rank: ', rank]
      rank_check = assert_rank(rank, 0, data=this_data)
      condition = control_flow_ops.with_dependencies([rank_check], condition)

  return control_flow_assert.Assert(condition, data, summarize=summarize)


@tf_export('debugging.assert_rank_in', v1=[])
@dispatch.add_dispatch_support
def assert_rank_in_v2(x, ranks, message=None, name=None):
  """Assert that `x` has a rank in `ranks`.

  This Op checks that the rank of `x` is in `ranks`.

  If `x` has a different rank, `message`, as well as the shape of `x` are
  printed, and `InvalidArgumentError` is raised.

  Args:
    x: `Tensor`.
    ranks: `Iterable` of scalar `Tensor` objects.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to "assert_rank_in".

  Returns:
    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
    If static checks determine `x` has matching rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: `x` does not have rank in `ranks`, but the rank cannot
      be statically determined.
    ValueError: If static checks determine `x` has mismatched rank.
  """
  return assert_rank_in(x=x, ranks=ranks, message=message, name=name)


@tf_export(v1=['debugging.assert_rank_in', 'assert_rank_in'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_rank_in')
def assert_rank_in(
    x, ranks, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank in `ranks`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank_in(x, (2, 4))]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    ranks:  Iterable of scalar `Tensor` objects.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_in".

  Returns:
    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
    If static checks determine `x` has matching rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has mismatched rank.
  """
  with ops.name_scope(
      name, 'assert_rank_in', (x,) + tuple(ranks) + tuple(data or [])):
    if not isinstance(x, sparse_tensor.SparseTensor):
      x = ops.convert_to_tensor(x, name='x')
    ranks = tuple([ops.convert_to_tensor(rank, name='rank') for rank in ranks])
    message = _message_prefix(message)

    if context.executing_eagerly() or isinstance(x, sparse_tensor.SparseTensor):
      name = ''
    else:
      name = x.name

    if data is None:
      data = [
          message, 'Tensor %s must have rank in' % name
      ] + list(ranks) + [
          'Received shape: ', array_ops.shape(x)
      ]

    try:
      assert_op = _assert_ranks_condition(x, ranks, _static_rank_in,
                                          _dynamic_rank_in, data, summarize)

    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%sTensor %s must have rank in %s.  Received rank %d, shape %s'
            % (
                message,
                name,
                tuple(r.item() for r in e.args[2]),
                e.args[1],
                x.get_shape(),
            )
        )
      else:
        raise

  return assert_op


@tf_export('debugging.assert_integer', v1=[])
@dispatch.add_dispatch_support
def assert_integer_v2(x, message=None, name=None):
  """Assert that `x` is of integer dtype.

  If `x` has a non-integer type, `message`, as well as the dtype of `x` are
  printed, and `InvalidArgumentError` is raised.

  This can always be checked statically, so this method returns nothing.

  Args:
    x: A `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to "assert_integer".

  Raises:
    TypeError:  If `x.dtype` is not a non-quantized integer type.
  """
  assert_integer(x=x, message=message, name=name)


@tf_export(v1=['debugging.assert_integer', 'assert_integer'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_integer')
def assert_integer(x, message=None, name=None):
  """Assert that `x` is of integer dtype.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_integer(x)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x: `Tensor` whose basetype is integer and is not quantized.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_integer".

  Raises:
    TypeError:  If `x.dtype` is anything other than non-quantized integer.

  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  """
  with ops.name_scope(name, 'assert_integer', [x]):
    x = ops.convert_to_tensor(x, name='x')
    if not x.dtype.is_integer:
      if context.executing_eagerly():
        name = 'tensor'
      else:
        name = x.name
      err_msg = (
          '%sExpected "x" to be integer type.  Found: %s of dtype %s'
          % (_message_prefix(message), name, x.dtype))
      raise TypeError(err_msg)

    return control_flow_ops.no_op('statically_determined_was_integer')


@tf_export('debugging.assert_type', v1=[])
@dispatch.add_dispatch_support
def assert_type_v2(tensor, tf_type, message=None, name=None):
  """Asserts that the given `Tensor` is of the specified type.

  This can always be checked statically, so this method returns nothing.

  Example:

  >>> a = tf.Variable(1.0)
  >>> tf.debugging.assert_type(a, tf_type= tf.float32)

  >>> b = tf.constant(21)
  >>> tf.debugging.assert_type(b, tf_type=tf.bool)
  Traceback (most recent call last):
  ...
  TypeError: ...

  >>> c = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2],
  ...  dense_shape=[3, 4])
  >>> tf.debugging.assert_type(c, tf_type= tf.int32)

  Args:
    tensor: A `Tensor`, `SparseTensor` or `tf.Variable` .
    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,
      etc).
    message: A string to prefix to the default message.
    name:  A name for this operation. Defaults to "assert_type"

  Raises:
    TypeError: If the tensor's data type doesn't match `tf_type`.
  """
  assert_type(tensor=tensor, tf_type=tf_type, message=message, name=name)


@tf_export(v1=['debugging.assert_type', 'assert_type'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_type')
def assert_type(tensor, tf_type, message=None, name=None):
  """Statically asserts that the given `Tensor` is of the specified type.

  Args:
    tensor: A `Tensor` or `SparseTensor`.
    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,
      etc).
    message: A string to prefix to the default message.
    name:  A name to give this `Op`.  Defaults to "assert_type"

  Raises:
    TypeError: If the tensors data type doesn't match `tf_type`.

  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  """
  tf_type = dtypes.as_dtype(tf_type)
  with ops.name_scope(name, 'assert_type', [tensor]):
    if not isinstance(tensor, sparse_tensor.SparseTensor):
      tensor = ops.convert_to_tensor(tensor, name='tensor')
    if tensor.dtype != tf_type:
      raise TypeError(
          f'{_message_prefix(message)}{getattr(tensor, "name", "tensor")}'
          f' must be of type {tf_type!r}; got {tensor.dtype!r}')

    return control_flow_ops.no_op('statically_determined_correct_type')


def _dimension_sizes(x):
  """Gets the dimension sizes of a tensor `x`.

  If a size can be determined statically it is returned as an integer,
  otherwise as a tensor.

  If `x` is a scalar it is treated as rank 1 size 1.

  Args:
    x: A `Tensor`.

  Returns:
    Dimension sizes.
  """
  dynamic_shape = array_ops.shape(x)
  rank = x.get_shape().rank
  rank_is_known = rank is not None
  if rank_is_known and rank == 0:
    return (1,)
  if rank_is_known and rank > 0:
    static_shape = x.get_shape().as_list()
    sizes = [
        int(size) if size is not None else dynamic_shape[i]
        for i, size in enumerate(static_shape)
    ]
    return sizes
  has_rank_zero = math_ops.equal(array_ops.rank(x), 0)
  return cond.cond(
      has_rank_zero, lambda: array_ops.constant([1]), lambda: dynamic_shape)


def _symbolic_dimension_sizes(symbolic_shape):
  # If len(symbolic_shape) == 0 construct a tuple
  if not symbolic_shape:
    return tuple([1])

  return symbolic_shape


def _has_known_value(dimension_size):
  not_none = dimension_size is not None
  try:
    int(dimension_size)
    can_be_parsed_as_int = True
  except (ValueError, TypeError):
    can_be_parsed_as_int = False
  return not_none and can_be_parsed_as_int


def _is_symbol_for_any_size(symbol):
  return symbol in [None, '.']


_TensorDimSizes = collections.namedtuple(
    '_TensorDimSizes',
    ['x', 'unspecified_dim', 'actual_sizes', 'symbolic_sizes'])


@tf_export('debugging.assert_shapes', v1=[])
@dispatch.add_dispatch_support
def assert_shapes_v2(shapes, data=None, summarize=None, message=None,
                     name=None):
  """Assert tensor shapes and dimension size relationships between tensors.

  This Op checks that a collection of tensors shape relationships
  satisfies given constraints.

  Example:

  >>> n = 10
  >>> q = 3
  >>> d = 7
  >>> x = tf.zeros([n,q])
  >>> y = tf.ones([n,d])
  >>> param = tf.Variable([1.0, 2.0, 3.0])
  >>> scalar = 1.0
  >>> tf.debugging.assert_shapes([
  ...  (x, ('N', 'Q')),
  ...  (y, ('N', 'D')),
  ...  (param, ('Q',)),
  ...  (scalar, ()),
  ... ])

  >>> tf.debugging.assert_shapes([
  ...   (x, ('N', 'D')),
  ...   (y, ('N', 'D'))
  ... ])
  Traceback (most recent call last):
  ...
  ValueError: ...

  If `x`, `y`, `param` or `scalar` does not have a shape that satisfies
  all specified constraints, `message`, as well as the first `summarize` entries
  of the first encountered violating tensor are printed, and
  `InvalidArgumentError` is raised.

  Size entries in the specified shapes are checked against other entries by
  their __hash__, except:
    - a size entry is interpreted as an explicit size if it can be parsed as an
      integer primitive.
    - a size entry is interpreted as *any* size if it is None or '.'.

  If the first entry of a shape is `...` (type `Ellipsis`) or '*' that indicates
  a variable number of outer dimensions of unspecified size, i.e. the constraint
  applies to the inner-most dimensions only.

  Scalar tensors and specified shapes of length zero (excluding the 'inner-most'
  prefix) are both treated as having a single dimension of size one.

  Args:
    shapes: dictionary with (`Tensor` to shape) items, or a list of
      (`Tensor`, shape) tuples. A shape must be an iterable.
    data: The tensors to print out if the condition is False.  Defaults to error
      message and first few entries of the violating tensor.
    summarize: Print this many entries of the tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_shapes".

  Raises:
    ValueError:  If static checks determine any shape constraint is violated.
  """
  assert_shapes(
      shapes, data=data, summarize=summarize, message=message, name=name)


@tf_export(v1=['debugging.assert_shapes'])
@dispatch.add_dispatch_support
def assert_shapes(shapes, data=None, summarize=None, message=None, name=None):
  """Assert tensor shapes and dimension size relationships between tensors.

  This Op checks that a collection of tensors shape relationships
  satisfies given constraints.

  Example:

  >>> n = 10
  >>> q = 3
  >>> d = 7
  >>> x = tf.zeros([n,q])
  >>> y = tf.ones([n,d])
  >>> param = tf.Variable([1.0, 2.0, 3.0])
  >>> scalar = 1.0
  >>> tf.debugging.assert_shapes([
  ...  (x, ('N', 'Q')),
  ...  (y, ('N', 'D')),
  ...  (param, ('Q',)),
  ...  (scalar, ()),
  ... ])

  >>> tf.debugging.assert_shapes([
  ...   (x, ('N', 'D')),
  ...   (y, ('N', 'D'))
  ... ])
  Traceback (most recent call last):
  ...
  ValueError: ...

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_shapes(shapes)]):
    output = tf.matmul(x, y, transpose_a=True)
  ```

  If `x`, `y`, `param` or `scalar` does not have a shape that satisfies
  all specified constraints, `message`, as well as the first `summarize` entries
  of the first encountered violating tensor are printed, and
  `InvalidArgumentError` is raised.

  Size entries in the specified shapes are checked against other entries by
  their __hash__, except:
    - a size entry is interpreted as an explicit size if it can be parsed as an
      integer primitive.
    - a size entry is interpreted as *any* size if it is None or '.'.

  If the first entry of a shape is `...` (type `Ellipsis`) or '*' that indicates
  a variable number of outer dimensions of unspecified size, i.e. the constraint
  applies to the inner-most dimensions only.

  Scalar tensors and specified shapes of length zero (excluding the 'inner-most'
  prefix) are both treated as having a single dimension of size one.

  Args:
    shapes: A list of (`Tensor`, `shape`) tuples, wherein `shape` is the
      expected shape of `Tensor`. See the example code above. The `shape` must
      be an iterable. Each element of the iterable can be either a concrete
      integer value or a string that abstractly represents the dimension.
      For example,
        - `('N', 'Q')` specifies a 2D shape wherein the first and second
          dimensions of shape may or may not be equal.
        - `('N', 'N', 'Q')` specifies a 3D shape wherein the first and second
          dimensions are equal.
        - `(1, 'N')` specifies a 2D shape wherein the first dimension is
          exactly 1 and the second dimension can be any value.
      Note that the abstract dimension letters take effect across different
      tuple elements of the list. For example,
      `tf.debugging.assert_shapes([(x, ('N', 'A')), (y, ('N', 'B'))]` asserts
      that both `x` and `y` are rank-2 tensors and their first dimensions are
      equal (`N`).
      `shape` can also be a `tf.TensorShape`.
    data: The tensors to print out if the condition is False.  Defaults to error
      message and first few entries of the violating tensor.
    summarize: Print this many entries of the tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_shapes".

  Returns:
    Op raising `InvalidArgumentError` unless all shape constraints are
    satisfied.
    If static checks determine all constraints are satisfied, a `no_op` is
    returned.

  Raises:
    ValueError:  If static checks determine any shape constraint is violated.
  """
  # If the user manages to assemble a dict containing tensors (possible in
  # Graph mode only), make sure we still accept that.
  if isinstance(shapes, dict):
    shapes = shapes.items()

  message_prefix = _message_prefix(message)
  with ops.name_scope(name, 'assert_shapes', [shapes, data]):
    # Shape specified as None implies no constraint
    shape_constraints = [(x if isinstance(x, sparse_tensor.SparseTensor) else
                          ops.convert_to_tensor(x), s)
                         for x, s in shapes if s is not None]

    executing_eagerly = context.executing_eagerly()

    def tensor_name(x):
      if executing_eagerly or isinstance(x, sparse_tensor.SparseTensor):
        return _shape_and_dtype_str(x)
      return x.name

    tensor_dim_sizes = []
    for tensor, symbolic_shape in shape_constraints:
      is_iterable = (
          hasattr(symbolic_shape, '__iter__') or
          hasattr(symbolic_shape, '__getitem__')  # For Python 2 compat.
      )
      if not is_iterable:
        raise ValueError(
            '%s'
            'Tensor %s.  Specified shape must be an iterable.  '
            'An iterable has the attribute `__iter__` or `__getitem__`.  '
            'Received specified shape: %s' %
            (message_prefix, tensor_name(tensor), symbolic_shape))

      # We convert this into a tuple to handle strings, lists and numpy arrays
      symbolic_shape_tuple = tuple(symbolic_shape)

      tensors_specified_innermost = False
      for i, symbol in enumerate(symbolic_shape_tuple):
        if symbol not in [Ellipsis, '*']:
          continue

        if i != 0:
          raise ValueError(
              '%s'
              'Tensor %s specified shape index %d.  '
              'Symbol `...` or `*` for a variable number of '
              'unspecified dimensions is only allowed as the first entry' %
              (message_prefix, tensor_name(tensor), i))

        tensors_specified_innermost = True

      # Only include the size of the specified dimensions since the 0th symbol
      # is either ellipsis or *
      tensor_dim_sizes.append(
          _TensorDimSizes(
              tensor, tensors_specified_innermost, _dimension_sizes(tensor),
              _symbolic_dimension_sizes(
                  symbolic_shape_tuple[1:]
                  if tensors_specified_innermost else symbolic_shape_tuple)))

    rank_assertions = []
    for sizes in tensor_dim_sizes:
      rank = len(sizes.symbolic_sizes)
      rank_zero_or_one = rank in [0, 1]
      if sizes.unspecified_dim:
        if rank_zero_or_one:
          # No assertion of rank needed as `x` only need to have rank at least
          # 0. See elif rank_zero_or_one case comment.
          continue
        assertion = assert_rank_at_least(
            x=sizes.x,
            rank=rank,
            data=data,
            summarize=summarize,
            message=message,
            name=name)
      elif rank_zero_or_one:
        # Rank 0 is treated as rank 1 size 1, i.e. there is
        # no distinction between the two in terms of rank.
        # See _dimension_sizes.
        assertion = assert_rank_in(
            x=sizes.x,
            ranks=[0, 1],
            data=data,
            summarize=summarize,
            message=message,
            name=name)
      else:
        assertion = assert_rank(
            x=sizes.x,
            rank=rank,
            data=data,
            summarize=summarize,
            message=message,
            name=name)
      rank_assertions.append(assertion)

    size_assertions = []
    size_specifications = {}
    for sizes in tensor_dim_sizes:
      for i, size_symbol in enumerate(sizes.symbolic_sizes):

        if _is_symbol_for_any_size(size_symbol):
          # Size specified as any implies no constraint
          continue

        if sizes.unspecified_dim:
          tensor_dim = i - len(sizes.symbolic_sizes)
        else:
          tensor_dim = i

        if size_symbol in size_specifications or _has_known_value(size_symbol):
          if _has_known_value(size_symbol):
            specified_size = int(size_symbol)
            size_check_message = 'Specified explicitly'
          else:
            specified_size, specified_by_y, specified_at_dim = (
                size_specifications[size_symbol])
            size_check_message = (
                'Specified by tensor %s dimension %d' %
                (tensor_name(specified_by_y), specified_at_dim))

          # This is extremely subtle. If actual_sizes is dynamic, we must
          # make sure a control dependency is inserted here so that this slice
          # can not execute until the rank is asserted to be enough for the
          # slice to not fail.
          with ops.control_dependencies(rank_assertions):
            actual_size = sizes.actual_sizes[tensor_dim]
          if _has_known_value(actual_size) and _has_known_value(specified_size):
            if int(actual_size) != int(specified_size):
              raise ValueError(
                  '%s%s.  Tensor %s dimension %s must have size %d.  '
                  'Received size %d, shape %s' %
                  (message_prefix, size_check_message, tensor_name(sizes.x),
                   tensor_dim, specified_size, actual_size,
                   sizes.x.get_shape()))
            # No dynamic assertion needed
            continue

          condition = math_ops.equal(
              ops.convert_to_tensor(actual_size),
              ops.convert_to_tensor(specified_size))
          data_ = data
          if data is None:
            data_ = [
                message_prefix, size_check_message,
                'Tensor %s dimension' % tensor_name(sizes.x), tensor_dim,
                'must have size', specified_size, 'Received shape: ',
                array_ops.shape(sizes.x)
            ]
          size_assertions.append(
              control_flow_assert.Assert(condition, data_, summarize=summarize))
        else:
          # Not sure if actual_sizes is a constant, but for safety, guard
          # on rank. See explanation above about actual_sizes need for safety.
          with ops.control_dependencies(rank_assertions):
            size = sizes.actual_sizes[tensor_dim]
          size_specifications[size_symbol] = (size, sizes.x, tensor_dim)

  # Ensure both assertions actually occur.
  with ops.control_dependencies(rank_assertions):
    shapes_assertion = control_flow_ops.group(size_assertions)

  return shapes_assertion


# pylint: disable=line-too-long
def _get_results_for_monotonic_comparison(x, compare_op):
  """Gets the difference x[1:] - x[:-1]."""
  x = array_ops.reshape(x, [-1])
  if not is_numeric_tensor(x):
    raise TypeError('Expected x to be numeric, instead found: %s' % x)

  # If x has less than 2 elements, there is nothing to compare.  So return [].
  is_shorter_than_two = math_ops.less(array_ops.size(x), 2)
  short_result = lambda: ops.convert_to_tensor([], dtype=bool)

  # With 2 or more elements, return x[1:] - x[:-1]
  s_len = array_ops.shape(x) - 1
  diff = lambda: compare_op(
      array_ops.strided_slice(x, [1], [1] + s_len),
      array_ops.strided_slice(x, [0], s_len),
  )
  return cond.cond(is_shorter_than_two, short_result, diff)


@tf_export(
    'debugging.is_numeric_tensor',
    v1=['debugging.is_numeric_tensor', 'is_numeric_tensor'])
@deprecation.deprecated_endpoints('is_numeric_tensor')
def is_numeric_tensor(tensor):
  """Returns `True` if the elements of `tensor` are numbers.

  Specifically, returns `True` if the dtype of `tensor` is one of the following:

  * `tf.float16`
  * `tf.float32`
  * `tf.float64`
  * `tf.int8`
  * `tf.int16`
  * `tf.int32`
  * `tf.int64`
  * `tf.uint8`
  * `tf.uint16`
  * `tf.uint32`
  * `tf.uint64`
  * `tf.qint8`
  * `tf.qint16`
  * `tf.qint32`
  * `tf.quint8`
  * `tf.quint16`
  * `tf.complex64`
  * `tf.complex128`
  * `tf.bfloat16`

  Returns `False` if `tensor` is of a non-numeric type or if `tensor` is not
  a `tf.Tensor` object.
  """
  return isinstance(tensor, tensor_lib.Tensor) and tensor.dtype in NUMERIC_TYPES


@tf_export(
    'math.is_non_decreasing',
    v1=[
        'math.is_non_decreasing', 'debugging.is_non_decreasing',
        'is_non_decreasing'
    ])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('debugging.is_non_decreasing',
                                  'is_non_decreasing')
def is_non_decreasing(x, name=None):
  """Returns `True` if `x` is non-decreasing.

  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
  If `x` has less than two elements, it is trivially non-decreasing.

  See also:  `is_strictly_increasing`

  >>> x1 = tf.constant([1.0, 1.0, 3.0])
  >>> tf.math.is_non_decreasing(x1)
  <tf.Tensor: shape=(), dtype=bool, numpy=True>
  >>> x2 = tf.constant([3.0, 1.0, 2.0])
  >>> tf.math.is_non_decreasing(x2)
  <tf.Tensor: shape=(), dtype=bool, numpy=False>

  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).  Defaults to "is_non_decreasing"

  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.

  Raises:
    TypeError: if `x` is not a numeric tensor.
  """
  with ops.name_scope(name, 'is_non_decreasing', [x]):
    diff = _get_results_for_monotonic_comparison(x, math_ops.greater_equal)
    # When len(x) = 1, diff = [], less_equal = [], and reduce_all([]) = True.
    return math_ops.reduce_all(diff)


@tf_export(
    'math.is_strictly_increasing',
    v1=[
        'math.is_strictly_increasing', 'debugging.is_strictly_increasing',
        'is_strictly_increasing'
    ])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('debugging.is_strictly_increasing',
                                  'is_strictly_increasing')
def is_strictly_increasing(x, name=None):
  """Returns `True` if `x` is strictly increasing.

  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.
  If `x` has less than two elements, it is trivially strictly increasing.

  See also:  `is_non_decreasing`

  >>> x1 = tf.constant([1.0, 2.0, 3.0])
  >>> tf.math.is_strictly_increasing(x1)
  <tf.Tensor: shape=(), dtype=bool, numpy=True>
  >>> x2 = tf.constant([3.0, 1.0, 2.0])
  >>> tf.math.is_strictly_increasing(x2)
  <tf.Tensor: shape=(), dtype=bool, numpy=False>

  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).
      Defaults to "is_strictly_increasing"

  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is strictly increasing.

  Raises:
    TypeError: if `x` is not a numeric tensor.
  """
  with ops.name_scope(name, 'is_strictly_increasing', [x]):
    diff = _get_results_for_monotonic_comparison(x, math_ops.greater)
    # When len(x) = 1, diff = [], less = [], and reduce_all([]) = True.
    return math_ops.reduce_all(diff)


def _assert_same_base_type(items, expected_type=None):
  r"""Asserts all items are of the same base type.

  Args:
    items: List of graph items (e.g., `Variable`, `Tensor`, `SparseTensor`,
        `Operation`, or `IndexedSlices`). Can include `None` elements, which
        will be ignored.
    expected_type: Expected type. If not specified, assert all items are
        of the same base type.

  Returns:
    Validated type, or none if neither expected_type nor items provided.

  Raises:
    ValueError: If any types do not match.
  """
  original_expected_type = expected_type
  mismatch = False
  for item in items:
    if item is not None:
      item_type = item.dtype.base_dtype
      if not expected_type:
        expected_type = item_type
      elif expected_type != item_type:
        mismatch = True
        break
  if mismatch:
    # Loop back through and build up an informative error message (this is very
    # slow, so we don't do it unless we found an error above).
    expected_type = original_expected_type
    original_item_str = None
    for item in items:
      if item is not None:
        item_type = item.dtype.base_dtype
        if not expected_type:
          expected_type = item_type
          original_item_str = item.name if hasattr(item, 'name') else str(item)
        elif expected_type != item_type:
          raise ValueError('%s, type=%s, must be of the same type (%s)%s.' % (
              item.name if hasattr(item, 'name') else str(item),
              item_type, expected_type,
              (' as %s' % original_item_str) if original_item_str else ''))
    return expected_type  # Should be unreachable
  else:
    return expected_type


@tf_export(
    'debugging.assert_same_float_dtype',
    v1=['debugging.assert_same_float_dtype', 'assert_same_float_dtype'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_same_float_dtype')
def assert_same_float_dtype(tensors=None, dtype=None):
  """Validate and return float type based on `tensors` and `dtype`.

  For ops such as matrix multiplication, inputs and weights must be of the
  same float type. This function validates that all `tensors` are the same type,
  validates that type is `dtype` (if supplied), and returns the type. Type must
  be a floating point type. If neither `tensors` nor `dtype` is supplied,
  the function will return `dtypes.float32`.

  Args:
    tensors: Tensors of input values. Can include `None` elements, which will be
        ignored.
    dtype: Expected type.

  Returns:
    Validated type.

  Raises:
    ValueError: if neither `tensors` nor `dtype` is supplied, or result is not
        float, or the common type of the inputs is not a floating point type.
  """
  if tensors:
    dtype = _assert_same_base_type(tensors, dtype)
  if not dtype:
    dtype = dtypes.float32
  elif not dtype.is_floating:
    raise ValueError('Expected floating point type, got %s.' % dtype)
  return dtype


@tf_export('debugging.assert_scalar', v1=[])
@dispatch.add_dispatch_support
def assert_scalar_v2(tensor, message=None, name=None):
  """Asserts that the given `tensor` is a scalar.

  This function raises `ValueError` unless it can be certain that the given
  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
  unknown.

  This is always checked statically, so this method returns nothing.

  Args:
    tensor: A `Tensor`.
    message: A string to prefix to the default message.
    name:  A name for this operation. Defaults to "assert_scalar"

  Raises:
    ValueError: If the tensor is not scalar (rank 0), or if its shape is
      unknown.
  """
  assert_scalar(tensor=tensor, message=message, name=name)


@tf_export(v1=['debugging.assert_scalar', 'assert_scalar'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_scalar')
def assert_scalar(tensor, name=None, message=None):
  """Asserts that the given `tensor` is a scalar (i.e. zero-dimensional).

  This function raises `ValueError` unless it can be certain that the given
  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
  unknown.

  Args:
    tensor: A `Tensor`.
    name:  A name for this operation. Defaults to "assert_scalar"
    message: A string to prefix to the default message.

  Returns:
    The input tensor (potentially converted to a `Tensor`).

  Raises:
    ValueError: If the tensor is not scalar (rank 0), or if its shape is
      unknown.
  """
  with ops.name_scope(name, 'assert_scalar', [tensor]) as name_scope:
    tensor = ops.convert_to_tensor(tensor, name=name_scope)
    shape = tensor.get_shape()
    message = _message_prefix(message)
    if shape.ndims != 0:
      if context.executing_eagerly():
        raise ValueError('%sExpected scalar shape, saw shape: %s.'
                         % (message, shape,))
      else:
        raise ValueError('%sExpected scalar shape for %s, saw shape: %s.'
                         % (message, tensor.name, shape))
    return tensor


def _message_prefix(message):
  if message:
    return '%s.  ' % message
  return ''


@tf_export('ensure_shape')
@dispatch.add_dispatch_support
def ensure_shape(x, shape, name=None):
  """Updates the shape of a tensor and checks at runtime that the shape holds.

  When executed, this operation asserts that the input tensor `x`'s shape
  is compatible with the `shape` argument.
  See `tf.TensorShape.is_compatible_with` for details.

  >>> x = tf.constant([[1, 2, 3],
  ...                  [4, 5, 6]])
  >>> x = tf.ensure_shape(x, [2, 3])

  Use `None` for unknown dimensions:

  >>> x = tf.ensure_shape(x, [None, 3])
  >>> x = tf.ensure_shape(x, [2, None])

  If the tensor's shape is not compatible with the `shape` argument, an error
  is raised:

  >>> x = tf.ensure_shape(x, [5])
  Traceback (most recent call last):
  ...
  tf.errors.InvalidArgumentError: Shape of tensor dummy_input [3] is not
    compatible with expected shape [5]. [Op:EnsureShape]

  During graph construction (typically tracing a `tf.function`),
  `tf.ensure_shape` updates the static-shape of the **result** tensor by
  merging the two shapes. See `tf.TensorShape.merge_with` for details.

  This is most useful when **you** know a shape that can't be determined
  statically by TensorFlow.

  The following trivial `tf.function` prints the input tensor's
  static-shape before and after `ensure_shape` is applied.

  >>> @tf.function
  ... def f(tensor):
  ...   print("Static-shape before:", tensor.shape)
  ...   tensor = tf.ensure_shape(tensor, [None, 3])
  ...   print("Static-shape after:", tensor.shape)
  ...   return tensor

  This lets you see the effect of `tf.ensure_shape` when the function is traced:
  >>> cf = f.get_concrete_function(tf.TensorSpec([None, None]))
  Static-shape before: (None, None)
  Static-shape after: (None, 3)

  >>> cf(tf.zeros([3, 3])) # Passes
  >>> cf(tf.constant([1, 2, 3])) # fails
  Traceback (most recent call last):
  ...
  InvalidArgumentError:  Shape of tensor x [3] is not compatible with expected shape [3,3].

  The above example raises `tf.errors.InvalidArgumentError`, because `x`'s
  shape, `(3,)`, is not compatible with the `shape` argument, `(None, 3)`

  Inside a `tf.function` or `v1.Graph` context it checks both the buildtime and
  runtime shapes. This is stricter than `tf.Tensor.set_shape` which only
  checks the buildtime shape.

  Note: This differs from `tf.Tensor.set_shape` in that it sets the static shape
  of the resulting tensor and enforces it at runtime, raising an error if the
  tensor's runtime shape is incompatible with the specified shape.
  `tf.Tensor.set_shape` sets the static shape of the tensor without enforcing it
  at runtime, which may result in inconsistencies between the statically-known
  shape of tensors and the runtime value of tensors.

  For example, of loading images of a known size:

  >>> @tf.function
  ... def decode_image(png):
  ...   image = tf.image.decode_png(png, channels=3)
  ...   # the `print` executes during tracing.
  ...   print("Initial shape: ", image.shape)
  ...   image = tf.ensure_shape(image,[28, 28, 3])
  ...   print("Final shape: ", image.shape)
  ...   return image

  When tracing a function, no ops are being executed, shapes may be unknown.
  See the [Concrete Functions Guide](https://www.tensorflow.org/guide/concrete_function)
  for details.

  >>> concrete_decode = decode_image.get_concrete_function(
  ...     tf.TensorSpec([], dtype=tf.string))
  Initial shape:  (None, None, 3)
  Final shape:  (28, 28, 3)

  >>> image = tf.random.uniform(maxval=255, shape=[28, 28, 3], dtype=tf.int32)
  >>> image = tf.cast(image,tf.uint8)
  >>> png = tf.image.encode_png(image)
  >>> image2 = concrete_decode(png)
  >>> print(image2.shape)
  (28, 28, 3)

  >>> image = tf.concat([image,image], axis=0)
  >>> print(image.shape)
  (56, 28, 3)
  >>> png = tf.image.encode_png(image)
  >>> image2 = concrete_decode(png)
  Traceback (most recent call last):
  ...
  tf.errors.InvalidArgumentError:  Shape of tensor DecodePng [56,28,3] is not
    compatible with expected shape [28,28,3].

  Caution: if you don't use the result of `tf.ensure_shape` the check may not
  run.

  >>> @tf.function
  ... def bad_decode_image(png):
  ...   image = tf.image.decode_png(png, channels=3)
  ...   # the `print` executes during tracing.
  ...   print("Initial shape: ", image.shape)
  ...   # BAD: forgot to use the returned tensor.
  ...   tf.ensure_shape(image,[28, 28, 3])
  ...   print("Final shape: ", image.shape)
  ...   return image

  >>> image = bad_decode_image(png)
  Initial shape:  (None, None, 3)
  Final shape:  (None, None, 3)
  >>> print(image.shape)
  (56, 28, 3)

  Args:
    x: A `Tensor`.
    shape: A `TensorShape` representing the shape of this tensor, a
      `TensorShapeProto`, a list, a tuple, or None.
    name: A name for this operation (optional). Defaults to "EnsureShape".

  Returns:
    A `Tensor`. Has the same type and contents as `x`.

  Raises:
    tf.errors.InvalidArgumentError: If `shape` is incompatible with the shape
    of `x`.
  """
  if not isinstance(shape, tensor_shape.TensorShape):
    shape = tensor_shape.TensorShape(shape)

  return array_ops.ensure_shape(x, shape, name=name)


@ops.RegisterGradient('EnsureShape')
def _ensure_shape_grad(op, grad):
  del op  # Unused.
  return grad
