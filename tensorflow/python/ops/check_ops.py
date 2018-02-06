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
"""Asserts and Boolean Checks.

See the @{$python/check_ops} guide.

@@assert_negative
@@assert_positive
@@assert_non_negative
@@assert_non_positive
@@assert_equal
@@assert_none_equal
@@assert_near
@@assert_less
@@assert_less_equal
@@assert_greater
@@assert_greater_equal
@@assert_rank
@@assert_rank_at_least
@@assert_rank_in
@@assert_type
@@assert_integer
@@assert_proper_iterable
@@assert_same_float_dtype
@@assert_scalar
@@is_non_decreasing
@@is_numeric_tensor
@@is_strictly_increasing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

NUMERIC_TYPES = frozenset(
    [dtypes.float32, dtypes.float64, dtypes.int8, dtypes.int16, dtypes.int32,
     dtypes.int64, dtypes.uint8, dtypes.qint8, dtypes.qint32, dtypes.quint8,
     dtypes.complex64])

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
    'is_non_decreasing',
    'is_numeric_tensor',
    'is_strictly_increasing',
]


def _maybe_constant_value_string(t):
  if not isinstance(t, ops.Tensor):
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


@tf_export('assert_proper_iterable')
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
      (ops.Tensor, sparse_tensor.SparseTensor, np.ndarray)
      + compat.bytes_or_text_types
  )
  if isinstance(values, unintentional_iterables):
    raise TypeError(
        'Expected argument "values" to be a "proper" iterable.  Found: %s' %
        type(values))

  if not hasattr(values, '__iter__'):
    raise TypeError(
        'Expected argument "values" to be iterable.  Found: %s' % type(values))


@tf_export('assert_negative')
def assert_negative(x, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x < 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_negative(x)]):
    output = tf.reduce_sum(x)
  ```

  Negative means, for every element `x[i]` of `x`, we have `x[i] < 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all negative.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_negative', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.in_eager_mode():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x < 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(x, zero, data=data, summarize=summarize)


@tf_export('assert_positive')
def assert_positive(x, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x > 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_positive(x)]):
    output = tf.reduce_sum(x)
  ```

  Positive means, for every element `x[i]` of `x`, we have `x[i] > 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_positive".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all positive.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_positive', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.in_eager_mode():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message, 'Condition x > 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(zero, x, data=data, summarize=summarize)


@tf_export('assert_non_negative')
def assert_non_negative(x, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x >= 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_non_negative(x)]):
    output = tf.reduce_sum(x)
  ```

  Non-negative means, for every element `x[i]` of `x`, we have `x[i] >= 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_non_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-negative.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_non_negative', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.in_eager_mode():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x >= 0 did not hold element-wise:',
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(zero, x, data=data, summarize=summarize)


@tf_export('assert_non_positive')
def assert_non_positive(x, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x <= 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_non_positive(x)]):
    output = tf.reduce_sum(x)
  ```

  Non-positive means, for every element `x[i]` of `x`, we have `x[i] <= 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_non_positive".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-positive.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_non_positive', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      if context.in_eager_mode():
        name = _shape_and_dtype_str(x)
      else:
        name = x.name
      data = [
          message,
          'Condition x <= 0 did not hold element-wise:'
          'x (%s) = ' % name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(x, zero, data=data, summarize=summarize)


@tf_export('assert_equal')
def assert_equal(x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x == y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_equal(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] == y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_equal".

  Returns:
    Op that raises `InvalidArgumentError` if `x == y` is False.
    @compatibility{eager} returns None

  Raises:
    InvalidArgumentError if the check can be performed immediately and
    `x == y` is False. The check can be performed immediately during
    eager execution or if `x` and `y` are statically known.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_equal', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')

    if context.in_eager_mode():
      eq = math_ops.equal(x, y)
      condition = math_ops.reduce_all(eq)
      if not condition:
        # Prepare a message with first elements of x and y.
        summary_msg = ''
        # Default to printing 3 elements like control_flow_ops.Assert (used
        # by graph mode) does.
        summarize = 3 if summarize is None else summarize
        if summarize:
          # reshape((-1,)) is the fastest way to get a flat array view.
          x_np = x.numpy().reshape((-1,))
          y_np = y.numpy().reshape((-1,))
          x_sum = min(x_np.size, summarize)
          y_sum = min(y_np.size, summarize)
          summary_msg = ('First %d elements of x:\n%s\n'
                         'First %d elements of y:\n%s\n' %
                         (x_sum, x_np[:x_sum],
                          y_sum, y_np[:y_sum]))

        # Get the values that actually differed and their indices.
        mask = math_ops.logical_not(eq)
        indices = array_ops.where(mask)
        indices_np = indices.numpy()
        x_vals = array_ops.boolean_mask(x, mask)
        y_vals = array_ops.boolean_mask(y, mask)
        summarize = min(summarize, indices_np.shape[0])

        raise errors.InvalidArgumentError(
            node_def=None, op=None,
            message=('%s\nCondition x == y did not hold.\n'
                     'Indices of first %s different values:\n%s\n'
                     'Corresponding x values:\n%s\n'
                     'Corresponding y values:\n%s\n'
                     '%s'
                     %
                     (message or '',
                      summarize, indices_np[:summarize],
                      x_vals.numpy().reshape((-1,))[:summarize],
                      y_vals.numpy().reshape((-1,))[:summarize],
                      summary_msg)))
      return

    if data is None:
      data = [
          message,
          'Condition x == y did not hold element-wise:',
          'x (%s) = ' % x.name, x,
          'y (%s) = ' % y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.equal(x, y))
    x_static = tensor_util.constant_value(x)
    y_static = tensor_util.constant_value(y)
    if x_static is not None and y_static is not None:
      condition_static = (x_static == y_static).all()
      _assert_static(condition_static, data)
    return control_flow_ops.Assert(condition, data, summarize=summarize)


@tf_export('assert_none_equal')
def assert_none_equal(
    x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x != y` holds for all elements.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_none_equal(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] != y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_none_equal".

  Returns:
    Op that raises `InvalidArgumentError` if `x != y` is ever False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_none_equal', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if context.in_eager_mode():
      x_name = _shape_and_dtype_str(x)
      y_name = _shape_and_dtype_str(y)
    else:
      x_name = x.name
      y_name = y.name

    if data is None:
      data = [
          message,
          'Condition x != y did not hold for every single element:',
          'x (%s) = ' % x_name, x,
          'y (%s) = ' % y_name, y
      ]
    condition = math_ops.reduce_all(math_ops.not_equal(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


@tf_export('assert_near')
def assert_near(
    x, y, rtol=None, atol=None, data=None, summarize=None, message=None,
    name=None):
  """Assert the condition `x` and `y` are close element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_near(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have

  ```tf.abs(x[i] - y[i]) <= atol + rtol * tf.abs(y[i])```.

  If both `x` and `y` are empty, this is trivially satisfied.

  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest
  representable positive number such that `1 + eps != eps`.  This is about
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
  Similar to `numpy.assert_allclose`, except tolerance depends on data type.
  This is due to the fact that `TensorFlow` is often used with `32bit`, `64bit`,
  and even `16bit` data.
  @end_compatibility
  """
  message = message or ''
  with ops.name_scope(name, 'assert_near', [x, y, rtol, atol, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y', dtype=x.dtype)

    eps = np.finfo(x.dtype.as_numpy_dtype).eps
    rtol = 10 * eps if rtol is None else rtol
    atol = 10 * eps if atol is None else atol

    rtol = ops.convert_to_tensor(rtol, name='rtol', dtype=x.dtype)
    atol = ops.convert_to_tensor(atol, name='atol', dtype=x.dtype)

    if context.in_eager_mode():
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
    return control_flow_ops.Assert(condition, data, summarize=summarize)


@tf_export('assert_less')
def assert_less(x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x < y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_less(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] < y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_less".

  Returns:
    Op that raises `InvalidArgumentError` if `x < y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_less', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if context.in_eager_mode():
      x_name = _shape_and_dtype_str(x)
      y_name = _shape_and_dtype_str(y)
    else:
      x_name = x.name
      y_name = y.name

    if data is None:
      data = [
          message,
          'Condition x < y did not hold element-wise:',
          'x (%s) = ' % x_name, x, 'y (%s) = ' % y_name, y
      ]
    condition = math_ops.reduce_all(math_ops.less(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


@tf_export('assert_less_equal')
def assert_less_equal(x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x <= y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_less_equal(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] <= y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_less_equal"

  Returns:
    Op that raises `InvalidArgumentError` if `x <= y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_less_equal', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if context.in_eager_mode():
      x_name = _shape_and_dtype_str(x)
      y_name = _shape_and_dtype_str(y)
    else:
      x_name = x.name
      y_name = y.name

    if data is None:
      data = [
          message,
          'Condition x <= y did not hold element-wise:'
          'x (%s) = ' % x_name, x, 'y (%s) = ' % y_name, y
      ]
    condition = math_ops.reduce_all(math_ops.less_equal(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


@tf_export('assert_greater')
def assert_greater(x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x > y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_greater(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] > y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_greater".

  Returns:
    Op that raises `InvalidArgumentError` if `x > y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_greater', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if context.in_eager_mode():
      x_name = _shape_and_dtype_str(x)
      y_name = _shape_and_dtype_str(y)
    else:
      x_name = x.name
      y_name = y.name

    if data is None:
      data = [
          message,
          'Condition x > y did not hold element-wise:'
          'x (%s) = ' % x_name, x, 'y (%s) = ' % y_name, y
      ]
    condition = math_ops.reduce_all(math_ops.greater(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


@tf_export('assert_greater_equal')
def assert_greater_equal(x, y, data=None, summarize=None, message=None,
                         name=None):
  """Assert the condition `x >= y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_greater_equal(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] >= y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to
      "assert_greater_equal"

  Returns:
    Op that raises `InvalidArgumentError` if `x >= y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_greater_equal', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if context.in_eager_mode():
      x_name = _shape_and_dtype_str(x)
      y_name = _shape_and_dtype_str(y)
    else:
      x_name = x.name
      y_name = y.name

    if data is None:
      data = [
          message,
          'Condition x >= y did not hold element-wise:'
          'x (%s) = ' % x_name, x, 'y (%s) = ' % y_name, y
      ]
    condition = math_ops.reduce_all(math_ops.greater_equal(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


def _assert_rank_condition(
    x, rank, static_condition, dynamic_condition, data, summarize):
  """Assert `x` has a rank that satisfies a given condition.

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    static_condition:   A python function that takes `[actual_rank, given_rank]`
      and returns `True` if the condition is satisfied, `False` otherwise.
    dynamic_condition:  An `op` that takes [actual_rank, given_rank]
      and return `True` if the condition is satisfied, `False` otherwise.
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

  return control_flow_ops.Assert(condition, data, summarize=summarize)


@tf_export('assert_rank')
def assert_rank(x, rank, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar integer `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
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
    x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')
    message = message or ''

    static_condition = lambda actual_rank, given_rank: actual_rank == given_rank
    dynamic_condition = math_ops.equal

    if context.in_eager_mode():
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
            '%s.  Tensor %s must have rank %d.  Received rank %d, shape %s' %
            (message, name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise

  return assert_op


@tf_export('assert_rank_at_least')
def assert_rank_at_least(
    x, rank, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank` or higher.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank_at_least(x, 2)]):
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
    message = message or ''

    static_condition = lambda actual_rank, given_rank: actual_rank >= given_rank
    dynamic_condition = math_ops.greater_equal

    if context.in_eager_mode():
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
            '%s.  Tensor %s must have rank at least %d.  Received rank %d, '
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

  return control_flow_ops.Assert(condition, data, summarize=summarize)


@tf_export('assert_rank_in')
def assert_rank_in(
    x, ranks, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank in `ranks`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank_in(x, (2, 4))]):
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
    x = ops.convert_to_tensor(x, name='x')
    ranks = tuple([ops.convert_to_tensor(rank, name='rank') for rank in ranks])
    message = message or ''

    if context.in_eager_mode():
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
            '%s.  Tensor %s must have rank in %s.  Received rank %d, '
            'shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise

  return assert_op


@tf_export('assert_integer')
def assert_integer(x, message=None, name=None):
  """Assert that `x` is of integer dtype.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_integer(x)]):
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
  message = message or ''
  with ops.name_scope(name, 'assert_integer', [x]):
    x = ops.convert_to_tensor(x, name='x')
    if not x.dtype.is_integer:
      if context.in_eager_mode():
        name = 'tensor'
      else:
        name = x.name
      err_msg = (
          '%s  Expected "x" to be integer type.  Found: %s of dtype %s'
          % (message, name, x.dtype))
      raise TypeError(err_msg)

    return control_flow_ops.no_op('statically_determined_was_integer')


@tf_export('assert_type')
def assert_type(tensor, tf_type, message=None, name=None):
  """Statically asserts that the given `Tensor` is of the specified type.

  Args:
    tensor: A tensorflow `Tensor`.
    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,
      etc).
    message: A string to prefix to the default message.
    name:  A name to give this `Op`.  Defaults to "assert_type"

  Raises:
    TypeError: If the tensors data type doesn't match `tf_type`.

  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_type', [tensor]):
    tensor = ops.convert_to_tensor(tensor, name='tensor')
    if tensor.dtype != tf_type:
      if context.in_graph_mode():
        raise TypeError(
            '%s  %s must be of type %s' % (message, tensor.name, tf_type))
      else:
        raise TypeError(
            '%s tensor must be of type %s' % (message, tf_type))

    return control_flow_ops.no_op('statically_determined_correct_type')


# pylint: disable=line-too-long
def _get_diff_for_monotonic_comparison(x):
  """Gets the difference x[1:] - x[:-1]."""
  x = array_ops.reshape(x, [-1])
  if not is_numeric_tensor(x):
    raise TypeError('Expected x to be numeric, instead found: %s' % x)

  # If x has less than 2 elements, there is nothing to compare.  So return [].
  is_shorter_than_two = math_ops.less(array_ops.size(x), 2)
  short_result = lambda: ops.convert_to_tensor([], dtype=x.dtype)

  # With 2 or more elements, return x[1:] - x[:-1]
  s_len = array_ops.shape(x) - 1
  diff = lambda: array_ops.strided_slice(x, [1], [1] + s_len)- array_ops.strided_slice(x, [0], s_len)
  return control_flow_ops.cond(is_shorter_than_two, short_result, diff)


@tf_export('is_numeric_tensor')
def is_numeric_tensor(tensor):
  return isinstance(tensor, ops.Tensor) and tensor.dtype in NUMERIC_TYPES


@tf_export('is_non_decreasing')
def is_non_decreasing(x, name=None):
  """Returns `True` if `x` is non-decreasing.

  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
  If `x` has less than two elements, it is trivially non-decreasing.

  See also:  `is_strictly_increasing`

  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).  Defaults to "is_non_decreasing"

  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.

  Raises:
    TypeError: if `x` is not a numeric tensor.
  """
  with ops.name_scope(name, 'is_non_decreasing', [x]):
    diff = _get_diff_for_monotonic_comparison(x)
    # When len(x) = 1, diff = [], less_equal = [], and reduce_all([]) = True.
    zero = ops.convert_to_tensor(0, dtype=diff.dtype)
    return math_ops.reduce_all(math_ops.less_equal(zero, diff))


@tf_export('is_strictly_increasing')
def is_strictly_increasing(x, name=None):
  """Returns `True` if `x` is strictly increasing.

  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.
  If `x` has less than two elements, it is trivially strictly increasing.

  See also:  `is_non_decreasing`

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
    diff = _get_diff_for_monotonic_comparison(x)
    # When len(x) = 1, diff = [], less = [], and reduce_all([]) = True.
    zero = ops.convert_to_tensor(0, dtype=diff.dtype)
    return math_ops.reduce_all(math_ops.less(zero, diff))


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
  return expected_type


@tf_export('assert_same_float_dtype')
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


@tf_export('assert_scalar')
def assert_scalar(tensor, name=None):
  with ops.name_scope(name, 'assert_scalar', [tensor]) as name_scope:
    tensor = ops.convert_to_tensor(tensor, name=name_scope)
    shape = tensor.get_shape()
    if shape.ndims != 0:
      if context.in_eager_mode():
        raise ValueError('Expected scalar shape, saw shape: %s.'
                         % (shape,))
      else:
        raise ValueError('Expected scalar shape for %s, saw shape: %s.'
                         % (tensor.name, shape))
    return tensor
