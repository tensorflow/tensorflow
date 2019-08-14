# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains AutoCastVariable, a variable which automatically casts itself."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import values as distribute_values
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.tracking import base as trackable


# TODO(reedwm) Make this subclass AutoCastVariable.
class AutoCastVariable(trackable.Trackable):
  """Variable that will cast itself to a different dtype in applicable contexts.

  This class wraps a floating-point tf.Variable. It emulates the variable
  interface and delegates to the wrapped variable, but it additionally will cast
  the wrapped variable under a `Graph._enable_variable_auto_cast(dtype)` context
  manager.

  For example:

  ```
  v = tf.Variable(1.0, dtype=tf.float32)
  v = AutoCastVariable(v)
  print(tf.identity(v).dtype)  # tf.float32
  with ops.get_default_graph()._enable_variable_auto_cast(tf.float16):
    print(tf.identity(v).dtype)  # tf.float16, as v will cast itself to float16
    print(v.dtype)  # tf.float16, as v.dtype also changes under the ctx manager.
  ```

  The purpose of this class is to allow Keras layers to create variables in
  float32, and automatically cast them to float16 or bfloat16 when the layer is
  called.
  """

  def __init__(self, variable):
    """Creates an AutoCastVariable instance.

    Args:
      variable: A floating-point resource variable to wrap.

    Raises:
      ValueError: If `variable` is not a floating-point resource variable
    """
    if not resource_variable_ops.is_resource_variable(variable):
      raise ValueError('variable must be of type tf.ResourceVariable, but got: '
                       '%s' % variable)
    if not variable.dtype.is_floating:
      raise ValueError('variable must be a floating point variable but has '
                       'type: %s' % variable.dtype.name)
    self._variable = variable

    # Delegate to the underlying variable for checkpointing.
    self._gather_saveables_for_checkpoint = (
        self._variable._gather_saveables_for_checkpoint)  # pylint: disable=protected-access

  @property
  def name(self):
    return self._variable.name

  def _should_cast(self):
    """Returns True if this variable should be casted when accessed."""
    g = ops.get_default_graph()
    # pylint:disable=protected-access
    return (g._auto_cast_variable_read_dtype is not None and
            self.true_dtype != g._auto_cast_variable_read_dtype)
    # pylint:enable=protected-access

  @property
  def dtype(self):
    """The dtype this variable will be casted to when read."""
    if self._should_cast():
      return ops.get_default_graph()._auto_cast_variable_read_dtype  # pylint:disable=protected-access
    else:
      return self._variable.dtype

  @property
  def true_dtype(self):
    """The dtype of the underlying variable, before any casts are done."""
    return self._variable.dtype

  def value(self):
    val = self._variable.value()
    if not self._should_cast():
      return val
    # We colocate_with(None) to ignore the existing device constraints, so that
    # the cast is always done on the variable's device
    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(val.device):
        return math_ops.cast(val, self.dtype)

  def read_value(self):
    val = self._variable.read_value()
    if not self._should_cast():
      return val
    return math_ops.cast(val, self.dtype)

  def sparse_read(self, indices, name=None):
    """Reads the value of this variable sparsely, using `gather`."""
    val = self._variable.sparse_read(indices, name=name)
    if not self._should_cast():
      return val
    return math_ops.cast(val, self.dtype)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    return self._variable.assign(
        value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    return self._variable.assign_add(
        delta, use_locking=use_locking, name=name, read_value=read_value)

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    return self._variable.assign_sub(
        delta, use_locking=use_locking, name=name, read_value=read_value)

  # TODO(reedwm): Support assigning variables with tf.compat.v1.assign(),
  # var.scatter_add, etc.

  def __getattr__(self, name):
    return getattr(self._variable, name)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts this variable to a tensor."""
    if not self._should_cast():
      return ops.internal_convert_to_tensor(self._variable, dtype, name,
                                            as_ref)
    # TODO(reedwm): Support as_ref?
    assert not as_ref
    if dtype is not None and not dtype.is_compatible_with(self.dtype):
      raise ValueError(
          'Incompatible type conversion requested to type {!r} for variable '
          'of type {!r}'.format(dtype.name, self.dtype.name))
    val = ops.internal_convert_to_tensor(self._variable,
                                         self._variable.dtype, name,
                                         as_ref=False)
    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(val.device):
        return math_ops.cast(val, self.dtype)

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def __repr__(self):
    if context.executing_eagerly() and not self._in_graph_mode:
      repr_str = ("<AutoCastVariable '{v.name}' shape={v.shape} "
                  'dtype={v.dtype.name} true_dtype={v.true_dtype.name}, '
                  'numpy={np_repr}>')
      return repr_str.format(
          v=self, np_repr=ops.numpy_text(self.read_value(), is_repr=True))
    else:
      repr_str = ("<AutoCastVariable '{v.name}' shape={v.shape} "
                  'dtype={v.dtype.name} true_dtype={v.true_dtype.name}>')
      return repr_str.format(v=self)

  # Operator overloads:
  # Note we only overload operators that support floating-point types, as
  # non-float variables cannot be wrapped with an AutoCastVariable.

  # pylint: disable=multiple-statements
  def __add__(self, o): return self.value() + o
  def __radd__(self, o): return o + self.value()
  def __sub__(self, o): return self.value() - o
  def __rsub__(self, o): return o - self.value()
  def __mul__(self, o): return self.value() * o
  def __rmul__(self, o): return o * self.value()
  def __truediv__(self, o): return self.value() / o
  def __rtruediv__(self, o): return o / self.value()
  def __floordiv__(self, o): return self.value() // o

  def __rfloordiv__(self, o): return o // self.value()
  def __mod__(self, o): return self.value() % o
  def __rmod__(self, o): return o % self.value()
  def __lt__(self, o): return self.value() < o
  def __le__(self, o): return self.value() <= o
  def __gt__(self, o): return self.value() > o
  def __ge__(self, o): return self.value() >= o
  def __getitem__(self, o): return self.value()[o]
  def __pow__(self, o, modulo=None): return pow(self.value(), o, modulo)
  def __rpow__(self, o): return pow(o, self.value())
  def __neg__(self): return -self.value()
  def __abs__(self): return abs(self.value())

  def __div__(self, o):
    try:
      return self.value().__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    try:
      return self.value().__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    try:
      return self.value().__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    try:
      return self.value().__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  # pylint: enable=multiple-statements

ops.register_tensor_conversion_function(
    AutoCastVariable, AutoCastVariable._dense_var_to_tensor)  # pylint:disable=protected-access
ops.register_dense_tensor_like_type(AutoCastVariable)


# We have DistributedVariable subclass to pass
# isinstance(..., DistributedVariable) checks when wrapping a
# DistributedVariable.
# TODO(reedwm): We should not wrap DistributedVariable, but instead have
# DistributedVariable wrap AutoCastVariable. Subclassing DistributedVariable is
# messy, because we do not fully implement the interface of DistributedVariable.
class AutoCastDistributedVariable(AutoCastVariable,
                                  distribute_values.DistributedVariable):
  """Version of AutoCastVariable that subclasses DistributedVariable."""

  def __init__(self, variable):
    if not isinstance(variable, distribute_values.DistributedValues):
      raise ValueError('variable must be of type DistributedValues, '
                       'but got: %s' % variable)
    super(AutoCastDistributedVariable, self).__init__(variable)

  def __repr__(self):
    return distribute_values.DistributedVariable.__repr__(self)
