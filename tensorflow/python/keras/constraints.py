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
# pylint: disable=invalid-name
# pylint: disable=g-classes-have-attributes
"""Constraints: functions that impose constraints on weight values."""

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.tools.docs import doc_controls


class Constraint:
  """Base class for weight constraints.

  A `Constraint` instance works like a stateless function.
  Users who subclass this
  class should override the `__call__` method, which takes a single
  weight parameter and return a projected version of that parameter
  (e.g. normalized or clipped). Constraints can be used with various Keras
  layers via the `kernel_constraint` or `bias_constraint` arguments.

  Here's a simple example of a non-negative weight constraint:

  >>> class NonNegative(tf.keras.constraints.Constraint):
  ...
  ...  def __call__(self, w):
  ...    return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

  >>> weight = tf.constant((-1.0, 1.0))
  >>> NonNegative()(weight)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.,  1.], dtype=float32)>

  >>> tf.keras.layers.Dense(4, kernel_constraint=NonNegative())
  """

  def __call__(self, w):
    """Applies the constraint to the input weight variable.

    By default, the inputs weight variable is not modified.
    Users should override this method to implement their own projection
    function.

    Args:
      w: Input weight variable.

    Returns:
      Projected variable (by default, returns unmodified inputs).
    """
    return w

  def get_config(self):
    """Returns a Python dict of the object config.

    A constraint config is a Python dictionary (JSON-serializable) that can
    be used to reinstantiate the same object.

    Returns:
      Python dict containing the configuration of the constraint object.
    """
    return {}


class MaxNorm(Constraint):
  """MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Also available via the shortcut function `tf.keras.constraints.max_norm`.

  Args:
    max_value: the maximum norm value for the incoming weights.
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.

  """

  def __init__(self, max_value=2, axis=0):
    self.max_value = max_value
    self.axis = axis

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    norms = backend.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = backend.clip(norms, 0, self.max_value)
    return w * (desired / (backend.epsilon() + norms))

  @doc_controls.do_not_generate_docs
  def get_config(self):
    return {'max_value': self.max_value, 'axis': self.axis}


class NonNeg(Constraint):
  """Constrains the weights to be non-negative.

  Also available via the shortcut function `tf.keras.constraints.non_neg`.
  """

  def __call__(self, w):
    return w * math_ops.cast(math_ops.greater_equal(w, 0.), backend.floatx())


class UnitNorm(Constraint):
  """Constrains the weights incident to each hidden unit to have unit norm.

  Also available via the shortcut function `tf.keras.constraints.unit_norm`.

  Args:
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.
  """

  def __init__(self, axis=0):
    self.axis = axis

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    return w / (
        backend.epsilon() + backend.sqrt(
            math_ops.reduce_sum(
                math_ops.square(w), axis=self.axis, keepdims=True)))

  @doc_controls.do_not_generate_docs
  def get_config(self):
    return {'axis': self.axis}


class MinMaxNorm(Constraint):
  """MinMaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have the norm between a lower bound and an upper bound.

  Also available via the shortcut function `tf.keras.constraints.min_max_norm`.

  Args:
    min_value: the minimum norm for the incoming weights.
    max_value: the maximum norm for the incoming weights.
    rate: rate for enforcing the constraint: weights will be
      rescaled to yield
      `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
      Effectively, this means that rate=1.0 stands for strict
      enforcement of the constraint, while rate<1.0 means that
      weights will be rescaled at each step to slowly move
      towards a value inside the desired interval.
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.
  """

  def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    self.min_value = min_value
    self.max_value = max_value
    self.rate = rate
    self.axis = axis

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    norms = backend.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = (
        self.rate * backend.clip(norms, self.min_value, self.max_value) +
        (1 - self.rate) * norms)
    return w * (desired / (backend.epsilon() + norms))

  @doc_controls.do_not_generate_docs
  def get_config(self):
    return {
        'min_value': self.min_value,
        'max_value': self.max_value,
        'rate': self.rate,
        'axis': self.axis
    }


class RadialConstraint(Constraint):
  """Constrains `Conv2D` kernel weights to be the same for each radius.

  Also available via the shortcut function
  `tf.keras.constraints.radial_constraint`.

  For example, the desired output for the following 4-by-4 kernel:

  ```
      kernel = [[v_00, v_01, v_02, v_03],
                [v_10, v_11, v_12, v_13],
                [v_20, v_21, v_22, v_23],
                [v_30, v_31, v_32, v_33]]
  ```

  is this::

  ```
      kernel = [[v_11, v_11, v_11, v_11],
                [v_11, v_33, v_33, v_11],
                [v_11, v_33, v_33, v_11],
                [v_11, v_11, v_11, v_11]]
  ```

  This constraint can be applied to any `Conv2D` layer version, including
  `Conv2DTranspose` and `SeparableConv2D`, and with either `"channels_last"` or
  `"channels_first"` data format. The method assumes the weight tensor is of
  shape `(rows, cols, input_depth, output_depth)`.
  """

  @doc_controls.do_not_generate_docs
  def __call__(self, w):
    w_shape = w.shape
    if w_shape.rank is None or w_shape.rank != 4:
      raise ValueError(
          'The weight tensor must be of rank 4, but is of shape: %s' % w_shape)

    height, width, channels, kernels = w_shape
    w = backend.reshape(w, (height, width, channels * kernels))
    # TODO(cpeter): Switch map_fn for a faster tf.vectorized_map once
    # backend.switch is supported.
    w = backend.map_fn(
        self._kernel_constraint,
        backend.stack(array_ops_stack.unstack(w, axis=-1), axis=0))
    return backend.reshape(
        backend.stack(array_ops_stack.unstack(w, axis=0), axis=-1),
        (height, width, channels, kernels))

  def _kernel_constraint(self, kernel):
    """Radially constraints a kernel with shape (height, width, channels)."""
    padding = backend.constant([[1, 1], [1, 1]], dtype='int32')

    kernel_shape = backend.shape(kernel)[0]
    start = backend.cast(kernel_shape / 2, 'int32')

    kernel_new = backend.switch(
        backend.cast(math_ops.floormod(kernel_shape, 2), 'bool'),
        lambda: kernel[start - 1:start, start - 1:start],
        lambda: kernel[start - 1:start, start - 1:start] + backend.zeros(  # pylint: disable=g-long-lambda
            (2, 2), dtype=kernel.dtype))
    index = backend.switch(
        backend.cast(math_ops.floormod(kernel_shape, 2), 'bool'),
        lambda: backend.constant(0, dtype='int32'),
        lambda: backend.constant(1, dtype='int32'))
    while_condition = lambda index, *args: backend.less(index, start)

    def body_fn(i, array):
      return i + 1, array_ops.pad(
          array,
          padding,
          constant_values=kernel[start + i, start + i])

    _, kernel_new = while_loop.while_loop(
        while_condition,
        body_fn, [index, kernel_new],
        shape_invariants=[
            index.get_shape(),
            tensor_shape.TensorShape([None, None])
        ])
    return kernel_new


# Aliases.

max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm
radial_constraint = RadialConstraint

# Legacy aliases.
maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm


def serialize(constraint):
  return serialize_keras_object(constraint)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='constraint')


def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret constraint identifier: ' +
                     str(identifier))
