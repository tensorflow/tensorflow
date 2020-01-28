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
"""Contains testing utilities related to mixed precision."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


def create_identity_with_grad_check_fn(expected_gradient, expected_dtype=None):
  """Returns a function that asserts it's gradient has a certain value.

  This serves as a hook to assert intermediate gradients have a certain value.
  This returns an identity function. The identity's gradient function is also
  the identity function, except it asserts that the gradient equals
  `expected_gradient` and has dtype `expected_dtype`.

  Args:
    expected_gradient: The gradient function asserts that the gradient is this
      value.
    expected_dtype: The gradient function asserts the gradient has this dtype.

  Returns:
    An identity function whose gradient function asserts the gradient has a
    certain value.
  """
  @custom_gradient.custom_gradient
  def _identity_with_grad_check(x):
    """Function that asserts it's gradient has a certain value."""
    x = array_ops.identity(x)
    def grad(dx):
      if expected_dtype:
        assert dx.dtype == expected_dtype, (
            'dx.dtype should be %s but is: %s' % (expected_dtype, dx.dtype))
      expected_tensor = ops.convert_to_tensor(expected_gradient, dtype=dx.dtype,
                                              name='expected_gradient')
      assert_op = check_ops.assert_equal(dx, expected_tensor)
      with ops.control_dependencies([assert_op]):
        dx = array_ops.identity(dx)
      return dx
    return x, grad
  # Keras sometimes has trouble serializing Lambda layers with a decorated
  # function. So we define and return a non-decorated function.
  def identity_with_grad_check(x):
    return _identity_with_grad_check(x)
  return identity_with_grad_check


def create_identity_with_nan_gradients_fn(have_nan_gradients):
  """Returns a function that optionally has NaN gradients.

  This serves as a hook to introduce NaN gradients to a model. This returns an
  identity function. The identity's gradient function will check if the boolean
  tensor `have_nan_gradients` is True. If so, the gradient will be NaN.
  Otherwise, the gradient will also be the identity.

  Args:
    have_nan_gradients: A scalar boolean tensor. If True, gradients will be NaN.
      Otherwise, the gradient function is the identity function.

  Returns:
    An identity function whose gradient function will return NaNs, if
    `have_nan_gradients` is True.
  """
  @custom_gradient.custom_gradient
  def _identity_with_nan_gradients(x):
    """Function whose gradient is NaN iff `have_nan_gradients` is True."""
    x = array_ops.identity(x)
    def grad(dx):
      # We need this control dependency, because otherwise the NaN could be
      # produced before `dx`. This in turn could cause the final gradient to be
      # produced because `dx`, causing the loss scale to be updated before `dx`,
      # which can cause `tf.assert_equal`s to fail.
      with ops.control_dependencies([dx]):
        nan_scalar = constant_op.constant(float('NaN'), dtype=dx.dtype)
      return control_flow_ops.cond(
          have_nan_gradients,
          lambda: array_ops.fill(array_ops.shape(dx), nan_scalar),
          lambda: dx
      )
    return x, grad
  # Keras sometimes has trouble serializing Lambda layers with a decorated
  # function. So we define and return a non-decorated function.
  def identity_with_nan_gradients(x):
    return _identity_with_nan_gradients(x)
  return identity_with_nan_gradients


class AssertTypeLayer(base_layer.Layer):
  """A layer which asserts it's inputs are a certain type."""

  def __init__(self, assert_type=None, **kwargs):
    self._assert_type = (dtypes.as_dtype(assert_type).name if assert_type
                         else None)
    super(AssertTypeLayer, self).__init__(**kwargs)

  def assert_input_types(self, inputs):
    """Asserts `inputs` are of the correct type. Should be called in call()."""
    if self._assert_type:
      inputs_flattened = nest.flatten(inputs)
      for inp in inputs_flattened:
        assert inp.dtype.base_dtype == self._assert_type, (
            'Input tensor has type %s which does not match assert type %s' %
            (inp.dtype.name, self._assert_type))


class MultiplyLayer(AssertTypeLayer):
  """A layer which multiplies its input by a scalar variable."""

  def __init__(self,
               regularizer=None,
               use_operator=False,
               var_name='v',
               **kwargs):
    """Initializes the MultiplyLayer.

    Args:
      regularizer: The regularizer on the scalar variable.
      use_operator: If True, add using the * operator. If False, add using
        tf.multiply.
      var_name: The name of the variable. It can be useful to pass a name other
        than 'v', to test having the attribute name (self.v) being different
        from the variable name.
      **kwargs: Passed to AssertTypeLayer constructor.
    """
    self._regularizer = regularizer
    if isinstance(regularizer, dict):
      self._regularizer = regularizers.deserialize(regularizer,
                                                   custom_objects=globals())
    self._use_operator = use_operator
    self._var_name = var_name
    super(MultiplyLayer, self).__init__(**kwargs)

  def build(self, _):
    self.v = self.add_weight(
        self._var_name, (), initializer='ones', regularizer=self._regularizer)
    self.built = True

  def call(self, inputs):
    self.assert_input_types(inputs)
    assert inputs.dtype == self.v.dtype
    return self._multiply(inputs, self.v)

  def _multiply(self, x, y):
    if self._use_operator:
      return x * y
    else:
      return math_ops.multiply(x, y)

  def get_config(self):
    config = super(MultiplyLayer, self).get_config()
    config['regularizer'] = regularizers.serialize(self._regularizer)
    config['use_operator'] = self._use_operator
    config['var_name'] = self._var_name
    config['assert_type'] = self._assert_type
    return config


class IdentityRegularizer(regularizers.Regularizer):

  def __call__(self, x):
    assert x.dtype == dtypes.float32
    return array_ops.identity(x)

  def get_config(self):
    return {}
