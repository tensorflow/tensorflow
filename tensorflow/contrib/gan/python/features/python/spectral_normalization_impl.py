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
"""Keras-like layers and utilities that implement Spectral Normalization.

Based on "Spectral Normalization for Generative Adversarial Networks" by Miyato,
et al in ICLR 2018. https://openreview.net/pdf?id=B1QRgziT-
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import numbers
import re

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import base_layer_utils as keras_base_layer_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging

__all__ = [
    'compute_spectral_norm', 'spectral_normalize', 'spectral_norm_regularizer',
    'spectral_normalization_custom_getter', 'keras_spectral_normalization'
]

# tf.bfloat16 should work, but tf.matmul converts those to tf.float32 which then
# can't directly be assigned back to the tf.bfloat16 variable.
_OK_DTYPES_FOR_SPECTRAL_NORM = (dtypes.float16, dtypes.float32, dtypes.float64)
_PERSISTED_U_VARIABLE_SUFFIX = 'spectral_norm_u'


def compute_spectral_norm(w_tensor, power_iteration_rounds=1, name=None):
  """Estimates the largest singular value in the weight tensor.

  Args:
    w_tensor: The weight matrix whose spectral norm should be computed.
    power_iteration_rounds: The number of iterations of the power method to
      perform. A higher number yeilds a better approximation.
    name: An optional scope name.

  Returns:
    The largest singular value (the spectral norm) of w.
  """
  with variable_scope.variable_scope(name, 'spectral_norm'):
    # The paper says to flatten convnet kernel weights from
    # (C_out, C_in, KH, KW) to (C_out, C_in * KH * KW). But TensorFlow's Conv2D
    # kernel weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = array_ops.reshape(w_tensor, (-1, w_tensor.get_shape()[-1]))

    # Persisted approximation of first left singular vector of matrix `w`.
    u_var = variable_scope.get_variable(
        _PERSISTED_U_VARIABLE_SUFFIX,
        shape=(w.shape[0], 1),
        dtype=w.dtype,
        initializer=init_ops.random_normal_initializer(),
        trainable=False)
    u = u_var

    # Use power iteration method to approximate spectral norm.
    for _ in range(power_iteration_rounds):
      # `v` approximates the first right singular vector of matrix `w`.
      v = nn.l2_normalize(math_ops.matmul(array_ops.transpose(w), u))
      u = nn.l2_normalize(math_ops.matmul(w, v))

    # Update persisted approximation.
    with ops.control_dependencies([u_var.assign(u, name='update_u')]):
      u = array_ops.identity(u)

    u = array_ops.stop_gradient(u)
    v = array_ops.stop_gradient(v)

    # Largest singular value of `w`.
    spectral_norm = math_ops.matmul(
        math_ops.matmul(array_ops.transpose(u), w), v)
    spectral_norm.shape.assert_is_fully_defined()
    spectral_norm.shape.assert_is_compatible_with([1, 1])

    return spectral_norm[0][0]


def spectral_normalize(w, power_iteration_rounds=1, name=None):
  """Normalizes a weight matrix by its spectral norm.

  Args:
    w: The weight matrix to be normalized.
    power_iteration_rounds: The number of iterations of the power method to
      perform. A higher number yeilds a better approximation.
    name: An optional scope name.

  Returns:
    A normalized weight matrix tensor.
  """
  with variable_scope.variable_scope(name, 'spectral_normalize'):
    w_normalized = w / compute_spectral_norm(
        w, power_iteration_rounds=power_iteration_rounds)
    return array_ops.reshape(w_normalized, w.get_shape())


def spectral_norm_regularizer(scale, power_iteration_rounds=1, scope=None):
  """Returns a functions that can be used to apply spectral norm regularization.

  Small spectral norms enforce a small Lipschitz constant, which is necessary
  for Wasserstein GANs.

  Args:
    scale: A scalar multiplier. 0.0 disables the regularizer.
    power_iteration_rounds: The number of iterations of the power method to
      perform. A higher number yeilds a better approximation.
    scope: An optional scope name.

  Returns:
    A function with the signature `sn(weights)` that applies spectral norm
    regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.0:
      raise ValueError(
          'Setting a scale less than 0 on a regularizer: %g' % scale)
    if scale == 0.0:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def sn(weights, name=None):
    """Applies spectral norm regularization to weights."""
    with ops.name_scope(scope, 'SpectralNormRegularizer', [weights]) as name:
      scale_t = ops.convert_to_tensor(
          scale, dtype=weights.dtype.base_dtype, name='scale')
      return math_ops.multiply(
          scale_t,
          compute_spectral_norm(
              weights, power_iteration_rounds=power_iteration_rounds),
          name=name)

  return sn


def _default_name_filter(name):
  """A filter function to identify common names of weight variables.

  Args:
    name: The variable name.

  Returns:
    Whether `name` is a standard name for a weight/kernel variables used in the
    Keras, tf.layers, tf.contrib.layers or tf.contrib.slim libraries.
  """
  match = re.match(r'(.*\/)?(depthwise_|pointwise_)?(weights|kernel)$', name)
  return match is not None


def spectral_normalization_custom_getter(name_filter=_default_name_filter,
                                         power_iteration_rounds=1):
  """Custom getter that performs Spectral Normalization on a weight tensor.

  Specifically it divides the weight tensor by its largest singular value. This
  is intended to stabilize GAN training, by making the discriminator satisfy a
  local 1-Lipschitz constraint.

  Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan].

  [sn-gan]: https://openreview.net/forum?id=B1QRgziT-

  To reproduce an SN-GAN, apply this custom_getter to every weight tensor of
  your discriminator. The last dimension of the weight tensor must be the number
  of output channels.

  Apply this to layers by supplying this as the `custom_getter` of a
  `tf.variable_scope`. For example:

    with tf.variable_scope('discriminator',
                           custom_getter=spectral_norm_getter()):
      net = discriminator_fn(net)

  IMPORTANT: Keras does not respect the custom_getter supplied by the
  VariableScope, so Keras users should use `keras_spectral_normalization`
  instead of (or in addition to) this approach.

  It is important to carefully select to which weights you want to apply
  Spectral Normalization. In general you want to normalize the kernels of
  convolution and dense layers, but you do not want to normalize biases. You
  also want to avoid normalizing batch normalization (and similar) variables,
  but in general such layers play poorly with Spectral Normalization, since the
  gamma can cancel out the normalization in other layers. By default we supply a
  filter that matches the kernel variable names of the dense and convolution
  layers of the tf.layers, tf.contrib.layers, tf.keras and tf.contrib.slim
  libraries. If you are using anything else you'll need a custom `name_filter`.

  This custom getter internally creates a variable used to compute the spectral
  norm by power iteration. It will update every time the variable is accessed,
  which means the normalized discriminator weights may change slightly whilst
  training the generator. Whilst unusual, this matches how the paper's authors
  implement it, and in general additional rounds of power iteration can't hurt.

  Args:
    name_filter: Optionally, a method that takes a Variable name as input and
      returns whether this Variable should be normalized.
    power_iteration_rounds: The number of iterations of the power method to
      perform per step. A higher number yeilds a better approximation of the
      true spectral norm.

  Returns:
    A custom getter function that applies Spectral Normalization to all
    Variables whose names match `name_filter`.

  Raises:
    ValueError: If name_filter is not callable.
  """
  if not callable(name_filter):
    raise ValueError('name_filter must be callable')

  def _internal_getter(getter, name, *args, **kwargs):
    """A custom getter function that applies Spectral Normalization.

    Args:
      getter: The true getter to call.
      name: Name of new/existing variable, in the same format as
        tf.get_variable.
      *args: Other positional arguments, in the same format as tf.get_variable.
      **kwargs: Keyword arguments, in the same format as tf.get_variable.

    Returns:
      The return value of `getter(name, *args, **kwargs)`, spectrally
      normalized.

    Raises:
      ValueError: If used incorrectly, or if `dtype` is not supported.
    """
    if not name_filter(name):
      return getter(name, *args, **kwargs)

    if name.endswith(_PERSISTED_U_VARIABLE_SUFFIX):
      raise ValueError(
          'Cannot apply Spectral Normalization to internal variables created '
          'for Spectral Normalization. Tried to normalized variable [%s]' %
          name)

    if kwargs['dtype'] not in _OK_DTYPES_FOR_SPECTRAL_NORM:
      raise ValueError('Disallowed data type {}'.format(kwargs['dtype']))

    # This layer's weight Variable/PartitionedVariable.
    w_tensor = getter(name, *args, **kwargs)

    if len(w_tensor.get_shape()) < 2:
      raise ValueError(
          'Spectral norm can only be applied to multi-dimensional tensors')

    return spectral_normalize(
        w_tensor,
        power_iteration_rounds=power_iteration_rounds,
        name=(name + '/spectral_normalize'))

  return _internal_getter


@contextlib.contextmanager
def keras_spectral_normalization(name_filter=_default_name_filter,
                                 power_iteration_rounds=1):
  """A context manager that enables Spectral Normalization for Keras.

  Keras doesn't respect the `custom_getter` in the VariableScope, so this is a
  bit of a hack to make things work.

  Usage:
    with keras_spectral_normalization():
      net = discriminator_fn(net)

  Args:
    name_filter: Optionally, a method that takes a Variable name as input and
      returns whether this Variable should be normalized.
    power_iteration_rounds: The number of iterations of the power method to
      perform per step. A higher number yeilds a better approximation of the
      true spectral norm.

  Yields:
    A context manager that wraps the standard Keras variable creation method
    with the `spectral_normalization_custom_getter`.
  """
  original_make_variable = keras_base_layer_utils.make_variable
  sn_getter = spectral_normalization_custom_getter(
      name_filter=name_filter, power_iteration_rounds=power_iteration_rounds)

  def make_variable_wrapper(name, *args, **kwargs):
    return sn_getter(original_make_variable, name, *args, **kwargs)

  keras_base_layer_utils.make_variable = make_variable_wrapper

  yield

  keras_base_layer_utils.make_variable = original_make_variable
