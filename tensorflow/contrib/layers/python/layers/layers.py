# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Higher level ops for building layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib.layers.python.layers import initializers

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope


# TODO(b/28426988): Remove legacy_* when all uses have migrated to new API.
__all__ = [
    'fully_connected', 'convolution2d', 'relu', 'relu6', 'linear',
    'legacy_fully_connected', 'legacy_convolution2d', 'legacy_relu',
    'legacy_relu6', 'legacy_linear'
]


def _apply_activation(y, activation_fn, output_collections):
  if activation_fn:
    y = activation_fn(y)
  ops.add_to_collections(list(output_collections or []) +
                         [ops.GraphKeys.ACTIVATIONS], y)
  return y


def legacy_fully_connected(x,
                           num_output_units,
                           activation_fn=None,
                           weight_init=initializers.xavier_initializer(),
                           bias_init=standard_ops.constant_initializer(0.),
                           name=None,
                           weight_collections=(ops.GraphKeys.WEIGHTS,),
                           bias_collections=(ops.GraphKeys.BIASES,),
                           output_collections=(ops.GraphKeys.ACTIVATIONS,),
                           trainable=True,
                           weight_regularizer=None,
                           bias_regularizer=None):
  # pylint: disable=anomalous-backslash-in-string
  r"""Adds the parameters for a fully connected layer and returns the output.

  A fully connected layer is generally defined as a matrix multiply:
  `y = f(w * x + b)` where `f` is given by `activation_fn`. If
  `activation_fn` is `None`, the result of `y = w * x + b` is
  returned.

  If `x` has shape [\\\(\\text{dim}_0, \\text{dim}_1, ..., \\text{dim}_n\\\)]
  with more than 2 dimensions (\\\(n > 1\\\)), then we repeat the matrix
  multiply along the first dimensions. The result r is a tensor of shape
  [\\\(\\text{dim}_0, ..., \\text{dim}_{n-1},\\\) `num_output_units`],
  where \\\( r_{i_0, ..., i_{n-1}, k} =
  \\sum_{0 \\leq j < \\text{dim}_n} x_{i_0, ... i_{n-1}, j} \cdot w_{j, k}\\\).
  This is accomplished by reshaping `x` to 2-D
  [\\\(\\text{dim}_0 \\cdot ... \\cdot \\text{dim}_{n-1}, \\text{dim}_n\\\)]
  before the matrix multiply and afterwards reshaping it to
  [\\\(\\text{dim}_0, ..., \\text{dim}_{n-1},\\\) `num_output_units`].

  This op creates `w` and optionally `b`. Bias (`b`) can be disabled by setting
  `bias_init` to `None`.

  The variable creation is compatible with `tf.variable_scope` and so can be
  reused with `tf.variable_scope` or `tf.make_template`.

  Most of the details of variable creation can be controlled by specifying the
  initializers (`weight_init` and `bias_init`) and in which collections to place
  the created variables (`weight_collections` and `bias_collections`; note that
  the variables are always added to the `VARIABLES` collection). The output of
  the layer can be placed in custom collections using `output_collections`.
  The collections arguments default to `WEIGHTS`, `BIASES` and `ACTIVATIONS`,
  respectively.

  A per layer regularization can be specified by setting `weight_regularizer`
  and `bias_regularizer`, which are applied to the weights and biases
  respectively, and whose output is added to the `REGULARIZATION_LOSSES`
  collection.

  Args:
    x: The input `Tensor`.
    num_output_units: The size of the output.
    activation_fn: A function that requires a single Tensor that is applied as a
      non-linearity. If None is used, do not apply any activation.
    weight_init: An optional weight initialization, defaults to
      `xavier_initializer`.
    bias_init: An initializer for the bias, defaults to 0. Set to `None` in
      order to disable bias.
    name: The name for this operation is used to name operations and to find
      variables. If specified it must be unique for this scope, otherwise a
      unique name starting with "fully_connected" will be created.  See
      `tf.variable_op_scope` for details.
    weight_collections: List of graph collections to which weights are added.
    bias_collections: List of graph collections to which biases are added.
    output_collections: List of graph collections to which outputs are added.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    weight_regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`. Used for weights.
    bias_regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`. Used for biases.

  Returns:
    The output of the fully connected layer.

  Raises:
    ValueError: if x has rank less than 2 or if its last dimension is not set.
  """
  # pylint: enable=anomalous-backslash-in-string
  with variable_scope.variable_op_scope([x], name, 'fully_connected'):
    dims = x.get_shape().dims
    if dims is None:
      raise ValueError('dims of x must be known but is None')
    if len(dims) < 2:
      raise ValueError('rank of x must be at least 2 not: %d' % len(dims))
    num_input_units = dims[-1].value
    if num_input_units is None:
      raise ValueError('last dimension of x must be known but is None')
    dtype = x.dtype.base_dtype

    weight_collections = set(list(weight_collections or []) +
                             [ops.GraphKeys.VARIABLES])
    w = variable_scope.get_variable('weights',
                                    shape=[num_input_units, num_output_units],
                                    dtype=dtype,
                                    initializer=weight_init,
                                    collections=weight_collections,
                                    regularizer=weight_regularizer,
                                    trainable=trainable)
    x_2_dim = x if len(dims) <= 2 else array_ops.reshape(x,
                                                         [-1, num_input_units])
    y = standard_ops.matmul(x_2_dim, w)

    if bias_init is not None:
      bias_collections = set(list(bias_collections or []) +
                             [ops.GraphKeys.VARIABLES])
      b = variable_scope.get_variable('bias',
                                      shape=[num_output_units],
                                      dtype=dtype,
                                      initializer=bias_init,
                                      collections=bias_collections,
                                      regularizer=bias_regularizer,
                                      trainable=trainable)

      y = nn.bias_add(y, b)

    if len(dims) > 2:
      out_shape = array_ops.unpack(array_ops.shape(x))
      out_shape[-1] = num_output_units

      y = array_ops.reshape(y, array_ops.pack(out_shape))

      static_shape = x.get_shape().as_list()
      static_shape[-1] = num_output_units
      y.set_shape(static_shape)

    return _apply_activation(y, activation_fn, output_collections)


def legacy_convolution2d(x,
                         num_output_channels,
                         kernel_size,
                         activation_fn=None,
                         stride=(1, 1),
                         padding='SAME',
                         weight_init=initializers.xavier_initializer_conv2d(),
                         bias_init=standard_ops.constant_initializer(0.),
                         name=None,
                         weight_collections=None,
                         bias_collections=None,
                         output_collections=None,
                         trainable=True,
                         weight_regularizer=None,
                         bias_regularizer=None):
  # pylint: disable=g-docstring-has-escape
  """Adds the parameters for a conv2d layer and returns the output.

  A neural network convolution layer is generally defined as:
  \\\\(y = f(conv2d(w, x) + b)\\\\) where **f** is given by `activation_fn`,
  **conv2d** is `tf.nn.conv2d` and `x` has shape
  `[batch, height, width, channels]`. The output of this op is of shape
  `[batch, out_height, out_width, num_output_channels]`, where `out_width` and
  `out_height` are determined by the `padding` argument. See `conv2D` for
  details.

  This op creates `w` and optionally `b` and adds various summaries that can be
  useful for visualizing learning or diagnosing training problems. Bias can be
  disabled by setting `bias_init` to `None`.

  The variable creation is compatible with `tf.variable_scope` and so can be
  reused with `tf.variable_scope` or `tf.make_template`.

  Most of the details of variable creation can be controlled by specifying the
  initializers (`weight_init` and `bias_init`) and which collections to place
  the created variables in (`weight_collections` and `bias_collections`).

  A per layer regularization can be specified by setting `weight_regularizer`.
  This is only applied to weights and not the bias.

  Args:
    x: A 4-D input `Tensor`.
    num_output_channels: The number of output channels (i.e. the size of the
      last dimension of the output).
    kernel_size: A length 2 `list` or `tuple` containing the kernel size.
    activation_fn: A function that requires a single Tensor that is applied as a
      non-linearity.
    stride: A length 2 `list` or `tuple` specifying the stride of the sliding
      window across the image.
    padding: A `string` from: "SAME", "VALID". The type of padding algorithm to
      use.
    weight_init: An optional initialization. If not specified, uses Xavier
      initialization (see `tf.learn.xavier_initializer`).
    bias_init: An initializer for the bias, defaults to 0. Set to`None` in order
      to disable bias.
    name: The name for this operation is used to name operations and to find
      variables. If specified it must be unique for this scope, otherwise a
      unique name starting with "convolution2d" will be created.  See
      `tf.variable_op_scope` for details.
    weight_collections: List of graph collections to which weights are added.
    bias_collections: List of graph collections to which biases are added.
    output_collections: List of graph collections to which outputs are added.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    weight_regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`. Used for weights.
    bias_regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`. Used for biases.

  Returns:
    The result of applying a 2-D convolutional layer.

  Raises:
    ValueError: If `kernel_size` or `stride` are not length 2.
  """
  with variable_scope.variable_op_scope([x], name, 'convolution2d'):
    num_input_channels = x.get_shape().dims[3].value

    if len(kernel_size) != 2:
      raise ValueError('kernel_size must be length 2: %d ' % kernel_size)
    if len(stride) != 2:
      raise ValueError('stride must be length 2: %d' % stride)

    stride = [1, stride[0], stride[1], 1]
    shape = [kernel_size[0], kernel_size[1], num_input_channels,
             num_output_channels]
    dtype = x.dtype.base_dtype

    weight_collections = set(list(weight_collections or []) +
                             [ops.GraphKeys.VARIABLES])
    w = variable_scope.get_variable('weights',
                                    shape=shape,
                                    dtype=dtype,
                                    initializer=weight_init,
                                    collections=weight_collections,
                                    regularizer=weight_regularizer,
                                    trainable=trainable)

    y = nn.conv2d(x, w, stride, padding)

    if bias_init is not None:
      bias_collections = set(list(bias_collections or []) +
                             [ops.GraphKeys.VARIABLES])
      b = variable_scope.get_variable('bias',
                                      shape=[num_output_channels],
                                      dtype=dtype,
                                      initializer=bias_init,
                                      collections=bias_collections,
                                      regularizer=bias_regularizer,
                                      trainable=trainable)

      y = nn.bias_add(y, b)

    return _apply_activation(y, activation_fn, output_collections)


# TODO(eiderm): Verify and fix autocomplete in colab (also relu6).
legacy_relu = functools.partial(legacy_fully_connected, activation_fn=nn.relu)


legacy_relu6 = functools.partial(legacy_fully_connected, activation_fn=nn.relu6)


# Simple alias for fully_connected which removes the activation_fn parameter.
legacy_linear = functools.partial(legacy_fully_connected, activation_fn=None)


# TODO(b/28426988): Replace with fns migrated from slim.
convolution2d = legacy_convolution2d
fully_connected = legacy_fully_connected
linear = legacy_linear
relu = legacy_relu
relu6 = legacy_relu6
