# Copyright 2015 Google Inc. All Rights Reserved.
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
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope


__all__ = ['fully_connected', 'convolution2d', 'relu', 'relu6', 'linear']


def _apply_activation(y, activation_fn, output_collections):
  if activation_fn:
    y = activation_fn(y)
  ops.add_to_collections(list(output_collections or []) +
                         [ops.GraphKeys.ACTIVATIONS], y)
  return y


def _make_variable(name, shape, dtype, collections, initializer, regularizer):
  """Makes a variables, adds it to collections, and applies regularization."""

  # We have to make sure to add w to VARIABLES, otherwise we'll get nasty
  # surprises when trying to save or load.
  collections = set(list(collections or []) + [ops.GraphKeys.VARIABLES])

  w = variable_scope.get_variable(name,
                                  shape=shape,
                                  dtype=dtype,
                                  initializer=initializer,
                                  collections=collections)
  if regularizer:
    loss = regularizer(w)
    if loss:
      ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, loss)

  return w


# Convenience function for _make_variable for weights.
_weight_variable = functools.partial(_make_variable, 'weights')


# Convenience function for _make_variable for biases.
_bias_variable = functools.partial(_make_variable, 'bias')


def fully_connected(x,
                    num_output_units,
                    activation_fn=None,
                    weight_init=initializers.xavier_initializer(),
                    bias_init=standard_ops.constant_initializer(0.),
                    name=None,
                    weight_collections=(ops.GraphKeys.WEIGHTS,),
                    bias_collections=(ops.GraphKeys.BIASES,),
                    output_collections=(ops.GraphKeys.ACTIVATIONS,),
                    weight_regularizer=None,
                    bias_regularizer=None):
  """Adds the parameters for a fully connected layer and returns the output.

  A fully connected layer is generally defined as a matrix multiply:
  `y = f(w * x + b)` where `f` is given by `activation_fn`. If
  `activation_fn` is `None`, the result of `y = w * x + b` is
  returned.

  This op creates `w` and optionally `b`. Bias (`b`) can be disabled by setting
  `bias_init` to `None`.

  The variable creation is compatible with `tf.variable_scope` and so can be
  reused with `tf.variable_scope` or `tf.make_template`.

  Most of the details of variable creation can be controlled by specifying the
  initializers (`weight_init` and `bias_init`) and which in collections to place
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
    weight_regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`. Used for weights.
    bias_regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`. Used for biases.

  Returns:
    The output of the fully connected layer.
  """
  with variable_scope.variable_op_scope([x], name, 'fully_connected'):
    num_input_units = x.get_shape().dims[1].value
    dtype = x.dtype.base_dtype

    w = _weight_variable(shape=[num_input_units, num_output_units],
                         dtype=dtype,
                         initializer=weight_init,
                         collections=weight_collections,
                         regularizer=weight_regularizer)

    y = standard_ops.matmul(x, w)

    if bias_init is not None:
      b = _bias_variable(shape=[num_output_units],
                         dtype=dtype,
                         initializer=bias_init,
                         collections=bias_collections,
                         regularizer=bias_regularizer)

      y = nn.bias_add(y, b)

    return _apply_activation(y, activation_fn, output_collections)


def convolution2d(x,
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
                  weight_regularizer=None,
                  bias_regularizer=None):
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
      raise ValueError('kernel_size must be length 2: ' % kernel_size)
    if len(stride) != 2:
      raise ValueError('stride must be length 2: ' % kernel_size)

    stride = [1, stride[0], stride[1], 1]
    shape = [kernel_size[0], kernel_size[1], num_input_channels,
             num_output_channels]
    dtype = x.dtype.base_dtype

    w = _weight_variable(shape=shape,
                         dtype=dtype,
                         initializer=weight_init,
                         collections=weight_collections,
                         regularizer=weight_regularizer)

    y = nn.conv2d(x, w, stride, padding)

    if bias_init is not None:
      b = _bias_variable(shape=[num_output_channels],
                         dtype=dtype,
                         initializer=bias_init,
                         collections=bias_collections,
                         regularizer=bias_regularizer)

      y = nn.bias_add(y, b)

    return _apply_activation(y, activation_fn, output_collections)


# TODO(eiderm): Verify and fix autocomplete in colab (also relu6).
relu = functools.partial(fully_connected, activation_fn=nn.relu)


relu6 = functools.partial(fully_connected, activation_fn=nn.relu6)


# Simple alias for fully_connected which removes the activation_fn parameter.
linear = functools.partial(fully_connected, activation_fn=None)
