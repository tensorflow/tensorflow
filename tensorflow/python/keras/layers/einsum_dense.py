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
"""Keras-based einsum dense layer."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.experimental.EinsumDense")
class EinsumDense(Layer):
  """A layer that uses tf.einsum as the backing computation.

  This layer can perform einsum calculations of arbitrary dimensionality.

  Arguments:
    equation: An equation describing the einsum to perform. This equation must
      be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
      `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum axis
      expression sequence.
    output_shape: The expected shape of the output tensor (excluding the batch
      dimension and any dimensions represented by ellipses). You can specify
      None for any dimension that is unknown or can be inferred from the input
      shape.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (that is, a "linear" activation: `a(x) = x`).
    bias_axes: A string containing the output dimension(s) to apply a bias to.
      Each character in the `bias_axes` string should correspond to a character
      in the output portion of the `equation` string.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Examples:

  **Biased dense layer with einsums**

  This example shows how to instantiate a standard Keras dense layer using
  einsum operations. This example is equivalent to
  `tf.keras.layers.Dense(64, use_bias=True)`.

  >>> layer = EinsumDense("ab,bc->ac", output_shape=64, bias_axes="c")
  >>> input_tensor = tf.keras.Input(shape=[32])
  >>> output_tensor = layer(input_tensor)
  >>> output_tensor
  <... shape=(None, 64) dtype=...>

  **Applying a dense layer to a sequence**

  This example shows how to instantiate a layer that applies the same dense
  operation to every element in a sequence. Here, the 'output_shape' has two
  values (since there are two non-batch dimensions in the output); the first
  dimension in the output_shape is `None`, because the sequence dimension `b`
  has an unknown shape.

  >>> layer = EinsumDense("abc,cd->abd",
  ...                     output_shape=(None, 64),
  ...                     bias_axes="d")
  >>> input_tensor = tf.keras.Input(shape=[32, 128])
  >>> output_tensor = layer(input_tensor)
  >>> output_tensor
  <... shape=(None, 32, 64) dtype=...>

  **Applying a dense layer to a sequence using ellipses**

  This example shows how to instantiate a layer that applies the same dense
  operation to every element in a sequence, but uses the ellipsis notation
  instead of specifying the batch and sequence dimensions.

  Because we are using ellipsis notation and have specified only one axis, the
  output_shape arg is a single value. When instantiated in this way, the layer
  can handle any number of sequence dimensions - including the case where no
  sequence dimension exists.

  >>> layer = EinsumDense("...x,xy->...y", output_shape=64, bias_axes="y")
  >>> input_tensor = tf.keras.Input(shape=[32, 128])
  >>> output_tensor = layer(input_tensor)
  >>> output_tensor
  <... shape=(None, 32, 64) dtype=...>
  """

  def __init__(self,
               equation,
               output_shape,
               activation=None,
               bias_axes=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(EinsumDense, self).__init__(**kwargs)
    self.equation = equation
    if isinstance(output_shape, int):
      self.partial_output_shape = [output_shape]
    else:
      self.partial_output_shape = list(output_shape)
    self.bias_axes = bias_axes
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    shape_data = _analyze_einsum_string(self.equation,
                                        self.bias_axes,
                                        input_shape,
                                        self.partial_output_shape)
    kernel_shape, bias_shape, self.full_output_shape = shape_data
    self.kernel = self.add_weight(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)

    if bias_shape is not None:
      self.bias = self.add_weight(
          "bias",
          shape=bias_shape,
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    super(EinsumDense, self).build(input_shape)

  def compute_output_shape(self, _):
    return tensor_shape.TensorShape(self.full_output_shape)

  def get_config(self):
    config = {
        "output_shape":
            self.partial_output_shape,
        "equation":
            self.equation,
        "activation":
            activations.serialize(self.activation),
        "bias_axes":
            self.bias_axes,
        "kernel_initializer":
            initializers.serialize(self.kernel_initializer),
        "bias_initializer":
            initializers.serialize(self.bias_initializer),
        "kernel_regularizer":
            regularizers.serialize(self.kernel_regularizer),
        "bias_regularizer":
            regularizers.serialize(self.bias_regularizer),
        "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
        "kernel_constraint":
            constraints.serialize(self.kernel_constraint),
        "bias_constraint":
            constraints.serialize(self.bias_constraint),
    }
    base_config = super(EinsumDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    ret = special_math_ops.einsum(self.equation, inputs, self.kernel)
    if self.bias is not None:
      ret += self.bias
    if self.activation is not None:
      ret = self.activation(ret)
    return ret


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
  """Analyzes an einsum string to determine the required weight shape."""

  dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

  # This is the case where no ellipses are present in the string.
  split_string = re.match("([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(split_string, bias_axes, input_shape,
                                 output_shape)

  # This is the case where ellipses are present on the left.
  split_string = re.match("0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(
        split_string, bias_axes, input_shape, output_shape, left_elided=True)

  # This is the case where ellipses are present on the right.
  split_string = re.match("([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(split_string, bias_axes, input_shape,
                                 output_shape)

  raise ValueError(
      "Invalid einsum equation '%s'. Equations must be in the form "
      "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...." % equation)


def _analyze_split_string(split_string,
                          bias_axes,
                          input_shape,
                          output_shape,
                          left_elided=False):
  """Analyze an pre-split einsum string to find the weight shape."""
  input_spec = split_string.group(1)
  weight_spec = split_string.group(2)
  output_spec = split_string.group(3)
  elided = len(input_shape) - len(input_spec)

  if isinstance(output_shape, int):
    output_shape = [output_shape]
  else:
    output_shape = list(output_shape)

  output_shape.insert(0, input_shape[0])

  if elided > 0 and left_elided:
    for i in range(1, elided):
      # We already inserted the 0th input dimension at dim 0, so we need to
      # start at location 1 here.
      output_shape.insert(1, input_shape[i])
  elif elided > 0 and not left_elided:
    for i in range(len(input_shape) - elided, len(input_shape)):
      output_shape.append(input_shape[i])

  if left_elided:
    # If we have beginning dimensions elided, we need to use negative indexing
    # to determine where in the input dimension our values are.
    input_dim_map = {
        dim: (i + elided) - len(input_shape) for i, dim in enumerate(input_spec)
    }
    # Because we've constructed the full output shape already, we don't need
    # to do negative indexing.
    output_dim_map = {dim: (i + elided) for i, dim in enumerate(output_spec)}
  else:
    input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
    output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

  for i, dim in enumerate(input_spec):
    input_shape_at_dim = input_shape[i]
    if dim in output_dim_map:
      output_shape_at_dim = output_shape[output_dim_map[dim]]
      if (output_shape_at_dim is not None and
          output_shape_at_dim != input_shape_at_dim):
        raise ValueError(
            "Input shape and output shape do not match at shared "
            "dimension '%s'. Input shape is %s, and output shape "
            "is %s." %
            (dim, input_shape_at_dim, output_shape[output_dim_map[dim]]))

  for dim in output_spec:
    if dim not in input_spec and dim not in weight_spec:
      raise ValueError("Dimension '%s' was specified in the output '%s' but "
                       "has no corresponding dim in the input spec '%s' or "
                       "weight spec '%s.'" % (dim, output_spec, input_spec,
                                              output_spec))

  weight_shape = []
  for dim in weight_spec:
    if dim in input_dim_map:
      weight_shape.append(input_shape[input_dim_map[dim]])
    elif dim in output_dim_map:
      weight_shape.append(output_shape[output_dim_map[dim]])
    else:
      raise ValueError("Weight dimension '%s' did not have a match in either "
                       "the input spec '%s' or the output spec '%s'. For this "
                       "layer, the weight must be fully specified." %
                       (dim, input_spec, output_spec))

  if bias_axes is not None:
    num_left_elided = elided if left_elided else 0
    idx_map = {
        char: output_shape[i + num_left_elided]
        for i, char in enumerate(output_spec)
    }

    for char in bias_axes:
      if char not in output_spec:
        raise ValueError("Bias dimension '%s' was requested, but is not a part "
                         "of the output specification '%s'" %
                         (char, output_spec))

    first_bias_location = min([output_spec.find(char) for char in bias_axes])
    bias_output_spec = output_spec[first_bias_location:]

    bias_shape = [
        idx_map[char] if char in bias_axes else 1 for char in bias_output_spec
    ]

    if not left_elided:
      for _ in range(elided):
        bias_shape.append(1)
  else:
    bias_shape = None

  return weight_shape, bias_shape, output_shape
