"""TensorFlow ops for Convolution NNs."""
#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.learn.python.learn.ops.batch_norm_ops import batch_normalize


def conv2d(tensor_in, n_filters, filter_shape, strides=None, padding='SAME',
           bias=True, activation=None, batch_norm=False):
    """Creates 2D convolutional subgraph with bank of filters.

    Uses tf.nn.conv2d under the hood.
    Creates a filter bank:
      [filter_shape[0], filter_shape[1], tensor_in[3], n_filters]
    and applies it to the input tensor.

    Args:
        tensor_in: input Tensor, 4D shape:
                   [batch, in_height, in_width, in_depth].
        n_filters: number of filters in the bank.
        filter_shape: Shape of filters, a list of ints, 1-D of length 2.
        strides: A list of ints, 1-D of length 4. The stride of the sliding
                 window for each dimension of input.
        padding: A string: 'SAME' or 'VALID'. The type of padding algorthim to
                 use.
        bias: Boolean, if to add bias.
        activation: Activation Op, optional. If provided applied on the output.
        batch_norm: Whether to apply batch normalization.

    Returns:
        A Tensor with resulting convolution.
    """
    with vs.variable_scope('convolution'):
        if strides is None:
            strides = [1, 1, 1, 1]
        input_shape = tensor_in.get_shape()
        filter_shape = list(filter_shape) + [input_shape[3], n_filters]
        filters = vs.get_variable('filters', filter_shape, dtypes.float32)
        output = nn.conv2d(tensor_in, filters, strides, padding)
        if bias:
            bias_var = vs.get_variable('bias', [1, 1, 1, n_filters],
                                       dtypes.float32)
            output = output + bias_var
        if batch_norm:
            output = batch_normalize(output, convnet=True)
        if activation:
            output = activation(output)
        return output

