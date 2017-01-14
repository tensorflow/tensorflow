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

"""TensorFlow ops for Convolution NNs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.layers import convolution2d

from tensorflow.python.ops import init_ops
from tensorflow.python.platform import tf_logging as logging


@deprecated(date="2016-08-15",
            instructions="Please use tf.contrib.layers.conv2d instead.")
def conv2d(tensor_in,
           n_filters,
           filter_shape,
           strides=None,
           padding="SAME",
           bias=True,
           activation=None,
           batch_norm=False):
  """Creates 2D convolutional subgraph with bank of filters.

  This is deprecated. Please use contrib.layers.convolution2d.

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
    padding: A string: 'SAME' or 'VALID'. The type of padding algorthim to use.
      See the [comment here]
      (https://www.tensorflow.org/api_docs/python/nn.html#convolution)
    bias: Boolean, if to add bias.
    activation: Activation Op, optional. If provided applied on the output.
    batch_norm: Whether to apply batch normalization.

  Returns:
    A Tensor with resulting convolution.
  """
  if batch_norm:
    logging.warn("batch_norm will not work with learn.ops.conv2d, "
                 "use tf.contrib.layers.conv2d.")

  if bias:
    bias_init = init_ops.zeros_initializer
  if strides is None:
    strides = [1, 1]
  else:
    strides = strides[1:3]  # only take height and width
    logging.warn("strides may not be passed correctly. Please instead "
                 "use and see documentation for contrib.layers.convolution2d.")
  return convolution2d(
      tensor_in,
      num_outputs=n_filters,
      kernel_size=list(filter_shape),
      stride=strides,
      padding=padding,
      biases_initializer=bias_init,
      activation_fn=activation)
