# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tensorflow op performing fused conv2d bias_add and relu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.fused_conv.ops import gen_fused_conv2d_bias_activation_op
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_fused_conv2d_bias_activation_op_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_fused_conv2d_bias_activation_op.so"))


def fused_conv2d_bias_activation(input_tensor,
                                 filter_tensor,
                                 bias,
                                 strides,
                                 padding,
                                 activation_mode,
                                 data_format=None,
                                 name=None):
  """Computes a fused 2-D convolution, adds bias, and applies relu.

      input_tensor: A 4-D tensor. The dimension order is interpreted
      according to the value of `data_format`, see below for details.
      filter_tensor: A 4-D tensor of shape
          `[filter_height, filter_width, in_channels, out_channels]`
      bias: 1-D with size of the `out_channels` dimension in filter.
      output: A 4-D tensor. The dimension order is determined by the value of
          `data_format`, see below for details.
      T: The data type for the elements of input, filter, bias, and output
      Tensors.
      strides: 1-D tensor of length 4.  The stride of the sliding window for
      each
          dimension of `input`. The dimension order is determined by the value
          of
          `data_format`, see below for details.
      padding: The type of padding algorithm to use.
      data_format: Specify the data format of the input and output data. With
      the
          default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
          Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
      activation_mode: Specify the activation function to apply to the output
      tensor
          of bias add. Currently only supports "Relu".

  Args:
    input_tensor: A `Tensor`. Must be one of the following types: `float32`.
    filter_tensor: A `Tensor`. Must have the same type as `input`.
    bias: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    activation_mode: A `string` from: `"Sigmoid", "Relu", "Relu6", "ReluX",
      "Tanh", "BandPass"`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to
      `"NHWC"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  return gen_fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
      input=input_tensor,
      filter=filter_tensor,
      bias=bias,
      strides=strides,
      padding=padding,
      activation_mode=activation_mode,
      data_format=data_format,
      name=name)
