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


# pylint: disable=redefined-builtin
def fused_conv2d_bias_activation(conv_input,
                                 filter,
                                 bias,
                                 strides=None,
                                 padding=None,
                                 conv_input_scale=1.0,
                                 side_input_scale=0.0,
                                 side_input=None,
                                 activation_mode="Relu",
                                 data_format=None,
                                 filter_format=None,
                                 name=None):
  """Fused 2D conv, bias and activation with optional side input.

  Computes a fused 2-D convolution scaled by conv_input_scale,
  adds an optional side input scaled by side_input_scale, adds biases,
  and applies ReLU. As an equation:
  output = ReLU(conv_input_scale * Conv(conv_input, filter) +
                side_input_scale * side_input + bias)
  Note: In int8 mode, The ReLU will clip the output to the range [0..127].

  Args:
    conv_input: A `Tensor` of the format specified by `data_format`.
    filter: A `Tensor` whose format depends on `data_format`:
        if `data_format` is "NCHW_VECT_C", filter should be "OIHW_VECT_I"
        otherwise, it should be "HWIO" format.
    bias: A 1-D `Tensor` of type `float32`, and dimensions equal to the
        number of output channels.
    strides: A list of 4 `ints` specifying convolution strides.
        if `data_format` is "NCHW" or "NCHW_VECT_C", the order should be NCHW.
        if `data_format` is "NHWC", the order should be NHWC.
    padding: A `string` from: `"SAME", "VALID"`.
    conv_input_scale: A scalar `float32` that will be multiplied by conv_input.
        This is optional and defaults to 1. However it should be set to
        specify the quantization scale when `data_format` is "NCHW_VECT_C".
    side_input_scale: A scalar `float32` that will be multiplied by side_input.
        This is optional and defaults to 0.
    side_input: A `Tensor` of the format specified by `data_format`.
        This is useful for imlementing ResNet blocks.
    activation_mode: (optional) currently must be the default "Relu".
        Note that in qint8 mode, it also clips to 127, so acts like ReluX.
    data_format: Specifies the data format.
        Possible values are:
        "NHWC" float [batch, height, width, channels]
        "NCHW" float [batch, channels, height, width]
        "NCHW_VECT_C" qint8 [batch, channels / 4, height, width, channels % 4]
        Defaults to `"NHWC"`.
        Performance is worst for `"NHWC"` and best for `"NCHW_VECT_C"`.
    filter_format: Specifies the filter format.
        Possible values are:
        "HWIO" float [kernel_height, kernel_width, input_channels,
                      output_channels ]
        "OIHW" float [output_channels, input_channels, kernel_height,
                      kernel_width ]
        "OIHW_VECT_I" qint8 [ output_channels, input_channels / 4,
                              kernel_height, kernel_width, input_channels % 4 ]
        Defaults to `"HWIO"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the format specified by `data_format`.
  """
  if strides is None:
    strides = [1, 1, 1, 1]
  if side_input is None:
    side_input = []
  return gen_fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
      conv_input,
      filter,
      bias,
      side_input,
      conv_input_scale,
      side_input_scale,
      padding=padding,
      strides=strides,
      activation_mode=activation_mode,
      data_format=data_format,
      filter_format=filter_format,
      name=name)
