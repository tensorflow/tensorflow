# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tensorflow op performing correlation cost operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.correlation_cost.ops import gen_correlation_cost_op
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import resource_loader

_correlation_cost_op_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_correlation_cost_op.so"))

# pylint: disable=redefined-builtin
def correlation_cost(input_a,
                     input_b,
                     kernel_size,
                     max_displacement,
                     stride_1,
                     stride_2,
                     pad,
                     data_format='NHWC',
                     name=None):
  """Correlation Cost Volume computation.

  Computes a cost volume using correlation for two inputs. For feature
  maps A, B with spatial dimensions w, h, c it computes

    output(a, b) = sum_{l in [-k,k]**2}  < I(a+l), J(b+l) >

  where the patches of size K=2d + 1 are centered in position a resp. b.

  The output shape is [B, C', H', W'], where

    r = max_displacement / stride_2;
    bd = max_displacement + (kernel_size - 1) / 2
    C' = (2 * r + 1) ** 2
    H' = H + 2 * (pad - bd) / stride_1
    W' = W + 2 * (pad - bd) / stride_1

  Note: When the data_format requests "NHWC", an additional explicit
    transpose operation is executed.

  Args:
    input_a: A `Tensor` of the format specified by `data_format`.
    input_b: A `Tensor` of the format specified by `data_format`.
    kernel_size: An integer specifying the height and width of the
        patch used to compute the per-patch costs.
    max_displacement: An integer specifying the maximum search radius
        for each position.
    stride_1: An integer specifying the stride length in the input.
    stride_2: An integer specifying the stride length in the patch.
    pad: An integer specifying the paddings in height and width.
    data_format: Specifies the data format.
        Possible values are:
        "NHWC" float [batch, height, width, channels]
        "NCHW" float [batch, channels, height, width]
        Defaults to `"NHWC"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the format specified by `data_format`.
  """

  with ops.name_scope(name, "correlation_cost"):
    op_call = gen_correlation_cost_op.correlation_cost
    ret = op_call(input_a, input_b,
                  kernel_size=kernel_size,
                  max_displacement=max_displacement,
                  stride_1=stride_1,
                  stride_2=stride_2,
                  pad=pad,
                  data_format=data_format)
    if data_format == 'NHWC':
      # this is easier to maintain without
      # specializing an additional cuda kernel
      return array_ops.transpose(ret, [0, 2, 3, 1])
    return ret


correlation_cost_grad = gen_correlation_cost_op.correlation_cost_grad

@ops.RegisterGradient("CorrelationCost")
def _correlation_cost_grad(op, grad_output):
  kernel_size = op.get_attr("kernel_size")
  max_displacement = op.get_attr("max_displacement")
  stride_1 = op.get_attr("stride_1")
  stride_2 = op.get_attr("stride_2")
  pad = op.get_attr("pad")
  data_format = op.get_attr("data_format")

  input_a = ops.convert_to_tensor(op.inputs[0], name="input_a")
  input_b = ops.convert_to_tensor(op.inputs[1], name="input_b")
  grad_output_tensor = ops.convert_to_tensor(grad_output, name="grad_output")

  op_call = gen_correlation_cost_op.correlation_cost_grad
  grads = op_call(input_a, input_b, grad_output_tensor,
                  kernel_size=kernel_size,
                  max_displacement=max_displacement,
                  stride_1=stride_1,
                  stride_2=stride_2,
                  pad=pad,
                  data_format=data_format)


  grad_input_a = ops.convert_to_tensor(grads[0], name="grad_input_a")
  grad_input_b = ops.convert_to_tensor(grads[1], name="grad_input_b")
  return [grad_input_a, grad_input_b]
