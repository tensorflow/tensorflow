# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Defines all the new composite ops used in the mnist example."""

# pylint: disable=g-direct-tensorflow-import
# pylint: disable=missing-function-docstring

import os
import sys
from absl import app

import tensorflow as tf

from tensorflow.compiler.mlir.tfr.python import composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.compiler.mlir.tfr.python.tfr_gen import tfr_gen_from_module
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import flags

Composite = composite.Composite
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output', None,
    'Path to write the genereated register op file and MLIR file.')

flags.DEFINE_bool('gen_register_op', True,
                  'Generate register op cc file or tfr mlir file.')


@Composite(
    'NewConv2D',
    inputs=['input_: T', 'filter_: T', 'bias: T'],
    attrs=[
        'stride_w: int', 'stride_h: int', 'dilation_w: int', 'dilation_h: int',
        'padding: {"SAME", "VALID"}', 'act: {"", "RELU", "RELU6", "TANH"} = ""'
    ],
    derived_attrs=['T: {float, int8}'],
    outputs=['o: T'])
def _composite_conv_add_relu(input_, filter_, bias, stride_w, stride_h,
                             dilation_w, dilation_h, padding, act):
  res = tf.raw_ops.Conv2D(
      input=input_,
      filter=filter_,
      strides=[1, stride_w, stride_h, 1],
      dilations=[1, dilation_w, dilation_h, 1],
      padding=padding)
  res = tf.raw_ops.Add(x=res, y=bias)
  if act == 'RELU':
    return tf.raw_ops.Relu(features=res)
  elif act == 'RELU6':
    return tf.raw_ops.Relu6(features=res)
  elif act == 'TANH':
    return tf.raw_ops.Tanh(x=res)
  else:
    return res


@tf.RegisterGradient('NewConv2D')
def _conv_add_relu_grad(op, grad):
  act = op.get_attr('act')
  y = op.outputs[0]
  if act == 'RELU':
    grad = gen_nn_ops.relu_grad(grad, y)
  elif act == 'RELU6':
    grad = gen_nn_ops.relu6_grad(grad, y)
  elif act == 'TANH':
    y = math_ops.conj(y)
    grad = gen_math_ops.tanh_grad(y, grad)

  broadcast_shape = tf.shape(y)
  input_value_shape = tf.shape(op.inputs[2])
  _, reduction_axes = tf.raw_ops.BroadcastGradientArgs(
      s0=broadcast_shape, s1=input_value_shape)
  updates_grad_reshaped = tf.reduce_sum(
      grad, axis=reduction_axes, keepdims=True)
  bias_grad = tf.reshape(updates_grad_reshaped, input_value_shape)

  dilations = [1, op.get_attr('dilation_w'), op.get_attr('dilation_h'), 1]
  strides = [1, op.get_attr('stride_w'), op.get_attr('stride_h'), 1]
  padding = op.get_attr('padding')
  shape_0, shape_1 = tf.shape_n([op.inputs[0], op.inputs[1]])
  return [
      tf.compat.v1.nn.conv2d_backprop_input(
          shape_0,
          op.inputs[1],
          grad,
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NHWC'),
      tf.compat.v1.nn.conv2d_backprop_filter(
          op.inputs[0],
          shape_1,
          grad,
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NHWC'), bias_grad
  ]


@Composite(
    'NewFullyConnected',
    inputs=['input_: T', 'filter_: T', 'bias: T'],
    attrs=['act: {"", "RELU", "RELU6", "TANH"} = ""'],
    derived_attrs=['T: {float, int8}'],
    outputs=['o: T'])
def _composite_fully_connected(input_, filter_, bias, act):
  res = tf.raw_ops.MatMul(
      a=input_, b=filter_, transpose_a=False, transpose_b=True)
  res = tf.raw_ops.Add(x=res, y=bias)
  if act == 'RELU':
    return tf.raw_ops.Relu(features=res)
  elif act == 'RELU6':
    return tf.raw_ops.Relu6(features=res)
  elif act == 'TANH':
    return tf.raw_ops.Tanh(x=res)
  else:
    return res


@tf.RegisterGradient('NewFullyConnected')
def _fully_connected_grad(op, grad):
  act = op.get_attr('act')
  y = op.outputs[0]
  if act == 'RELU':
    grad = gen_nn_ops.relu_grad(grad, y)
  elif act == 'RELU6':
    grad = gen_nn_ops.relu6_grad(grad, y)
  elif act == 'TANH':
    y = math_ops.conj(y)
    grad = gen_math_ops.tanh_grad(y, grad)

  broadcast_shape = tf.shape(y)
  input_value_shape = tf.shape(op.inputs[2])
  _, reduction_axes = tf.raw_ops.BroadcastGradientArgs(
      s0=broadcast_shape, s1=input_value_shape)
  updates_grad_reshaped = tf.reduce_sum(
      grad, axis=reduction_axes, keepdims=True)
  bias_grad = tf.reshape(updates_grad_reshaped, input_value_shape)

  a = math_ops.conj(op.inputs[0])
  b = math_ops.conj(op.inputs[1])
  grad_a = gen_math_ops.mat_mul(grad, b)
  grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True)
  return [grad_a, grad_b, bias_grad]


@Composite(
    'NewMaxPool',
    inputs=['input_: T'],
    attrs=[
        'stride_w: int', 'stride_h: int', 'filter_width: int',
        'filter_height: int', 'padding: {"SAME", "VALID"}'
    ],
    derived_attrs=['T: {float, int8}'],
    outputs=['o: T'])
def _composite_max_pool(input_, stride_w, stride_h, filter_width, filter_height,
                        padding):
  ksize = [1, filter_width, filter_height, 1]
  strides = [1, stride_w, stride_h, 1]
  return tf.raw_ops.MaxPool(
      input=input_, ksize=ksize, strides=strides, padding=padding)


@tf.RegisterGradient('NewMaxPool')
def _max_pool_grad(op, grad):
  filter_width = op.get_attr('filter_width')
  filter_height = op.get_attr('filter_height')
  stride_w = op.get_attr('stride_w')
  stride_h = op.get_attr('stride_h')
  padding = op.get_attr('padding')
  return tf.raw_ops.MaxPoolGrad(
      orig_input=op.inputs[0],
      orig_output=op.outputs[0],
      grad=grad,
      ksize=[1, filter_width, filter_height, 1],
      strides=[1, stride_w, stride_h, 1],
      padding=padding,
      data_format='NHWC')


def main(_):
  if FLAGS.gen_register_op:
    assert FLAGS.output.endswith('.cc')
    generated_code = gen_register_op(sys.modules[__name__], '_composite_')
  else:
    assert FLAGS.output.endswith('.mlir')
    generated_code = tfr_gen_from_module(sys.modules[__name__], '_composite_',)

  dirname = os.path.dirname(FLAGS.output)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  with open(FLAGS.output, 'w') as f:
    f.write(generated_code)


if __name__ == '__main__':
  app.run(main=main)
