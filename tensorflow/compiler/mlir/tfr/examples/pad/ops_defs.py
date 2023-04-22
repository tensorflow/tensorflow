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
"""Defines the mirror pad and mirror pad grad."""

# pylint: disable=g-direct-tensorflow-import
# pylint: disable=missing-function-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
import tensorflow as tf

from tensorflow.compiler.mlir.tfr.python import composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.compiler.mlir.tfr.python.tfr_gen import tfr_gen_from_module
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import flags

Composite = composite.Composite
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output', None,
    'Path to write the genereated register op file and MLIR file.')

flags.DEFINE_bool('gen_register_op', True,
                  'Generate register op cc file or tfr mlir file.')


@Composite(
    'NewMirrorPad',
    inputs=['input_: T', 'paddings: Tpaddings'],
    attrs=['mode: {"REFLECT", "SYMMETRIC"}'],
    derived_attrs=['T: type', 'Tpaddings: {int32, int64} = DT_INT32'],
    outputs=['output: T'])
def _composite_mirror_pad(input_, paddings, mode):
  shape = input_.shape.as_list()
  for i in range(len(shape)):
    rdims = tf.raw_ops.OneHot(
        indices=i, depth=len(shape), on_value=True, off_value=False, axis=-1)
    rarray = tf.raw_ops.Reverse(tensor=input_, dims=rdims)

    left_padding_size = tf.raw_ops.GatherNd(params=paddings, indices=[i, 0])
    right_padding_size = tf.raw_ops.GatherNd(params=paddings, indices=[i, 1])

    if mode == 'REFLECT':
      left_padding, _ = tf.raw_ops.SplitV(
          value=rarray,
          size_splits=[left_padding_size, -1],
          axis=i,
          num_split=2)
      _, right_padding = tf.raw_ops.SplitV(
          value=rarray,
          size_splits=[-1, right_padding_size],
          axis=i,
          num_split=2)
    else:
      _, left_padding = tf.raw_ops.SplitV(
          value=rarray,
          size_splits=[-1, left_padding_size],
          axis=i,
          num_split=2)
      right_padding, _ = tf.raw_ops.SplitV(
          value=rarray,
          size_splits=[right_padding_size, -1],
          axis=i,
          num_split=2)

    input_ = tf.raw_ops.Concat(
        concat_dim=i, values=[left_padding, input_, right_padding])
  return input_


@tf.RegisterGradient('NewMirrorPad')
def _mirror_pad_grad(op, grad):
  mode = op.get_attr('mode')
  return [gen_array_ops.mirror_pad_grad(grad, op.inputs[1], mode=mode), None]


@Composite(
    'NewMirrorPadGrad',
    inputs=['input_: T', 'paddings: Tpaddings'],
    attrs=['mode: {"REFLECT", "SYMMETRIC"}'],
    derived_attrs=['T: type', 'Tpaddings: {int32, int64} = DT_INT32'],
    outputs=['output: T'])
def _composite_mirror_pad_grad(input_, paddings, mode):
  shape = input_.shape.as_list()
  for i in range(len(shape)):
    rdims = tf.raw_ops.OneHot(
        indices=i, depth=len(shape), on_value=True, off_value=False, axis=-1)
    left_padding_size = tf.raw_ops.GatherNd(params=paddings, indices=[i, 0])
    right_padding_size = tf.raw_ops.GatherNd(params=paddings, indices=[i, 1])

    left_padding, core, right_padding = tf.raw_ops.SplitV(
        value=input_,
        size_splits=[left_padding_size, -1, right_padding_size],
        axis=i,
        num_split=3)
    reversed_left_padding = tf.raw_ops.Reverse(tensor=left_padding, dims=rdims)
    reversed_right_padding = tf.raw_ops.Reverse(
        tensor=right_padding, dims=rdims)
    zero_like = tf.raw_ops.ZerosLike(x=core)
    left_offset, _ = tf.raw_ops.SplitV(
        value=zero_like,
        size_splits=[-1, left_padding_size],
        axis=i,
        num_split=2)
    right_offset, _ = tf.raw_ops.SplitV(
        value=zero_like,
        size_splits=[-1, right_padding_size],
        axis=i,
        num_split=2)

    if mode == 'REFLECT':
      from_left_padding = tf.raw_ops.Concat(
          concat_dim=i, values=[left_offset, reversed_left_padding])
      from_right_padding = tf.raw_ops.Concat(
          concat_dim=i, values=[reversed_right_padding, right_offset])
    else:
      from_left_padding = tf.raw_ops.Concat(
          concat_dim=i, values=[reversed_left_padding, left_offset])
      from_right_padding = tf.raw_ops.Concat(
          concat_dim=i, values=[right_offset, reversed_right_padding])
    input_ = tf.raw_ops.AddN(
        inputs=[from_left_padding, core, from_right_padding])

  return input_


@tf.RegisterGradient('NewMirrorPadGrad')
def _mirror_pad_grad_grad(op, grad):
  mode = op.get_attr('mode')
  return [gen_array_ops.mirror_pad(grad, op.inputs[1], mode=mode), None]


def main(_):
  if FLAGS.gen_register_op:
    assert FLAGS.output.endswith('.cc')
    generated_code = gen_register_op(sys.modules[__name__], '_composite_')
  else:
    assert FLAGS.output.endswith('.mlir')
    generated_code = tfr_gen_from_module(sys.modules[__name__], '_composite_')

  dirname = os.path.dirname(FLAGS.output)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  with open(FLAGS.output, 'w') as f:
    f.write(generated_code)


if __name__ == '__main__':
  app.run(main=main)
