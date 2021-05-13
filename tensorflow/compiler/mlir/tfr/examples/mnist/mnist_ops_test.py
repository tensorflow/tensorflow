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
"""Tests for tensorflow.compiler.mlir.tfr.examples.mnist.ops_defs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow.compiler.mlir.tfr.examples.mnist import gen_mnist_ops
from tensorflow.compiler.mlir.tfr.examples.mnist import ops_defs
from tensorflow.compiler.mlir.tfr.python import test_utils
from tensorflow.python.framework import load_library
from tensorflow.python.platform import test

_lib_dir = os.path.dirname(gen_mnist_ops.__file__)
_lib_name = os.path.basename(gen_mnist_ops.__file__)[4:].replace('.py', '.so')
load_library.load_op_library(os.path.join(_lib_dir, _lib_name))


class MnistOpsDefsTest(test_utils.OpsDefsTest):

  def test_new_conv2d_relu(self):
    input_ = tf.random.uniform([1, 4, 4, 1])
    filter_ = tf.random.uniform([2, 2, 1, 8])
    bias = tf.zeros([8])
    kwargs = {
        'input_': input_,
        'filter_': filter_,
        'bias': bias,
        'stride_w': 2,
        'stride_h': 2,
        'dilation_w': 1,
        'dilation_h': 1,
        'padding': 'SAME',
        'act': 'RELU'
    }

    self._assertOpAndComposite([input_, filter_, bias],
                               tf.function(gen_mnist_ops.new_conv2d),
                               ops_defs._composite_conv_add_relu, kwargs)

  def test_new_conv2d_relu6(self):
    input_ = tf.random.uniform([1, 4, 4, 1])
    filter_ = tf.random.uniform([2, 2, 1, 8])
    bias = tf.zeros([8])
    kwargs = {
        'input_': input_,
        'filter_': filter_,
        'bias': bias,
        'stride_w': 2,
        'stride_h': 2,
        'dilation_w': 1,
        'dilation_h': 1,
        'padding': 'SAME',
        'act': 'RELU6'
    }

    self._assertOpAndComposite([input_, filter_, bias],
                               tf.function(gen_mnist_ops.new_conv2d),
                               ops_defs._composite_conv_add_relu, kwargs)

  def test_new_conv2d_tanh(self):
    self.skipTest('Fix tanh gradients')
    input_ = tf.random.uniform([1, 4, 4, 1])
    filter_ = tf.random.uniform([2, 2, 1, 8])
    bias = tf.zeros([8])
    kwargs = {
        'input_': input_,
        'filter_': filter_,
        'bias': bias,
        'stride_w': 2,
        'stride_h': 2,
        'dilation_w': 1,
        'dilation_h': 1,
        'padding': 'SAME',
        'act': 'TANH'
    }

    self._assertOpAndComposite([input_, filter_, bias],
                               tf.function(gen_mnist_ops.new_conv2d),
                               ops_defs._composite_conv_add_relu, kwargs)

  def test_new_fully_connected(self):
    input_ = tf.random.uniform([2, 4])
    filter_ = tf.random.uniform([3, 4])
    bias = tf.zeros([3])
    kwargs = {'input_': input_, 'filter_': filter_, 'bias': bias, 'act': 'RELU'}

    self._assertOpAndComposite([input_, filter_, bias],
                               tf.function(gen_mnist_ops.new_fully_connected),
                               ops_defs._composite_fully_connected, kwargs)

  def test_new_max_pool(self):
    input_ = tf.random.uniform([8, 4, 4, 1])
    kwargs = {
        'input_': input_,
        'stride_w': 2,
        'stride_h': 2,
        'filter_width': 1,
        'filter_height': 1,
        'padding': 'SAME',
    }

    self._assertOpAndComposite([input_],
                               tf.function(gen_mnist_ops.new_max_pool),
                               ops_defs._composite_max_pool, kwargs)


if __name__ == '__main__':
  os.environ[
      'TF_MLIR_TFR_LIB_DIR'] = 'tensorflow/compiler/mlir/tfr/examples/mnist'
  test.main()
