# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Grappler LayoutOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver


def weight(shape):
  """weights generates a weight of a given shape."""
  return random_ops.truncated_normal(shape, seed=0, stddev=0.1)


def bias(shape):
  """bias generates a bias of a given shape."""
  return constant_op.constant(0.1, shape=shape)


def conv2d(x, w):
  """conv2d returns a 2d convolution layer with full stride."""
  return nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return nn.max_pool(
      x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Taken from tensorflow/examples/tutorials/mnist/mnist_deep.py
def two_layer_model(x):
  x_image = array_ops.reshape(x, [-1, 28, 28, 1])
  w_conv1 = weight([5, 5, 1, 32])
  b_conv1 = bias([32])
  h_conv1 = nn.relu(conv2d(x_image, w_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  w_conv2 = weight([5, 5, 32, 64])
  b_conv2 = bias([64])
  h_conv2 = nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  return h_pool2


def loop():
  random_seed.set_random_seed(0)
  x1 = random_ops.truncated_normal([1, 784], seed=0)
  x2 = random_ops.truncated_normal([1, 784], seed=0)
  x3 = random_ops.truncated_normal([1, 784], seed=0)
  x4 = random_ops.truncated_normal([1, 784], seed=0)
  elems = (x1, x2, x3, x4)
  outputs = functional_ops.map_fn(two_layer_model, elems, dtype=dtypes.float32)
  return outputs


def get_config():
  rewrite_options = rewriter_config_pb2.RewriterConfig(
      optimize_tensor_layout=True)
  graph_options = config_pb2.GraphOptions(
      rewrite_options=rewrite_options, build_cost_model=1)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  return config


class LayoutOptimizerTest(test.TestCase):
  """Tests the Grappler layout optimizer."""

  def testTwoConvLayers(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      output = two_layer_model(x)

      with session.Session() as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if node.name.startswith('LayoutOptimizerTranspose'):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self.assertIn('LayoutOptimizerTransposeNHWCToNCHW-Conv2D-Reshape-0',
                    nodes)
      self.assertIn('LayoutOptimizerTransposeNCHWToNHWC-Relu_1-MaxPool_1',
                    nodes)

      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testLoop(self):
    if test.is_gpu_available(cuda_only=True):
      output = loop()

      with session.Session() as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testGradient(self):
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')

    random_seed.set_random_seed(0)
    x = random_ops.truncated_normal([1, 200, 200, 3], seed=0)
    y = conv_layers.conv2d(x, 32, [3, 3])
    z = conv_layers.conv2d(y, 32, [3, 3])
    optimizer = gradient_descent.GradientDescentOptimizer(1e-4)
    loss = math_ops.reduce_mean(z)
    train_op = optimizer.minimize(loss)
    graph = ops.get_default_graph()
    graph.add_to_collection('train_op', train_op)
    meta_graph = saver.export_meta_graph(graph_def=graph.as_graph_def())

    rewrite_options = rewriter_config_pb2.RewriterConfig(
        optimize_tensor_layout=True)
    optimized_graph = tf_optimizer.OptimizeGraph(rewrite_options, meta_graph)

    found = 0
    for node in optimized_graph.node:
      if node.op in ['Conv2D', 'Conv2DBackpropFilter', 'Conv2DBackpropInput']:
        found += 1
        self.assertEqual(node.attr['data_format'].s, 'NCHW')
    self.assertEqual(found, 5)


if __name__ == '__main__':
  test.main()
