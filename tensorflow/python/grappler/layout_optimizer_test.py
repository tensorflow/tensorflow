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

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_lib


def _weight(shape):
  """Generates a weight of a given shape."""
  return random_ops.truncated_normal(shape, seed=0, stddev=0.1)


def _bias(shape):
  """Generates a bias of a given shape."""
  return constant_op.constant(0.1, shape=shape)


def _conv2d(x, w):
  """Returns a 2d convolution layer with full stride."""
  return nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
  """Downsamples a feature map by 2X."""
  return nn.max_pool(
      x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Taken from tensorflow/examples/tutorials/mnist/mnist_deep.py
def _two_layer_model(x):
  x_image = array_ops.reshape(x, [-1, 28, 28, 1])
  w_conv1 = _weight([5, 5, 1, 32])
  b_conv1 = _bias([32])
  h_conv1 = nn.relu(_conv2d(x_image, w_conv1) + b_conv1)
  h_pool1 = _max_pool_2x2(h_conv1)
  w_conv2 = _weight([5, 5, 32, 64])
  b_conv2 = _bias([64])
  h_conv2 = nn.relu(_conv2d(h_pool1, w_conv2) + b_conv2)
  h_pool2 = _max_pool_2x2(h_conv2)
  return h_pool2


def _loop():
  random_seed.set_random_seed(0)
  x1 = random_ops.truncated_normal([1, 784], seed=0)
  x2 = random_ops.truncated_normal([1, 784], seed=0)
  x3 = random_ops.truncated_normal([1, 784], seed=0)
  x4 = random_ops.truncated_normal([1, 784], seed=0)
  elems = (x1, x2, x3, x4)
  outputs = functional_ops.map_fn(_two_layer_model, elems, dtype=dtypes.float32)
  return outputs


def _get_config(layout_optimizer=True):
  if layout_optimizer:
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
  else:
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF)
  graph_options = config_pb2.GraphOptions(
      rewrite_options=rewrite_options, build_cost_model=1)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  return config


def _simple_metagraph(depthwise=False):
  random_seed.set_random_seed(0)
  x = variables.Variable(random_ops.truncated_normal([1, 200, 200, 3], seed=0))
  conv = conv_layers.separable_conv2d if depthwise else conv_layers.conv2d
  y = conv(x, 32, [3, 3])
  z = conv(y, 32, [3, 3])
  optimizer = gradient_descent.GradientDescentOptimizer(1e-4)
  loss = math_ops.reduce_mean(z)
  train_op = optimizer.minimize(loss)
  graph = ops.get_default_graph()
  graph.add_to_collection('train_op', train_op)
  meta_graph = saver_lib.export_meta_graph(graph_def=graph.as_graph_def())
  return meta_graph


def _get_cluster():
  named_device = device_properties_pb2.NamedDevice()
  named_device.name = '/GPU:0'
  named_device.properties.type = 'GPU'
  named_device.properties.environment['architecture'] = '4'
  cluster = gcluster.Cluster(devices=[named_device])
  return cluster


class LayoutOptimizerTest(test.TestCase):
  """Tests the Grappler layout optimizer."""

  def _train(self, checkpoint_path, layout_optimizer=False, restore=False):
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with session.Session(
        config=_get_config(layout_optimizer), graph=graph) as sess:
      batch = 2
      height = 6
      width = 7
      input_channels = 3
      shape = [batch, height, width, input_channels]
      image = array_ops.placeholder(dtype='float32', shape=shape)
      conv1 = conv_layers.conv2d(image, 32, [3, 3])
      conv2 = conv_layers.conv2d(conv1, 32, [3, 3])
      optimizer = gradient_descent.GradientDescentOptimizer(0.01)
      loss = math_ops.reduce_mean(conv2)
      train_op = optimizer.minimize(loss)
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

      if restore:
        saver.restore(sess, checkpoint_path)
      else:
        sess.run(variables.global_variables_initializer())

      np.random.seed(0)
      for _ in range(2):
        image_val = np.random.rand(*shape).astype(np.float32)
        sess.run([loss, train_op], feed_dict={image: image_val})

      if restore:
        all_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        all_vars_values = [var.eval(session=sess) for var in all_vars]
        return all_vars_values
      else:
        saver.save(sess, checkpoint_path)

  def testTwoConvLayers(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      output = _two_layer_model(x)

      with session.Session() as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
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
      output = _loop()

      with session.Session() as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testGradient(self):
    meta_graph = _simple_metagraph()
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
    optimized_graph = tf_optimizer.OptimizeGraph(
        rewrite_options, meta_graph, cluster=_get_cluster())

    found = 0
    for node in optimized_graph.node:
      if node.op in ['Conv2D', 'Conv2DBackpropFilter', 'Conv2DBackpropInput']:
        found += 1
        self.assertEqual(node.attr['data_format'].s, 'NCHW')
    self.assertEqual(found, 5)

  def testDepthwise(self):
    meta_graph = _simple_metagraph(depthwise=True)
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
    optimized_graph = tf_optimizer.OptimizeGraph(
        rewrite_options, meta_graph, cluster=_get_cluster())

    found = 0
    for node in optimized_graph.node:
      if node.op in [
          'DepthwiseConv2dNative', 'DepthwiseConv2dNativeBackpropFilter',
          'DepthwiseConv2dNativeBackpropInput'
      ]:
        found += 1
        self.assertEqual(node.attr['data_format'].s, 'NCHW')
    self.assertEqual(found, 6)

  def testCheckpointCompatibility(self):
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')

    checkpoint_path = self.get_temp_dir()
    self._train(checkpoint_path)
    vars_expected = self._train(checkpoint_path, restore=True)
    vars_layout_optimized = self._train(
        checkpoint_path, restore=True, layout_optimizer=True)

    for var_expected, var_layout_optimized in zip(vars_expected,
                                                  vars_layout_optimized):
      self.assertAllClose(var_expected, var_layout_optimized, atol=1e-6)


if __name__ == '__main__':
  test.main()
