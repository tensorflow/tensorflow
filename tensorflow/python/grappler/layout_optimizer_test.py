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
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
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


def _model_with_second_port():
  random_seed.set_random_seed(0)
  x = random_ops.truncated_normal([2, 5, 5, 4], seed=0)
  scale = constant_op.constant(0.1, shape=[4])
  offset = constant_op.constant(0.3, shape=[4])
  y, mean, _ = nn.fused_batch_norm(x, scale, offset)
  mul = math_ops.add(y, mean)
  output = array_ops.identity(mul)
  return output


def _model_with_branch(x):
  x_image = array_ops.reshape(x, [-1, 28, 28, 1])
  w_conv1 = _weight([5, 5, 1, 32])
  w_conv2 = _weight([5, 5, 1, 32])
  c_conv1 = _conv2d(x_image, w_conv1)
  c_conv2 = _conv2d(x_image, w_conv2)
  add = math_ops.add(c_conv1, c_conv2)
  return add


def _model_with_vec_and_4d(x):
  x_image = array_ops.reshape(x, [-1, 28, 28, 1])
  w_conv1 = _weight([5, 5, 1, 32])
  c_conv1 = _conv2d(x_image, w_conv1)
  vector = constant_op.constant(6.4, shape=[32])
  add = math_ops.add(c_conv1, vector)
  return add


def _loop():
  random_seed.set_random_seed(0)
  x1 = random_ops.truncated_normal([1, 784], seed=0)
  x2 = random_ops.truncated_normal([1, 784], seed=0)
  x3 = random_ops.truncated_normal([1, 784], seed=0)
  x4 = random_ops.truncated_normal([1, 784], seed=0)
  elems = (x1, x2, x3, x4)
  outputs = map_fn.map_fn(_two_layer_model, elems, dtype=dtypes.float32)
  return outputs


def _loop_with_branch():
  random_seed.set_random_seed(0)
  x1 = random_ops.truncated_normal([1, 784], seed=0)
  x2 = random_ops.truncated_normal([1, 784], seed=0)
  x3 = random_ops.truncated_normal([1, 784], seed=0)
  x4 = random_ops.truncated_normal([1, 784], seed=0)
  elems = (x1, x2, x3, x4)
  outputs = map_fn.map_fn(_model_with_branch, elems, dtype=dtypes.float32)
  return outputs


def _loop_with_vec_and_4d():
  random_seed.set_random_seed(0)
  x1 = random_ops.truncated_normal([1, 784], seed=0)
  x2 = random_ops.truncated_normal([1, 784], seed=0)
  x3 = random_ops.truncated_normal([1, 784], seed=0)
  x4 = random_ops.truncated_normal([1, 784], seed=0)
  elems = (x1, x2, x3, x4)
  outputs = map_fn.map_fn(_model_with_vec_and_4d, elems, dtype=dtypes.float32)
  return outputs


def _get_config(layout_optimizer=True):
  if layout_optimizer:
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.ON,
        # do not remove duplicated nodes
        arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF)
  else:
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
        # do not remove duplicated nodes
        arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF)
  rewrite_options.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(
      rewrite_options=rewrite_options, build_cost_model=1)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  config.graph_options.optimizer_options.opt_level = -1
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
  named_device.properties.num_cores = 24
  named_device.properties.frequency = 1000
  named_device.properties.environment['architecture'] = '4'
  cluster = gcluster.Cluster(devices=[named_device])
  return cluster


def _is_transpose(node):
  return node.endswith('TransposeNHWCToNCHW-LayoutOptimizer') or node.endswith(
      'TransposeNCHWToNHWC-LayoutOptimizer')


def _is_permute(node):
  return node.endswith('VecPermuteNHWCToNCHW-LayoutOptimizer') or node.endswith(
      'VecPermuteNCHWToNHWC-LayoutOptimizer')


class LayoutOptimizerTest(test.TestCase):
  """Tests the Grappler layout optimizer."""

  def _assert_trans_nchw_to_nhwc(self, name, nodes):
    self.assertIn(name + '-TransposeNCHWToNHWC-LayoutOptimizer', nodes)

  def _assert_trans_nhwc_to_nchw(self, name, nodes):
    self.assertIn(name + '-TransposeNHWCToNCHW-LayoutOptimizer', nodes)

  def _assert_map_nhwc_to_nchw(self, name, nodes):
    self.assertIn(name + '-DimMapNHWCToNCHW-LayoutOptimizer', nodes)

  def _assert_vec_nchw_to_nhwc(self, name, nodes):
    self.assertIn(name + '-VecPermuteNCHWToNHWC-LayoutOptimizer', nodes)

  def _assert_vec_nhwc_to_nchw(self, name, nodes):
    self.assertIn(name + '-VecPermuteNHWCToNCHW-LayoutOptimizer', nodes)

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
        self.evaluate(variables.global_variables_initializer())

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

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Relu_1-0-0', nodes)

      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSplitWithNonConstAxis(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      dim = array_ops.placeholder(dtype='int32')
      split = array_ops.split(conv, 2, axis=dim)
      scale = constant_op.constant(0.1, shape=[32])
      offset = constant_op.constant(0.3, shape=[32])
      bn0 = nn.fused_batch_norm(split[0], scale, offset)
      bn1 = nn.fused_batch_norm(split[1], scale, offset)
      add = bn0[0] + bn1[0]
      output = array_ops.identity(add)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={dim: 3})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata, feed_dict={dim: 3})

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('add_2-0-0', nodes)
      self._assert_map_nhwc_to_nchw('split-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSplitVWithNonConstAxis(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      dim = array_ops.placeholder(dtype='int32')
      sizes = constant_op.constant([50, 10, 4], shape=[3])
      split = gen_array_ops.split_v(
          value=conv, size_splits=sizes, axis=dim, num_split=3)
      output = math_ops.reduce_sum(split[0])

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={dim: 3})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata, feed_dict={dim: 3})

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('SplitV-0-0', nodes)
      self._assert_map_nhwc_to_nchw('SplitV-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testPadWithConstPaddings(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      paddings_val = [[1, 2], [3, 4], [5, 6], [7, 8]]
      paddings = constant_op.constant(
          paddings_val, dtype='int32', name='PaddingsConst')
      pad = array_ops.pad(conv, paddings)
      output = array_ops.identity(pad)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Pad-0-0', nodes)
      self.assertIn('Pad-1-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReduceSum(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv)
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testCast(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      cast = math_ops.cast(conv, dtype='bool')
      output = array_ops.identity(cast)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Cast-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSqueeze(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[1, 2])
      squeeze = array_ops.squeeze(reduce_sum)
      output = array_ops.identity(squeeze)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSqueezeAlongHW(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[1, 2], keepdims=True)
      squeeze = array_ops.squeeze(reduce_sum, axis=[1, 2])
      output = array_ops.identity(squeeze)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSqueezeAlongNHW(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[0, 1, 2], keepdims=True)
      squeeze = array_ops.squeeze(reduce_sum, axis=[0, 1, 2])
      output = array_ops.identity(squeeze)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReduceSumAlongHWC(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[1, 2, 3])
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReduceSumAlongNHW(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[0, 1, 2])
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReduceSumAlongC(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[3])
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReduceSumAlongCKeepDims(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[3], keepdims=True)
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Sum-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReduceSumAlongHKeepDims(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[2], keepdims=True)
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReduceSumAlongWCKeepDims(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[2, 3], keepdims=True)
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testConcatWithControlDependency(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      axis = constant_op.constant(3)
      var = variables.Variable(3)
      assign = state_ops.assign(var, 6)
      with ops.control_dependencies([assign]):
        concat = array_ops.concat([conv, conv], axis)
      output = array_ops.identity(concat)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('concat-0-0', nodes)
      self.assertIn('concat-2-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testFill(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = array_ops.placeholder(dtype='float32')
      conv = _two_layer_model(x)
      shape = array_ops.shape(conv)
      scalar = array_ops.constant(5.7)
      fill = array_ops.fill(shape, scalar)
      output = array_ops.identity(fill)

      x_val = [3.4] * 784
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={x: x_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                x: x_val
            })

      nodes = []
      num_transposes = 0
      num_vec_permute = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        if _is_permute(node.name):
          num_vec_permute += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      # Two vector permute nodes were initially added in the Expand phase of
      # LayoutOptimizer; they cancelled out each other in the Collapse phase.
      expected_vec_permute = 0
      self.assertEqual(expected_vec_permute, num_vec_permute)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Fill-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testTile(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      multiple = array_ops.placeholder(dtype='int32')
      tile = array_ops.tile(conv, multiple)
      output = array_ops.identity(tile)

      multiple_val = [2, 3, 4, 1]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={multiple: multiple_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                multiple: multiple_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Tile-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('Tile-1', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReverseWithConstDims(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      dims = constant_op.constant([3, 1], name='DimsConst')
      reverse = array_ops.reverse(conv, dims)
      output = array_ops.identity(reverse)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('ReverseV2-0-0', nodes)
      self.assertIn('ReverseV2-1-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testReverseWithNonConstDims(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      dims = array_ops.placeholder(dtype='int32')
      reverse = array_ops.reverse(conv, dims)
      output = array_ops.identity(reverse)

      dims_val = [2, 3]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={dims: dims_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                dims: dims_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('ReverseV2-0-0', nodes)
      self._assert_map_nhwc_to_nchw('ReverseV2-1', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSelectOp(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      add = math_ops.add(conv, conv)
      mean = math_ops.reduce_mean(conv)
      condition = math_ops.less(conv, mean)
      select = gen_math_ops.select(condition, conv, add)
      output = array_ops.identity(select)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Select-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSelectOpConditionUnknownShape(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      add = math_ops.add(conv, conv)
      condition = array_ops.placeholder(dtype='bool')
      select = gen_math_ops.select(condition, conv, add)
      output = array_ops.identity(select)

      condition_val = np.zeros((1, 7, 7, 64))
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={condition: condition_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={condition: condition_val})

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 3
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSelectOpScalarCondition(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      add = math_ops.add(conv, conv)
      condition = constant_op.constant(True)
      select = gen_math_ops.select(condition, conv, add)
      output = array_ops.identity(select)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Select-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testPadWithNonConstPaddings(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      paddings = array_ops.placeholder(dtype='int32')
      pad = array_ops.pad(conv, paddings)
      output = array_ops.identity(pad)

      paddings_val = [[1, 2], [3, 4], [5, 6], [7, 8]]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={paddings: paddings_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                paddings: paddings_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Pad-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('Pad-1', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testMaxPoolV2(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      ksize = constant_op.constant([1, 2, 3, 1], shape=[4])
      strides = array_ops.placeholder(dtype='int32', shape=[4])
      max_pool = gen_nn_ops.max_pool_v2(conv, ksize, strides, 'VALID')
      output = array_ops.identity(max_pool)

      strides_val = [1, 3, 2, 1]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={strides: strides_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                strides: strides_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('MaxPoolV2-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('MaxPoolV2-2', nodes)
      self.assertIn('MaxPoolV2-1-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testMaxPoolGradV2(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      ksize = constant_op.constant([1, 2, 3, 1], shape=[4])
      strides = array_ops.placeholder(dtype='int32', shape=[4])
      max_pool_grad = gen_nn_ops.max_pool_grad_v2(conv, conv, conv, ksize,
                                                  strides, 'VALID')
      output = array_ops.identity(max_pool_grad)

      strides_val = [1, 3, 2, 1]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={strides: strides_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                strides: strides_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('MaxPoolGradV2-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('MaxPoolGradV2-4', nodes)
      self.assertIn('MaxPoolGradV2-3-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testSliceWithNonConstAxis(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      size = array_ops.placeholder(dtype='int32')
      s = array_ops.slice(conv, [0, 0, 0, 0], size)
      output = array_ops.identity(s)

      size_val = [1, 2, 3, 4]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={size: size_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                size: size_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Slice-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('Slice-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testStridedSliceWithNonConstAxis(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      end = array_ops.placeholder(dtype='int32')
      s = array_ops.strided_slice(conv, [0, 0, 0, 0], end, strides=[1, 2, 3, 1])
      output = array_ops.identity(s)

      end_val = [1, 2, 3, 4]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={end: end_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                end: end_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('StridedSlice-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('StridedSlice-2', nodes)
      self.assertIn('StridedSlice-1-LayoutOptimizer', nodes)
      self.assertIn('StridedSlice-3-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testStridedSliceWithMask1011(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      # This will generate a StridedSlice op with begin mask and
      # end mask 11(1011).
      s = conv[:, :, 1:-1, :]
      output = array_ops.identity(s)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('strided_slice-0-0', nodes)
      self.assertIn('strided_slice-1-LayoutOptimizer', nodes)
      self.assertIn('strided_slice-2-LayoutOptimizer', nodes)
      self.assertIn('strided_slice-3-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testStridedSliceWithMask0111(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      # This will generate a StridedSlice op with begin mask and
      # end mask 7(0111).
      s = conv[:, :, :, 1:-1]
      output = array_ops.identity(s)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('strided_slice-0-0', nodes)
      self.assertIn('strided_slice-1-LayoutOptimizer', nodes)
      self.assertIn('strided_slice-2-LayoutOptimizer', nodes)
      self.assertIn('strided_slice-3-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testStridedSliceGradWithNonConstAxis(self):
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      end = array_ops.placeholder(dtype='int32')
      shape = array_ops.shape(conv)
      end_val = [1, 2, 3, 4]
      s = array_ops.strided_slice(
          conv, [0, 0, 0, 0], end_val, strides=[1, 2, 3, 1])
      s_grad = array_ops.strided_slice_grad(shape, [0, 0, 0, 0], end,
                                            [1, 2, 3, 1], s)
      output = array_ops.identity(s_grad)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={end: end_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                end: end_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('StridedSliceGrad-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('StridedSliceGrad-2', nodes)
      self.assertIn('StridedSlice-1-LayoutOptimizer', nodes)
      self.assertIn('StridedSlice-2-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testShapeN(self):
    if test.is_gpu_available(cuda_only=True):
      x = array_ops.placeholder(dtype='float32')
      conv = _two_layer_model(x)
      shapen = array_ops.shape_n([conv, conv])
      output = math_ops.add(shapen[0], shapen[1])

      x_val = [1.7] * 784
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={x: x_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                x: x_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_vec_nchw_to_nhwc('ShapeN-0-0', nodes)
      self.assertAllEqual(output_val_ref, output_val)

  def testShapeNFollowedByNotConvertibleNodeReshape(self):
    if test.is_gpu_available(cuda_only=True):
      x = array_ops.placeholder(dtype='float32')
      conv = _two_layer_model(x)
      conv_reshape = array_ops.reshape(conv, [1, 1, 1, -1])
      shapen = array_ops.shape_n([conv, conv_reshape])
      shape = array_ops.identity(shapen[1])
      ones = array_ops.ones(shape)
      output = math_ops.add_n([conv_reshape, ones])

      x_val = [1.7] * 784
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={x: x_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={x: x_val})

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testLoop(self):
    if test.is_gpu_available(cuda_only=True):
      output = _loop()

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('map/while/Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('map/while/MaxPool_1-0-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testLoopWithBranch(self):
    if test.is_gpu_available(cuda_only=True):
      output = _loop_with_branch()

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 3
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('map/while/Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('map/while/Add_1-0-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testLoopWithVecAnd4D(self):
    if test.is_gpu_available(cuda_only=True):
      output = _loop_with_vec_and_4d()

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('map/while/Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('map/while/Add_1-0-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  def testBinaryOpSecondPort(self):
    if test.is_gpu_available(cuda_only=True):
      output = _model_with_second_port()

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('FusedBatchNorm-0', nodes)
      self._assert_trans_nchw_to_nhwc('Add-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.run_deprecated_v1
  def testGradient(self):
    meta_graph = _simple_metagraph()
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            layout_optimizer=rewriter_config_pb2.RewriterConfig.ON,
            min_graph_nodes=-1))
    optimized_graph = tf_optimizer.OptimizeGraph(
        config, meta_graph, cluster=_get_cluster())

    found = 0
    for node in optimized_graph.node:
      if node.op in ['Conv2D', 'Conv2DBackpropFilter', 'Conv2DBackpropInput']:
        found += 1
        self.assertEqual(node.attr['data_format'].s, b'NCHW')
    self.assertEqual(found, 5)

  @test_util.run_deprecated_v1
  def testDepthwise(self):
    meta_graph = _simple_metagraph(depthwise=True)
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            layout_optimizer=rewriter_config_pb2.RewriterConfig.ON,
            min_graph_nodes=-1))
    optimized_graph = tf_optimizer.OptimizeGraph(
        config, meta_graph, cluster=_get_cluster())

    found = 0
    for node in optimized_graph.node:
      if node.op in [
          'DepthwiseConv2dNative', 'DepthwiseConv2dNativeBackpropFilter',
          'DepthwiseConv2dNativeBackpropInput'
      ]:
        found += 1
        self.assertEqual(node.attr['data_format'].s, b'NCHW')
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
