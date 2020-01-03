# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Grappler AutoMixedPrecision."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent


def _input(shape):
  """Generates an input of a given shape."""
  return variables.Variable(random_ops.truncated_normal(shape, seed=0))


def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  return variables.Variable(
      lambda: init_ops.glorot_uniform_initializer(seed=0)(shape))


def _bias(shape):
  """Generates a bias of a given shape."""
  return constant_op.constant(0.1, shape=shape)


def _conv2d(x, w):
  """Returns a 2d convolution layer with full stride."""
  return nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def _conv3d(x, w):
  """Returns a 3d convolution layer with full stride."""
  return nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
  """Downsamples a feature map by 2X."""
  return nn.max_pool(
      x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def _fused_batchnorm(x, scale, offset):
  """Batchnorm."""
  return nn_impl.fused_batch_norm(
      x, scale=scale, offset=offset, is_training=True)


def _conv_bn(x):
  """Conv followed by batchnorm."""
  i = array_ops.reshape(x, [-1, 8, 8, 1])
  f = _weight([3, 3, 1, 6])
  x = _conv2d(i, f)
  s = _weight([6])
  o = _weight([6])
  y, _, _ = _fused_batchnorm(x, s, o)
  y = array_ops.identity(y)
  return y


def _conv3d_bn(x):
  """Conv3D followed by batchnorm."""
  i = array_ops.reshape(x, [-1, 8, 8, 8, 1])
  f = _weight([3, 3, 3, 1, 6])
  x = _conv3d(i, f)
  s = _weight([6])
  o = _weight([6])
  x = array_ops.reshape(x, [-1, 8, 8, 6])
  y, _, _ = _fused_batchnorm(x, s, o)
  y = array_ops.identity(y)
  return y


def _matmul_act(x):
  """Matmul followed by activation."""
  i = array_ops.reshape(x, [8, 8])
  f = _weight([8, 8])
  x = math_ops.matmul(i, f)
  y = nn.relu(x)
  return y


def _conv_pool(x):
  """(Conv -> bias -> relu -> max_pool) x2."""
  x_image = array_ops.reshape(x, [-1, 8, 8, 1])
  w_conv1 = _weight([3, 3, 1, 6])
  b_conv1 = _bias([6])
  h_conv1 = nn.relu(nn.bias_add(_conv2d(x_image, w_conv1), b_conv1))
  h_pool1 = _max_pool_2x2(h_conv1)
  w_conv2 = _weight([3, 3, 6, 4])
  b_conv2 = _bias([4])
  h_conv2 = nn.relu(nn.bias_add(_conv2d(h_pool1, w_conv2), b_conv2))
  h_pool2 = _max_pool_2x2(h_conv2)
  return h_pool2


def _simple_loop(x, functor):
  """Simple loop whose body is provided by the functor."""
  init = (constant_op.constant(0), x)
  c = lambda i, j: i < 4
  b = lambda i, j: (i + 1, functor(j))
  ij = control_flow_ops.while_loop(c, b, init)
  return ij


def _loop_vars_intertwined(x0, y0, functor_x, functor_y):
  """Loop whose loop variables are intertwined."""
  c = lambda i, j, x, y: j < 4
  b = lambda i, j, x, y: (j + 1, i + 1, functor_y(y), functor_x(x))
  init = (constant_op.constant(0), constant_op.constant(0), x0, y0)
  ijzw = control_flow_ops.while_loop(c, b, init)
  return ijzw


def _lstm_cell(prev_c, prev_h, x):
  """Create an LSTM cell."""
  # i: input gate
  # f: forget gate
  # o: output gate
  # c: cell state
  # x: input
  # h: embedding
  bias = _bias([4])
  w = _weight([8, 16])
  ifoc = math_ops.matmul(array_ops.concat([x, prev_h], axis=1), w)
  i, f, o, c = array_ops.split(ifoc, 4, axis=1)
  i = math_ops.sigmoid(nn.bias_add(i, bias))
  f = math_ops.sigmoid(nn.bias_add(f, bias))
  o = math_ops.sigmoid(nn.bias_add(o, bias))
  c = math_ops.tanh(nn.bias_add(c, bias))
  next_c = f * prev_c + i * c
  next_h = o * math_ops.tanh(next_c)
  return next_c, next_h


def _recurrent_lstm(c, h):
  """Dynamic single-layer LSTM with TensorArray."""

  def cond(i, c, h, ta_x):
    del c, h, ta_x
    return i < 4

  def body(i, c, h, ta_x):
    x = ta_x.read(i)
    next_c, next_h = _lstm_cell(c, h, x)
    return (i + 1, next_c, next_h, ta_x)

  ta_x = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=4)
  for i in range(0, 4):
    ta_x = ta_x.write(
        i, constant_op.constant(0.1, shape=[8, 4], dtype=dtypes.float32))
  init = (constant_op.constant(0), c, h, ta_x)
  r = control_flow_ops.while_loop(cond, body, init)
  return r


def _make_node_with_color(color, input_tensor, name=None):
  """Returns a node representative of the specified list type."""
  color = color.lower()
  if color == 'w':  # White node
    weights = _weight(input_tensor.get_shape().as_list())
    return math_ops.matmul(input_tensor, weights, name=name)
  if color == 'g':  # Gray node
    return math_ops.add(input_tensor, 0.1, name=name)
  if color == 'c':  # Clear node
    return nn.relu(input_tensor, name=name)
  if color == 'b':  # Black node
    return math_ops.sqrt(math_ops.pow(input_tensor, 2.), name=name)
  raise ValueError('Invalid node color: ' + str(color))


def _build_simple_loop_graph(inp_colors, body_colors, out_colors):
  """Builds a test graph with a simple loop."""
  a = _input([8, 8])
  for i, color in enumerate(inp_colors):
    a = _make_node_with_color(color, a, 'input_%i' % i)

  def body(x):
    for i, color in enumerate(body_colors):
      x = _make_node_with_color(color, x, 'body_%i' % i)
    return x

  _, a = _simple_loop(a, body)
  for i, color in enumerate(out_colors):
    a = _make_node_with_color(color, a, 'output_%i' % i)
  a = array_ops.identity(a)
  return a


def _get_config(auto_mixed_precision=True):
  """Returns a ConfigProto with auto mixed precision enabled if appropriate."""
  if auto_mixed_precision:
    rewrite_config = rewriter_config_pb2.RewriterConfig(
        auto_mixed_precision=rewriter_config_pb2.RewriterConfig.ON,
        # do not remove duplicated nodes
        arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF)
  else:
    rewrite_config = rewriter_config_pb2.RewriterConfig(
        auto_mixed_precision=rewriter_config_pb2.RewriterConfig.OFF,
        # do not remove duplicated nodes
        arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF)
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(
      rewrite_options=rewrite_config, build_cost_model=1)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  config.graph_options.optimizer_options.opt_level = -1
  return config


def _is_cast_to_fp16(node_name):
  return node_name.endswith('-CastToFp16-AutoMixedPrecision')


def _is_cast_to_fp32(node_name):
  return node_name.endswith('-CastToFp32-AutoMixedPrecision')


def _count_casts(nodes):
  num_to_fp16 = 0
  num_to_fp32 = 0
  for node in nodes:
    if _is_cast_to_fp16(node.name):
      num_to_fp16 += 1
    elif _is_cast_to_fp32(node.name):
      num_to_fp32 += 1
  return num_to_fp16, num_to_fp32


def _build_node_map(nodes):
  node_map = {}
  for node in nodes:
    node_map[node.name] = node
  return node_map


def _example_noninlined_funcdef_shape(op):
  return [op.inputs[0].shape]


@function.Defun(
    shape_func=_example_noninlined_funcdef_shape,
    func_name='example_noninlined_funcdef_grad',
    noinline=True)
def _example_noninlined_funcdef_grad(features, grad):
  """Gradient of Swish function defined below."""
  sigmoid_features = math_ops.sigmoid(features)
  activation_grad = (
      sigmoid_features * (1.0 + features * (1.0 - sigmoid_features)))
  return grad * activation_grad


@function.Defun(
    grad_func=_example_noninlined_funcdef_grad,
    shape_func=_example_noninlined_funcdef_shape,
    func_name='example_noninlined_funcdef',
    noinline=True)
def _example_noninlined_funcdef(features):
  """Computes the Swish activation function: `x * sigmoid(x)`."""
  return features * math_ops.sigmoid(features)


class AutoMixedPrecisionTest(test.TestCase):
  """Tests the Grappler auto mixed precision optimizer."""
  IGNORE_PERF_VAR = 'TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'

  # TODO(benbarsdell): Add tests for eager mode with a tf.function.

  def setUp(self):
    super(AutoMixedPrecisionTest, self).setUp()
    # Enable the tests to be run on pre-Volta GPUs by telling the grappler pass
    # to ignore performance and always transform the graph.
    self._original_ignore_perf_value = os.getenv(self.IGNORE_PERF_VAR)
    os.environ[self.IGNORE_PERF_VAR] = '1'

  def tearDown(self):
    if self._original_ignore_perf_value is not None:
      os.environ[self.IGNORE_PERF_VAR] = self._original_ignore_perf_value
    else:
      del os.environ[self.IGNORE_PERF_VAR]
    super(AutoMixedPrecisionTest, self).tearDown()

  def _assert_output_fp16(self, node_map, node_name, output_port=0):
    self.assertEqual(node_map[node_name].output_info[output_port].dtype,
                     types_pb2.DT_HALF)

  def _run(self, fetches):
    """Runs the graph and returns the evaluation of the fetches."""
    with session.Session(config=_get_config(False)) as sess:
      sess.run(variables.global_variables_initializer())
      output_val_ref = self.evaluate(fetches)

    with session.Session(config=_get_config()) as sess:
      sess.run(variables.global_variables_initializer())
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(fetches, run_metadata=metadata)

    return output_val_ref, output_val, metadata.cost_graph

  def _run_simple_loop_test(self, inp, body, out):
    """Runs a test of a simple loop.

    The loop has different node colors in different sections of the graph. The
    arguments must be strings where each character represents the color of a
    node in that section of the graph: w = white, g = gray, c = clear,
    b = black. CAPITALIZED characters indicate that the node is expected to be
    changed to DT_HALF during graph optimization.

    inp -> loop [ body ] -> out.

    Args:
      inp: A string of letters indicating the colors and expected dtypes of the
        input nodes.
      body: A string of letters indicating the colors and expected dtypes of the
        body nodes.
      out: A string of letters indicating the colors and expected dtypes of the
        output nodes.
    """
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      expected_types = []
      for section in [inp, body, out]:
        section_expected_types = []
        for color in section:
          if color.isupper():
            expected_type = types_pb2.DT_HALF
          else:
            expected_type = types_pb2.DT_FLOAT
          section_expected_types.append(expected_type)
        expected_types.append(section_expected_types)

      a = _build_simple_loop_graph(inp, body, out)
      output_val_ref, output_val, cost_graph = self._run(a)
      node_map = _build_node_map(cost_graph.node)

      section_names = ['input', 'while/body', 'output']
      all_types_correct = True
      for section_name, expected_types in zip(section_names, expected_types):
        for i, expected_type in enumerate(expected_types):
          node_name = section_name + '_%i' % i
          output_port = 0
          optimized_type = node_map[node_name].output_info[output_port].dtype
          if optimized_type != expected_type:
            print('Expected node %s to have type %s but got type %s' %
                  (node_name, expected_type, optimized_type))
            all_types_correct = False
      self.assertTrue(all_types_correct)
      self.assertAllClose(output_val_ref, output_val, atol=2e-3, rtol=1e-3)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_bn(self):
    """Test graph with convolution followed by batch norm."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      x = _conv_bn(x)
      output = _conv_bn(x)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)
      num_to_fp16, num_to_fp32 = _count_casts(cost_graph.node)

      self._assert_output_fp16(node_map, 'Conv2D')
      self._assert_output_fp16(node_map, 'FusedBatchNormV3')
      self._assert_output_fp16(node_map, 'Conv2D_1')
      self.assertEqual(num_to_fp16,
                       3)  # Before Conv2D:0, Conv2D:1, Conv2D_1:1
      self.assertEqual(num_to_fp32, 1)  # After FusedBatchNormV3:0
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  # TODO: enable these tests when cuDNN is upgraded to >= 7.6.2. Same with the
  # test_conv3d() below.
  @unittest.skip('Test case should be skipped when cuDNN < 7.6.2')
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv3d_bn(self):
    """Test graph with convolution followed by batch norm."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 8, 1])
      x = _conv3d_bn(x)
      output = _conv3d_bn(x)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)
      num_to_fp16, num_to_fp32 = _count_casts(cost_graph.node)

      self._assert_output_fp16(node_map, 'Conv3D')
      self._assert_output_fp16(node_map, 'FusedBatchNormV3')
      self._assert_output_fp16(node_map, 'Conv3D_1')
      self.assertEqual(num_to_fp16, 3)  # Before Conv3D:0, Conv3D:1, Conv3D_1:1
      self.assertEqual(num_to_fp32, 1)  # After FusedBatchNormV3:0
      self.assertAllClose(output_val_ref, output_val, atol=1e-2, rtol=1e-2)

  @unittest.skip('Test case should be skipped when cuDNN < 7.6.2')
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv3d(self):
    """Test grad ops with convolution3d graph."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 8, 1])
      f = _weight([3, 3, 3, 1, 6])
      y = _conv3d(x, f)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x, f])
      output = (y, g)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)
      self._assert_output_fp16(node_map, 'Conv3D')
      self._assert_output_fp16(node_map,
                               'gradients/Conv3D_grad/Conv3DBackpropInputV2')
      self._assert_output_fp16(node_map,
                               'gradients/Conv3D_grad/Conv3DBackpropFilterV2')

      output_val_ref, output_val, cost_graph = self._run(output)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_bn_dropout(self):
    """Test dropout precision of convolution batch norm graph."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      y = _conv_bn(x)
      y = nn.dropout(y, rate=0.5)
      y = math_ops.add(y, 1, name='addition')
      y = _conv_bn(y)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(
          learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)
      self._assert_output_fp16(node_map, 'Conv2D')
      self._assert_output_fp16(node_map, 'FusedBatchNormV3')
      # We do not assert dropout's dtype because we do not want to rely on the
      # node names of dropout's internal implementation.
      self._assert_output_fp16(node_map, 'addition')
      self._assert_output_fp16(node_map, 'Conv2D_1')

      output_val_ref, output_val, cost_graph = self._run(output)
      self.assertAllClose(output_val_ref, output_val, atol=2e-3, rtol=2e-3)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_pool(self):
    """Test graph with convolution followed by pooling."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      output = _conv_pool(x)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)
      num_to_fp16, num_to_fp32 = _count_casts(cost_graph.node)

      self._assert_output_fp16(node_map, 'Conv2D')
      self._assert_output_fp16(node_map, 'Relu')
      self._assert_output_fp16(node_map, 'MaxPool')
      self._assert_output_fp16(node_map, 'Conv2D_1')
      self.assertEqual(num_to_fp16, 4)
      self.assertEqual(num_to_fp32, 1)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_simple_loop(self):
    """Test graph with while loop."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y = _simple_loop(x, _matmul_act)[1]
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)

      self._assert_output_fp16(node_map, 'while/MatMul')
      self._assert_output_fp16(node_map, 'while/Relu')
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_loop_with_vars_intertwined(self):
    """Test graph with intertwined while loops."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      _, _, k, l = _loop_vars_intertwined(
          array_ops.ones(array_ops.shape(x)), x, _matmul_act, _matmul_act)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(k, [x])
      output = (k, l, g)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)

      self._assert_output_fp16(node_map, 'while/MatMul')
      self._assert_output_fp16(node_map, 'while/Relu')
      self._assert_output_fp16(node_map, 'while/MatMul_1')
      self._assert_output_fp16(node_map, 'while/Relu_1')
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_multi_paths(self):
    """Test graph with multiple paths."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 3])
      x1, x2, x3 = array_ops.split(x, num_or_size_splits=3, axis=3)
      y1 = _conv_pool(x1)
      y2 = _conv_pool(x2)
      y3 = _conv_pool(x3)
      y = array_ops.concat([y1, y2, y3], axis=3)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)

      self._assert_output_fp16(node_map, 'split')
      for suffix in [''] + ['_%i' % i for i in range(1, 6)]:
        self._assert_output_fp16(node_map, 'Conv2D' + suffix)
        self._assert_output_fp16(node_map, 'Relu' + suffix)
        self._assert_output_fp16(node_map, 'MaxPool' + suffix)
      self._assert_output_fp16(node_map, 'concat')
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_multi_paths_2(self):
    """Test graph with multiple paths."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y1 = _matmul_act(x)
      y2 = _matmul_act(x)
      y = y1 + y2 + x
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (g, y)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)

      self._assert_output_fp16(node_map, 'MatMul')
      self._assert_output_fp16(node_map, 'Relu')
      self._assert_output_fp16(node_map, 'MatMul_1')
      self._assert_output_fp16(node_map, 'Relu_1')
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_recurrent_lstm(self):
    """Test graph with recurrent lstm."""
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      init_c = _input([8, 4])
      init_h = _input([8, 4])
      _, _, h, _ = _recurrent_lstm(init_c, init_h)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(h, [init_c, init_h])
      output = (h, g)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)

      self._assert_output_fp16(node_map, 'while/concat')
      self._assert_output_fp16(node_map, 'while/MatMul')
      self._assert_output_fp16(node_map, 'while/split')
      self._assert_output_fp16(node_map, 'while/Sigmoid')
      self._assert_output_fp16(node_map, 'while/Sigmoid_1')
      self._assert_output_fp16(node_map, 'while/Sigmoid_2')
      self._assert_output_fp16(node_map, 'while/Tanh')
      self._assert_output_fp16(node_map, 'while/Tanh_1')
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_1(self):
    self._run_simple_loop_test('W', 'C', 'C')

  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_2(self):
    self._run_simple_loop_test('C', 'C', 'W')

  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_3(self):
    self._run_simple_loop_test('W', 'G', 'W')

  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_4(self):
    self._run_simple_loop_test('W', 'gbg', 'W')

  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_5(self):
    self._run_simple_loop_test('b', 'gWC', 'c')

  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_6(self):
    self._run_simple_loop_test('b', 'CWCG', 'C')

  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_7(self):
    self._run_simple_loop_test('C', 'GWCG', 'C')

  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_8(self):
    self._run_simple_loop_test('C', 'CgbgWC', 'g')

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_noninlined_funcdef(self):
    """Test graph with non-inlined function subgraph.

    This requires the grappler pass to handle an OpDef that only appears in the
    graph's function registry instead of the global op registry.
    """
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y = _matmul_act(x)
      y = _example_noninlined_funcdef(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (g, y)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)

      self._assert_output_fp16(node_map, 'MatMul')
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_ingraph_train_loop(self):
    """Tests a graph containing a while loop around a training update.

    This requires the grappler pass to take special care with its handling of
    Enter ops that appear in front of reads from non-resource variables. See
    the use of NodeImplicitlyReadsVariable in auto_mixed_precision.cc.
    """
    if tf2.enabled():
      # This test tests non-resource variables, which are only used in TF1.
      self.skipTest('TensorFlow 1 required')
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(1234)
      np.random.seed(1234)
      num_iter, bs, nchan, nclass = 100, 64, 32, 100

      data = np.random.normal(size=(bs * num_iter, nchan)).astype(np.float32)
      labels = np.random.randint(nclass, size=(bs * num_iter,))
      ds = dataset_ops.Dataset.from_tensor_slices((data, labels))
      ds = ds.batch(bs).prefetch(3)
      it = ds.make_one_shot_iterator()

      def body(_, i):
        i += 1
        x, yt = it.get_next()
        dense = layers.Dense(nclass)
        y = dense(x)
        loss = losses.sparse_softmax_cross_entropy(yt, y)
        opt = adam.AdamOptimizer()
        train_op = opt.minimize(loss, var_list=dense.trainable_weights)
        with ops.control_dependencies([train_op]):
          loss = array_ops.identity(loss)
        return loss, i

      begin, end = constant_op.constant(0), constant_op.constant(num_iter)
      loss, _ = control_flow_ops.while_loop(
          lambda loss, i: math_ops.less(i, end), body, [0.0, begin])

      output_val_ref, output_val, cost_graph = self._run(loss)
      node_map = _build_node_map(cost_graph.node)

      self._assert_output_fp16(node_map, 'while/dense/MatMul')
      self._assert_output_fp16(
          node_map, 'while/gradients/while/dense/MatMul_grad/MatMul_1')
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
  test.main()
