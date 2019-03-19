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

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import rewriter_config_pb2 as rwcpb2
from tensorflow.core.protobuf import config_pb2 as cpb2

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test
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
  b = lambda i, j: (i+1, functor(j))
  ij = control_flow_ops.while_loop(c, b, init)
  return ij



def _loop_vars_intertwined(x0, y0, functor_x, functor_y):
  """Loop whose loop variables are intertwined."""
  c = lambda i, j, x, y: j < 4
  b = lambda i, j, x, y: (j+1, i+1, functor_y(y), functor_x(x))
  init = (constant_op.constant(0), constant_op.constant(0), x0, y0)
  ijzw = control_flow_ops.while_loop(c, b, init)
  return ijzw


def _lstm_cell(prev_c, prev_h, x):
  """ LSTMCell
  i: input gate
  f: forget gate
  o: output gate
  c: cell state
  x: input
  h: embedding
  """
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
  """ Dynamic single-layer LSTM with TensorArray """
  def cond(i, c, h, ta_x):
    del c
    del h
    del ta_x
    return i < 4

  def body(i, c, h, ta_x):
    x = ta_x.read(i)
    next_c, next_h = _lstm_cell(c, h, x)
    return (i+1, next_c, next_h, ta_x)

  ta_x = tensor_array_ops.TensorArray(
      dtype=dtypes.float32,
      size=4)
  for i in range(0, 4):
    ta_x = ta_x.write(
        i, constant_op.constant(0.1, shape=[8, 4],
                                dtype=dtypes.float32))
  init = (constant_op.constant(0), c, h, ta_x)
  r = control_flow_ops.while_loop(cond, body, init)
  return r


def _make_node_with_color(color, input_tensor, name=None):
  """ Returns a node representative of the specified list type """
  color = color.lower()
  if color == 'w': # White node
    weights = _weight(input_tensor.get_shape().as_list())
    return math_ops.matmul(input_tensor, weights, name=name)
  if color == 'g': # Gray node
    return math_ops.sqrt(input_tensor, name=name)
  if color == 'c': # Clear node
    return nn.relu(input_tensor, name=name)
  if color == 'b': # Black node
    return math_ops.log(input_tensor, name=name)
  raise ValueError("Invalid node color: " + str(color))


def _build_intertwined_loop_graph(inp_a_colors, inp_b_colors, body_a_colors,
                                  body_b_colors, out_a_colors, out_b_colors):
  """ Builds a test graph with intertwined loops """
  a = _input([8, 8])
  for i, color in enumerate(inp_a_colors):
    a = _make_node_with_color(color, a, 'inputA_%i' % i)
  b = _input([8, 8])
  for i, color in enumerate(inp_b_colors):
    b = _make_node_with_color(color, b, 'inputB_%i' % i)
  def body_a(x):
    for i, color in enumerate(body_a_colors):
      x = _make_node_with_color(color, x, 'bodyA_%i' % i)
    return x
  def body_b(x):
    for i, color in enumerate(body_b_colors):
      x = _make_node_with_color(color, x, 'bodyB_%i' % i)
    return x
  a, b = _loop_vars_intertwined(a, b, body_a, body_b)[2:]
  for i, color in enumerate(out_a_colors):
    a = _make_node_with_color(color, a, 'outputA_%i' % i)
  for i, color in enumerate(out_b_colors):
    b = _make_node_with_color(color, b, 'outputB_%i' % i)
  a = array_ops.identity(a)
  b = array_ops.identity(b)
  return a, b


def _get_config(auto_mixed_precision=True):
  """ Returns a config object with automatic mixed precision
      activated if appropriate """
  if auto_mixed_precision:
    rewrite_config = rwcpb2.RewriterConfig(
        auto_mixed_precision=rwcpb2.RewriterConfig.ON,
        # do not remove duplicated nodes
        arithmetic_optimization=rwcpb2.RewriterConfig.OFF)
  else:
    rewrite_config = rwcpb2.RewriterConfig(
        auto_mixed_precision=rwcpb2.RewriterConfig.OFF,
        # do not remove duplicated nodes
        arithmetic_optimization=rwcpb2.RewriterConfig.OFF)
  rewrite_config.min_graph_nodes = -1
  graph_options = cpb2.GraphOptions(
      rewrite_options=rewrite_config, build_cost_model=1)
  config = cpb2.ConfigProto(graph_options=graph_options)
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


class AutoMixedPrecisionTest(test.TestCase):
  """Tests the Grappler auto mixed precision optimizer."""
  MIN_GPU_ARCH = (7, 0)

  def _assert_output_fp16(self, node_map, node_name, output_port=0):
    self.assertEqual(node_map[node_name].output_info[output_port].dtype,
                     types_pb2.DT_HALF)

  def _run(self, fetches):
    """ Runs the graph and returns the evaluation of the fetches """
    with session.Session(config=_get_config(False)) as sess:
      sess.run(variables.global_variables_initializer())
      output_val_ref = self.evaluate(fetches)

    with session.Session(config=_get_config()) as sess:
      sess.run(variables.global_variables_initializer())
      metadata = cpb2.RunMetadata()
      output_val = sess.run(fetches, run_metadata=metadata)

    return output_val_ref, output_val, metadata.cost_graph

  def _run_intertwined_loop_test(self, inp_a, inp_b, body_a, body_b, out_a,
                                 out_b, expected_num_to_fp16,
                                 expected_num_to_fp32):
    """Runs a test of an intertwined loop with different node colors in
    different sections of the graph. The arguments must be strings where each
    character represents the color of a node in that section of the graph:
    w = white, g = gray, c = clear, b = black. CAPITALIZED characters indicate
    that the node is expected to be changed to DT_HALF during graph
    optimization.

    inp_a -> loop [ body_a ] -> out_a
               :             |
                ======<<=====
               |             :
    inp_b -> loop [ body_b ] -> out_b
    """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
      random_seed.set_random_seed(0)
      expected_types = []
      for section in [inp_a, inp_b, body_a, body_b, out_a, out_b]:
        section_expected_types = []
        for color in section:
          if color.isupper():
            expected_type = types_pb2.DT_HALF
          else:
            expected_type = types_pb2.DT_FLOAT
          section_expected_types.append(expected_type)
        expected_types.append(section_expected_types)

      a, b = _build_intertwined_loop_graph(inp_a, inp_b, body_a, body_b,
                                           out_a, out_b)
      output_val_ref, output_val, cost_graph = self._run((a, b))
      node_map = _build_node_map(cost_graph.node)
      num_to_fp16, num_to_fp32 = _count_casts(cost_graph.node)

      section_names = ['inputA', 'inputB', 'while/bodyA', 'while/bodyB',
                       'outputA', 'outputB']
      all_types_correct = True
      for section_name, expected_types in zip(section_names, expected_types):
        for i, expected_type in enumerate(expected_types):
          node_name = section_name + '_%i' % i
          output_port = 0
          optimized_type = node_map[node_name].output_info[output_port].dtype
          if optimized_type != expected_type:
            print("Expected node %s to have type %s but got type %s" %
                  (node_name, expected_type, optimized_type))
            all_types_correct = False
      self.assertTrue(all_types_correct)
      self.assertEqual(num_to_fp16, expected_num_to_fp16)
      self.assertEqual(num_to_fp32, expected_num_to_fp32)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  def test_conv_bn(self):
    """ Test graph with convolution followed by batch norm """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      x = _conv_bn(x)
      output = _conv_bn(x)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)
      num_to_fp16, num_to_fp32 = _count_casts(cost_graph.node)

      self._assert_output_fp16(node_map, 'Conv2D')
      self._assert_output_fp16(node_map, 'FusedBatchNorm')
      self._assert_output_fp16(node_map, 'Conv2D_1')
      self.assertEqual(num_to_fp16, 3) # Before Conv2D:0, Conv2D:1, Conv2D_1:1
      self.assertEqual(num_to_fp32, 1) # After FusedBatchNorm:0
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  def test_conv_bn_dropout(self):
    """ Test dropout precision of convolution batch norm graph """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      y = _conv_bn(x)
      y = nn.dropout(y, rate=0.5)
      y = _conv_bn(y)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

      output_val_ref, output_val, cost_graph = self._run(output)
      node_map = _build_node_map(cost_graph.node)
      num_to_fp16, num_to_fp32 = _count_casts(cost_graph.node)
      self._assert_output_fp16(node_map, 'Conv2D')
      self._assert_output_fp16(node_map, 'FusedBatchNorm')
      self._assert_output_fp16(node_map, 'dropout/mul')
      self._assert_output_fp16(node_map, 'dropout/Cast')
      self._assert_output_fp16(node_map, 'dropout/mul_1')
      self._assert_output_fp16(node_map, 'Conv2D_1')

      output_val_ref, output_val, cost_graph = self._run(output)
      self.assertEqual(num_to_fp16, 3) # Before Conv2D:0, Conv2D:1, Conv2D_1:1
      self.assertEqual(num_to_fp32, 1) # After Conv2D_1:0
      self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  def test_conv_pool(self):
    """ Test graph with convolution followed by pooling """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
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

  def test_simple_loop(self):
    """ Test graph with while loop """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
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

  def test_loop_with_vars_intertwined(self):
    """ Test graph with intertwined while loops """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      _, _, k, l = _loop_vars_intertwined(array_ops.ones(array_ops.shape(x)),
                                          x, _matmul_act, _matmul_act)
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

  def test_multi_paths(self):
    """ Test graph with multiple paths """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
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

  def test_multi_paths_2(self):
    """ Test graph with multiple paths """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
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

  def test_recurrent_lstm(self):
    """ Test graph with recurrent lstm """
    if test.is_gpu_available(cuda_only=True,
                             min_cuda_compute_capability=self.MIN_GPU_ARCH):
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

  def test_propagation_through_intertwined_loop_1(self):
    self._run_intertwined_loop_test('C', 'C', 'bgW', 'C', 'g', 'b', 4, 3)

  def test_propagation_through_intertwined_loop_2(self):
    # Note that this results in NextIteration and Merge being painted different
    # colors, requiring NextIteration to be forced to match.
    self._run_intertwined_loop_test('b', 'g', 'gW', 'C', 'c', 'C', 3, 2)

  def test_propagation_through_intertwined_loop_3(self):
    self._run_intertwined_loop_test('g', 'g', 'g', 'g', 'W', 'c', 3, 2)

  def test_propagation_through_intertwined_loop_4(self):
    self._run_intertwined_loop_test('W', 'g', 'g', 'g', 'g', 'g', 3, 2)

  def test_propagation_through_intertwined_loop_5(self):
    self._run_intertwined_loop_test('W', 'c', 'b', 'c', 'c', 'W', 4, 2)

  def test_propagation_through_intertwined_loop_6(self):
    self._run_intertwined_loop_test('b', 'g', 'g', 'g', 'g', 'W', 2, 1)

  def test_propagation_through_intertwined_loop_7(self):
    self._run_intertwined_loop_test('c', 'c', 'bWg', 'c', 'g', 'b', 2, 1)

  def test_propagation_through_intertwined_loop_8(self):
    self._run_intertwined_loop_test('C', 'C', 'C', 'C', 'W', 'g', 3, 2)

  def test_propagation_through_intertwined_loop_9(self):
    self._run_intertwined_loop_test('W', 'g', 'G', 'G', 'g', 'W', 4, 2)

  def test_propagation_through_intertwined_loop_10(self):
    self._run_intertwined_loop_test('g', 'g', 'GWG', 'G', 'g', 'g', 3, 2)

if __name__ == '__main__':
  test.main()
