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
"""Tests for graph_matcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python import ops as contrib_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.quantize.python import graph_matcher
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class GraphMatcherTest(test_util.TensorFlowTestCase):

  def test_conv_layer(self):
    g = ops.Graph()
    with g.as_default():
      inputs = array_ops.placeholder(dtypes.float32, shape=[8, 5, 5, 3])

    with contrib_ops.arg_scope(
        [layers.batch_norm], fused=True, is_training=True, trainable=True):
      return layers.convolution(
          inputs,
          num_outputs=16,
          kernel_size=3,
          stride=1,
          padding='VALID',
          activation_fn=nn_ops.relu,
          normalizer_fn=layers.batch_norm,
          normalizer_params={},
          weights_initializer=initializers.xavier_initializer(),
          weights_regularizer=None,
          biases_initializer=init_ops.zeros_initializer(),
          biases_regularizer=None,
          reuse=None,
          trainable=True,
          scope=None)

    inputs_pattern = graph_matcher.OpTypePattern('*', name='inputs')
    relu_pattern = graph_matcher.OpTypePattern(
        'Relu',
        name='relu',
        inputs=[
            graph_matcher.OpTypePattern(
                'FusedBatchNorm',
                inputs=[
                    graph_matcher.OpTypePattern(
                        'Conv2D', inputs=[inputs_pattern, '*']), '*', '*', '*',
                    '*'
                ])
        ])
    matcher = graph_matcher.GraphMatcher(relu_pattern)
    match_results = list(matcher.match_graph(g))
    self.assertEqual(1, len(match_results))
    match_result = match_results[0]
    self.assertEqual(match_result.get_tensor(inputs_pattern), inputs)
    self.assertEqual(match_result.get_tensor('inputs'), inputs)

  def test_multiple_outputs(self):
    #   -         +
    #  / \y0   y1/ \
    # x    split    z
    #       |
    #       y         (nodes are ops; edges are going up)
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32, shape=[1], name='x')
      y = array_ops.placeholder(dtypes.float32, shape=[2], name='y')
      y0, y1 = array_ops.split(y, num_or_size_splits=2, axis=0)
      z = array_ops.placeholder(dtypes.float32, shape=[1], name='z')
      math_ops.add(x, y0)
      math_ops.subtract(y1, z)

    y1_pattern = graph_matcher.OpTypePattern('*')
    minus_pattern = graph_matcher.OpTypePattern('Sub', inputs=[y1_pattern, '*'])
    matcher = graph_matcher.GraphMatcher(minus_pattern)

    match_results = list(matcher.match_graph(g))
    self.assertEqual(1, len(match_results))
    match_result = match_results[0]

    self.assertEqual(y0.op, y1.op)
    self.assertEqual(match_result.get_op(y1_pattern), y1.op)
    self.assertEqual(match_result.get_tensor(y1_pattern), y1)

  def test_oneof_type_pattern(self):
    #   -   +
    #  / \ / \
    # x   y   z
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32, shape=[], name='x')
      y = array_ops.placeholder(dtypes.float32, shape=[], name='y')
      z = array_ops.placeholder(dtypes.float32, shape=[], name='z')
      plus = x + y
      minus = y - z

    add_or_sub_pattern = graph_matcher.OpTypePattern(
        'Add|Sub', inputs=['*', '*'])
    matcher = graph_matcher.GraphMatcher(add_or_sub_pattern)
    self.assertEqual([
        match_result.get_op(add_or_sub_pattern)
        for match_result in matcher.match_graph(g)
    ], [plus.op, minus.op])

  def test_oneof_pattern(self):
    reshape_pattern = graph_matcher.OpTypePattern('Reshape')
    transpose_pattern = graph_matcher.OneofPattern([
        graph_matcher.OpTypePattern(
            'Transpose',
            name='transpose',
            inputs=[
                graph_matcher.OpTypePattern(
                    'Slice', name='slice', inputs=[reshape_pattern, '*', '*']),
                '*'
            ]),
        graph_matcher.OpTypePattern(
            'Transpose', name='transpose', inputs=[reshape_pattern, '*'])
    ])

    matcher = graph_matcher.GraphMatcher(transpose_pattern)

    g = ops.Graph()
    with g.as_default():
      inputs = array_ops.placeholder(dtypes.float32, shape=[6])
      reshape = array_ops.reshape(inputs, [2, 3])
      transpose = array_ops.transpose(reshape)
      [match_result] = list(matcher.match_graph(g))
      self.assertEqual(match_result.get_tensor(reshape_pattern), reshape)
      self.assertEqual(match_result.get_tensor('slice'), None)
      self.assertEqual(match_result.get_op('transpose'), transpose.op)

    g = ops.Graph()
    with g.as_default():
      inputs = array_ops.placeholder(dtypes.float32, shape=[6])
      reshape = array_ops.reshape(inputs, [2, 3])
      slicing = array_ops.slice(reshape, [0, 0], [-1, -1])
      transpose = array_ops.transpose(slicing)
      [match_result] = list(matcher.match_graph(g))
      self.assertEqual(match_result.get_tensor(reshape_pattern), reshape)
      self.assertEqual(match_result.get_tensor('slice'), slicing)
      self.assertEqual(match_result.get_op('transpose'), transpose.op)


if __name__ == '__main__':
  googletest.main()
