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
"""Unit tests for the quantize_graph graph rewriting API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.quantize.python import quantize_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


# TODO(suharshs): Add tests for testing experimental APIs and additional
# input arguments
class QuantizeGraphTest(test_util.TensorFlowTestCase):
  # We have a lot of other tests that test the details of the rewrite, here we
  # just the specific features of the quantize_graph API.

  def _RunTestOverParameters(self, test_fn):
    rewrite_fns = [
        quantize_graph.create_training_graph,
        quantize_graph.create_eval_graph,
        quantize_graph.experimental_create_training_graph,
        quantize_graph.experimental_create_eval_graph,
    ]
    for fn in rewrite_fns:
      test_fn(fn)

  def testRewrite(self):
    self._RunTestOverParameters(self._TestRewrite)

  def _TestRewrite(self, fn):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      inputs = array_ops.zeros((batch_size, height, width, depth))
      conv = layers.conv2d(
          inputs,
          32, [5, 5],
          stride=2,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=None,
          scope='test')
      _ = nn_ops.relu6(conv)

    orig_variable_names = set(
        [v.name for v in graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

    fn(graph)

    q_variables = graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    # Ensure that variables were added.
    self.assertTrue(len(orig_variable_names) < len(q_variables))

  def testDefaultGraph(self):
    self._RunTestOverParameters(self._TestRewrite)

  def _TestDefaultGraph(self, fn):
    with ops.Graph().as_default() as g:
      batch_size, height, width, depth = 5, 128, 128, 3
      inputs = array_ops.zeros((batch_size, height, width, depth))
      conv = layers.conv2d(
          inputs,
          32, [5, 5],
          stride=2,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=None,
          scope='test')
      _ = nn_ops.relu6(conv)

      orig_variable_names = set(
          [v.name for v in g.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

      fn()

      q_variables = g.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      # Ensure that variables were added.
      self.assertTrue(len(orig_variable_names) < len(q_variables))

  def _WeightInit(self, stddev):
    """Returns truncated normal variable initializer.

    Function is defined purely to shorten the name so that it stops wrapping.

    Args:
      stddev: Standard deviation of normal variable.

    Returns:
      An initialized that initialzes with a truncated normal variable.
    """
    return init_ops.truncated_normal_initializer(stddev=stddev)


if __name__ == '__main__':
  googletest.main()
