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
"""Tests for the cost analyzer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import cost_analyzer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam


class CostAnalysisTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBasicCost(self):
    """Make sure arguments can be passed correctly."""
    a = constant_op.constant(10, name="a")
    b = constant_op.constant(20, name="b")
    c = math_ops.add_n([a, b], name="c")
    d = math_ops.add_n([b, c], name="d")
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    report = cost_analyzer.GenerateCostReport(mg, per_node_report=True)

    # Check the report headers
    self.assertTrue(b"Total time measured in ns (serialized):" in report)
    self.assertTrue(b"Total time measured in ns (actual):" in report)
    self.assertTrue(b"Total time analytical in ns (upper bound):" in report)
    self.assertTrue(b"Total time analytical in ns (lower bound):" in report)
    self.assertTrue(b"Overall efficiency (analytical upper/actual):" in report)
    self.assertTrue(b"Overall efficiency (analytical lower/actual):" in report)
    self.assertTrue(b"Below is the per-node report summary:" in report)

    # Also print the report to make it easier to debug
    print("{}".format(report))

  @test_util.run_deprecated_v1
  def testVerbose(self):
    """Make sure the full report is generated with verbose=True."""
    a = constant_op.constant(10, name="a")
    b = constant_op.constant(20, name="b")
    c = math_ops.add_n([a, b], name="c")
    d = math_ops.add_n([b, c], name="d")
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    report = cost_analyzer.GenerateCostReport(
        mg, per_node_report=True, verbose=True)

    # Check the report headers
    self.assertTrue(b"Below is the full per-node report:" in report)

    # Also print the report to make it easier to debug
    print("{}".format(report))

  @test_util.run_deprecated_v1
  def testSmallNetworkCost(self):
    image = array_ops.placeholder(dtypes.float32, shape=[1, 28, 28, 1])
    label = array_ops.placeholder(dtypes.float32, shape=[1, 10])
    w = variables.Variable(
        random_ops.truncated_normal([5, 5, 1, 32], stddev=0.1))
    b = variables.Variable(random_ops.truncated_normal([32], stddev=0.1))
    conv = nn_ops.conv2d(image, w, strides=[1, 1, 1, 1], padding="SAME")
    h_conv = nn_ops.relu(conv + b)
    h_conv_flat = array_ops.reshape(h_conv, [1, -1])

    w_fc = variables.Variable(
        random_ops.truncated_normal([25088, 10], stddev=0.1))
    b_fc = variables.Variable(random_ops.truncated_normal([10], stddev=0.1))
    y_conv = nn_ops.softmax(math_ops.matmul(h_conv_flat, w_fc) + b_fc)

    cross_entropy = math_ops.reduce_mean(
        -math_ops.reduce_sum(label * math_ops.log(y_conv), axis=[1]))
    _ = adam.AdamOptimizer(1e-4).minimize(cross_entropy)

    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())
    report = cost_analyzer.GenerateCostReport(mg)

    # Print the report to make it easier to debug
    print("{}".format(report))

    self.assertTrue(b"MatMul" in report)
    self.assertTrue(b"ApplyAdam" in report)
    self.assertTrue(b"Conv2D" in report)
    self.assertTrue(b"Conv2DBackpropFilter" in report)
    self.assertTrue(b"Softmax" in report)

    for op_type in [b"MatMul", b"Conv2D", b"Conv2DBackpropFilter"]:
      matcher = re.compile(
          br"\s+" + op_type + br",\s*(\d+),\s*(\d+),\s*([\d\.eE+-]+)%,\s*" +
          br"([\d\.eE+-]+)%,\s*(-?\d+),\s*(\d+),", re.MULTILINE)
      m = matcher.search(report)

      op_count = int(m.group(1))
      # upper = int(m.group(5))
      lower = int(m.group(6))
      if op_type == b"MatMul":
        self.assertEqual(3, op_count)
      else:
        self.assertEqual(1, op_count)
      self.assertTrue(0 <= lower)
      # self.assertTrue(0 < upper)
      # self.assertTrue(lower <= upper)

  @test_util.run_deprecated_v1
  def testBasicMemory(self):
    """Make sure arguments can be passed correctly."""
    with test_util.device(use_gpu=False):
      a = constant_op.constant(10, name="a")
      b = constant_op.constant(20, name="b")
      c = math_ops.add_n([a, b], name="c")
      d = math_ops.add_n([b, c], name="d")
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(d)
      mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    report = cost_analyzer.GenerateMemoryReport(mg)

    # Print the report to make it easier to debug
    print("{}".format(report))

    # Check the report
    self.assertTrue(
        "Peak usage for device /job:localhost/replica:0/task:0/device:CPU:0: "
        "16 bytes"
        in report)
    self.assertTrue("  a:0 uses 4 bytes" in report)
    self.assertTrue("  b:0 uses 4 bytes" in report)
    self.assertTrue("  c:0 uses 4 bytes" in report)
    self.assertTrue("  d:0 uses 4 bytes" in report)


if __name__ == "__main__":
  test.main()
