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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.grappler import model_analyzer
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class PyWrapOptimizeGraphTest(test.TestCase):

  def testBasic(self):
    """Make sure arguments can be passed correctly."""
    a = constant_op.constant([10, 11], name="a")
    b = constant_op.constant([10], name="b")
    c = math_ops.add(a, b, name="c")
    d = math_ops.add_n([a, c], name="d")
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    report = model_analyzer.GenerateModelReport(mg)

    # Check the report headers
    self.assertTrue(b"a [Const]" in report)
    self.assertTrue(b"a [Const]" in report)
    self.assertTrue(b"c [Add]" in report)
    self.assertTrue(b"d [AddN]" in report)

    # Also print the report to make it easier to debug
    print("{}".format(report))

  def testDebugMode(self):
    """Make sure arguments can be passed correctly."""
    a = constant_op.constant([10, 11], name="a")
    b = constant_op.constant([10], name="b")
    c = math_ops.add(a, b, name="c")
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(c)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    report = model_analyzer.GenerateModelReport(mg, debug=True)

    # Check the report headers
    self.assertTrue(b"input 0 (int32) has known value" in report)
    self.assertTrue(b"input 1 (int32) has known value" in report)

    # Also print the report to make it easier to debug
    print("{}".format(report))


if __name__ == "__main__":
  test.main()
