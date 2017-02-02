# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.contrib.graph_editor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import graph_editor as ge
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RerouteTest(test.TestCase):

  def setUp(self):
    self.graph = ops.Graph()
    with self.graph.as_default():
      self.a0 = constant_op.constant(1.0, shape=[2], name="a0")
      self.b0 = constant_op.constant(2.0, shape=[2], name="b0")
      self.c0 = math_ops.add(self.a0, self.b0, name="c0")
      self.a1 = constant_op.constant(3.0, shape=[2], name="a1")
      self.b1 = constant_op.constant(4.0, shape=[2], name="b1")
      self.c1 = math_ops.add(self.a1, self.b1, name="c1")
      self.a2 = constant_op.constant(3.0, shape=[3], name="a2")
      self.b2 = constant_op.constant(4.0, shape=[3], name="b2")
      self.c2 = math_ops.add(self.a2, self.b2, name="c2")

  def test_swap(self):
    ge.reroute.swap_ts([self.a0, self.b0], [self.a1, self.b1])
    self.assertTrue(ge.matcher("c0").input_ops("a1", "b1")(self.c0.op))
    self.assertTrue(ge.matcher("c1").input_ops("a0", "b0")(self.c1.op))

  def test_multiswap(self):
    with self.graph.as_default():
      a3 = constant_op.constant(3.0, shape=[2], name="a3")
    ge.reroute.swap(
        ge.sgv(a3.op).remap_outputs([0, 0]), ge.sgv(self.a0.op, self.a1.op))
    self.assertTrue(ge.matcher("c0").input_ops("a3", "b0")(self.c0.op))
    self.assertTrue(ge.matcher("c1").input_ops("a3", "b1")(self.c1.op))

  def test_reroute(self):
    ge.reroute.reroute_a2b_ts([self.a0, self.b0], [self.a1, self.b1])
    self.assertTrue(ge.matcher("c0").input_ops("a0", "b0")(self.c0.op))
    self.assertTrue(ge.matcher("c1").input_ops("a0", "b0")(self.c1.op))

    ge.reroute.reroute_b2a_ts([self.a0, self.b0], [self.a1, self.b1])
    self.assertTrue(ge.matcher("c0").input_ops("a1", "b1")(self.c0.op))
    self.assertTrue(ge.matcher("c1").input_ops("a1", "b1")(self.c1.op))

  def test_compatibility(self):
    with self.assertRaises(ValueError):
      ge.reroute.reroute_a2b_ts([self.a0, self.b0], [self.a2, self.b2])

  def test_reroute_can_modify(self):
    graph = ops.Graph()
    # create a special graph where "a" is an ambiguous tensor. That is
    # it is both an input and an output of the ops in sgv0.
    with graph.as_default():
      a = constant_op.constant(1.0, shape=[2], name="a")
      b = constant_op.constant(2.0, shape=[2], name="b")
      c = math_ops.add(a, b, name="c")
      d = math_ops.add(a, c, name="d")

      e = constant_op.constant(1.0, shape=[2], name="e")
      f = constant_op.constant(2.0, shape=[2], name="f")
      g = math_ops.add(e, f, name="g")

    sgv0 = ge.sgv(a.op, b.op, c.op)
    sgv1 = ge.sgv(e.op, f.op)

    ge.reroute.swap_outputs(sgv0, sgv1)
    self.assertTrue(
        ge.matcher("g").input_ops("a", ge.matcher("c")
                                  .input_ops("a", "b"))(g.op))
    self.assertTrue(ge.matcher("d").input_ops("e", "f")(d.op))


if __name__ == "__main__":
  test.main()
