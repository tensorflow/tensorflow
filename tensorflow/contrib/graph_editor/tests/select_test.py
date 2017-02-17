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

import re

from tensorflow.contrib import graph_editor as ge
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class SelectTest(test.TestCase):

  def setUp(self):
    self.graph = ops_lib.Graph()
    with self.graph.as_default():
      self.a = constant_op.constant([1., 1.], shape=[2], name="a")
      with ops_lib.name_scope("foo"):
        self.b = constant_op.constant([2., 2.], shape=[2], name="b")
        self.c = math_ops.add(self.a, self.b, name="c")
        self.d = constant_op.constant([3., 3.], shape=[2], name="d")
        with ops_lib.name_scope("bar"):
          self.e = math_ops.add(self.c, self.d, name="e")
          self.f = math_ops.add(self.c, self.d, name="f")
          self.g = math_ops.add(self.c, self.a, name="g")
          with ops_lib.control_dependencies([self.c.op]):
            self.h = math_ops.add(self.f, self.g, name="h")

  def test_regex(self):
    """Test for ge.can_be_regex and ge.make_regex."""
    self.assertTrue(ge.can_be_regex("foo"))
    self.assertTrue(ge.can_be_regex(re.compile("foo")))
    regex = re.compile("foo")
    self.assertIs(ge.make_regex(regex), regex)

  def test_get_input_output_ts(self):
    """Test for ge._get_input_ts abd ge._get_output_ts."""
    self.assertEqual(len(ge.select._get_input_ts(self.graph)), 6)
    self.assertEqual(len(ge.select._get_output_ts(self.graph)), 8)

  def test_get_filter(self):
    """Test for various filtering operations on ts ops."""
    # TODO(fkp): parameterise
    self.assertEqual(len(ge.filter_ops(self.graph, True)), 8)
    self.assertEqual(
        len(ge.filter_ops(self.graph, lambda op: op.node_def.op == "Const")), 3)
    self.assertEqual(
        len(ge.filter_ops(self.graph, lambda op: op.node_def.op == "Add")), 5)
    self.assertEqual(
        len(ge.filter_ops_from_regex(self.graph, r"^.*\b[abc]$")), 3)

    self.assertEqual(len(ge.filter_ts(self.graph, True)), 8)
    self.assertEqual(
        len(ge.filter_ts_from_regex(self.graph, r"^.*/[fgh]:\d$")), 3)

    self.assertEqual(len(ge.get_name_scope_ops(self.graph, "foo/")), 7)
    self.assertEqual(len(ge.get_name_scope_ops(self.graph, "foo/bar")), 4)

  def test_get_ops_ios(self):
    """Test for ge.get_ops_ios."""
    control_outputs = ge.util.ControlOutputs(self.graph)
    self.assertEqual(
        len(ge.get_ops_ios(
            self.h.op, control_ios=control_outputs)), 3)
    self.assertEqual(len(ge.get_ops_ios(self.h.op)), 2)
    self.assertEqual(
        len(ge.get_ops_ios(
            self.c.op, control_ios=control_outputs)), 6)
    self.assertEqual(len(ge.get_ops_ios(self.c.op)), 5)

  def test_compute_boundary_ts_0(self):
    """Test for ge.compute_boundary_ts."""
    input_ts, output_ts, inside_ts = ge.compute_boundary_ts(self.g.op)
    self.assertEqual(list(input_ts), [self.c, self.a])
    self.assertEqual(list(output_ts), [self.g])
    self.assertEqual(list(inside_ts), [])

  def test_compute_boundary_ts_1(self):
    """Test for ge.compute_boundary_ts."""
    input_ts, output_ts, inside_ts = ge.compute_boundary_ts(
        [self.g.op, self.h.op])
    self.assertEqual(list(input_ts), [self.c, self.a, self.f])
    self.assertEqual(list(output_ts), [self.h])
    self.assertEqual(list(inside_ts), [self.g])

  def test_compute_boundary_ts_2(self):
    """Test for ge.compute_boundary_ts."""
    graph = ops_lib.Graph()
    with graph.as_default():
      a = constant_op.constant(1, name="a")
      b = constant_op.constant(1, name="b")
      c = math_ops.add(a, b, name="c")
      _ = a + c
    input_ts, output_ts, inside_ts = ge.compute_boundary_ts([a.op, c.op])
    self.assertEqual(list(input_ts), [b])
    self.assertEqual(list(output_ts), [a, c])
    self.assertEqual(list(inside_ts), [a])

  def test_get_within_boundary_ops_0(self):
    """Test for test_get_within_boundary_ops."""
    control_outputs = ge.util.ControlOutputs(self.graph)
    ops = ge.get_within_boundary_ops(
        ops=self.graph,
        seed_ops=self.f.op,
        boundary_ops=[self.c.op, self.h.op],
        inclusive=False,
        control_ios=control_outputs)
    self.assertEqual(len(ops), 3)

  def test_get_within_boundary_ops_1(self):
    """Test for ge.test_get_within_boundary_ops."""
    ops = ge.get_within_boundary_ops(
        ops=self.graph, seed_ops=self.h.op, boundary_ops=[self.f.op, self.g.op])
    self.assertEqual(len(ops), 3)

  def test_get_walks_intersection(self):
    """Test for ge.get_walks_intersection_ops."""
    ops = ge.get_walks_intersection_ops([self.c.op], [self.g.op])
    self.assertEqual(len(ops), 2)

  def test_get_walks_union(self):
    """Test for ge.get_walks_union_ops."""
    ops = ge.get_walks_union_ops([self.f.op], [self.g.op])
    self.assertEqual(len(ops), 6)

  def test_select_ops(self):
    parameters = (
        (("^foo/",), 7),
        (("^foo/bar/",), 4),
        (("^foo/bar/", "a"), 5),)
    for param, length in parameters:
      ops = ge.select_ops(*param, graph=self.graph)
      self.assertEqual(len(ops), length)

  def test_select_ts(self):
    parameters = (
        (".*:0", 8),
        (r".*/bar/\w+:0", 4),)
    for regex, length in parameters:
      ts = ge.select_ts(regex, graph=self.graph)
      self.assertEqual(len(ts), length)

  def test_select_ops_and_ts(self):
    parameters = (
        (("^foo/.*",), 7, 0),
        (("^foo/.*", "(?#ts)^foo/bar/.*"), 7, 4),)
    for param, l0, l1 in parameters:
      ops, ts = ge.select_ops_and_ts(*param, graph=self.graph)
      self.assertEqual(len(ops), l0)
      self.assertEqual(len(ts), l1)


if __name__ == "__main__":
  test.main()
