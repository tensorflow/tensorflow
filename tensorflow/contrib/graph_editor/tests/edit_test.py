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


class EditTest(test.TestCase):
  """edit module test.

  Generally the tests are in two steps:
  - modify an existing graph.
  - then make sure it has the expected topology using the graph matcher.
  """

  def setUp(self):
    self.graph = ops.Graph()
    with self.graph.as_default():
      self.a = constant_op.constant([1., 1.], shape=[2], name="a")
      with ops.name_scope("foo"):
        self.b = constant_op.constant([2., 2.], shape=[2], name="b")
        self.c = math_ops.add(self.a, self.b, name="c")
        self.d = constant_op.constant([3., 3.], shape=[2], name="d")
        with ops.name_scope("bar"):
          self.e = math_ops.add(self.c, self.d, name="e")
          self.f = math_ops.add(self.c, self.d, name="f")
          self.g = math_ops.add(self.c, self.a, name="g")
          with ops.control_dependencies([self.c.op]):
            self.h = math_ops.add(self.f, self.g, name="h")

  def test_detach(self):
    """Test for ge.detach."""
    sgv = ge.sgv(self.c.op, self.a.op)
    control_outputs = ge.ControlOutputs(self.graph)
    ge.detach(sgv, control_ios=control_outputs)
    # make sure the detached graph is as expected.
    self.assertTrue(
        ge.OpMatcher("^foo/c$").input_ops("a", "geph__b_0")(self.c.op))

  def test_connect(self):
    """Test for ge.connect."""
    with self.graph.as_default():
      x = constant_op.constant([1., 1.], shape=[2], name="x")
      y = constant_op.constant([2., 2.], shape=[2], name="y")
      z = math_ops.add(x, y, name="z")

    sgv = ge.sgv(x.op, y.op, z.op)
    ge.connect(sgv, ge.sgv(self.e.op).remap_inputs([0]))
    self.assertTrue(
        ge.OpMatcher("^foo/bar/e$").input_ops("^z$", "foo/d$")(self.e.op))

  def test_bypass(self):
    """Test for ge.bypass."""
    ge.bypass(ge.sgv(self.f.op).remap_inputs([0]))
    self.assertTrue(
        ge.OpMatcher("^foo/bar/h$").input_ops("^foo/c$", "foo/bar/g$")(
            self.h.op))


if __name__ == "__main__":
  test.main()
