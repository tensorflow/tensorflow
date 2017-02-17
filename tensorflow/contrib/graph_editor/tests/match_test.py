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


class MatchTest(test.TestCase):

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

  def test_simple_match(self):
    self.assertTrue(ge.OpMatcher("^.*/f$")(self.f.op))
    self.assertTrue(
        ge.OpMatcher("^.*/f$").input_ops("^.*/c$", "^.*/d$")(self.f.op))
    self.assertTrue(ge.OpMatcher("^.*/f$").input_ops(True, "^.*/d$")(self.f.op))
    self.assertTrue(
        ge.OpMatcher("^.*/f$").input_ops(
            ge.match.op_type("Add"), ge.match.op_type("Const"))(self.f.op))
    self.assertTrue(
        ge.OpMatcher("^.*/f$").input_ops("^.*/c$", "^.*/d$")
        .output_ops(ge.OpMatcher("^.*/h$")
                    .control_input_ops("^.*/c$"))(self.f.op))
    self.assertTrue(
        ge.OpMatcher("^.*/f$").input_ops("^.*/c$", "^.*/d$").output_ops(
            ge.OpMatcher("^.*/h$").control_input_ops("^.*/c$")
            .output_ops([]))(self.f.op))


if __name__ == "__main__":
  test.main()
