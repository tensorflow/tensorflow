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

import collections
import numpy as np
from tensorflow.contrib import graph_editor as ge
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# Precision tolerance for floating-point value tests.
ERROR_TOLERANCE = 1e-3


class TransformTest(test.TestCase):

  def setUp(self):
    self.graph = ops.Graph()
    with self.graph.as_default():
      c0 = constant_op.constant(1.0, shape=[10], name="Const")
      c1 = constant_op.constant(1.0, shape=[10], name="Const")
      c2 = constant_op.constant(1.0, shape=[10], name="Const")
      i = constant_op.constant(1.0, shape=[10], name="Input")
      self.o = math_ops.add(c2, math_ops.add(c1, math_ops.add(c0, i)))

  def test_copy(self):
    graph = ops.Graph()
    _, info = ge.copy(self.graph, graph)
    self.assertEqual(
        set(op.name for op in self.graph.get_operations()),
        set(op.name for op in graph.get_operations()))
    src_ops = self.graph.get_operations()
    dst_ops = graph.get_operations()
    for op in src_ops:
      op_ = info.transformed(op)
      self.assertTrue(op_ in dst_ops)
      self.assertEqual(op.name, op_.name)
      self.assertEqual(info.original(op_), op)
    src_ts = ge.util.get_tensors(self.graph)
    dst_ts = ge.util.get_tensors(graph)
    for t in src_ts:
      t_ = info.transformed(t)
      self.assertTrue(t_ in dst_ts)
      self.assertEqual(t.name, t_.name)
      self.assertEqual(info.original(t_), t)

  def test_copy_assert(self):
    ops.reset_default_graph()
    a = constant_op.constant(1)
    b = constant_op.constant(1)
    eq = math_ops.equal(a, b)
    assert_op = control_flow_ops.Assert(eq, [a, b])
    with ops.control_dependencies([assert_op]):
      _ = math_ops.add(a, b)
    sgv = ge.make_view([assert_op, eq.op, a.op, b.op])
    copier = ge.Transformer()
    _, info = copier(sgv, sgv.graph, "", "")
    new_assert_op = info.transformed(assert_op)
    self.assertIsNotNone(new_assert_op)

  def test_transform(self):
    transformer = ge.Transformer()

    def my_transform_op_handler(info, op):
      add_noise = op.name.startswith("Add")
      op_, op_outputs_ = ge.transform.copy_op_handler(info, op)
      if not add_noise:
        return op_, op_outputs_
      # add some noise to op
      with info.graph_.as_default():
        t_ = math_ops.add(
            constant_op.constant(1.0, shape=[10], name="Noise"),
            op_.outputs[0],
            name="AddNoise")
      # return the "noisy" op
      return op_, [t_]

    transformer.transform_op_handler = my_transform_op_handler

    graph = ops.Graph()
    transformer(self.graph, graph, "", "")
    matcher0 = ge.OpMatcher("AddNoise").input_ops(
        "Noise", ge.OpMatcher("Add").input_ops("Const", "Input"))
    matcher1 = ge.OpMatcher("AddNoise_1").input_ops(
        "Noise_1", ge.OpMatcher("Add_1").input_ops("Const_1", matcher0))
    matcher2 = ge.OpMatcher("AddNoise_2").input_ops(
        "Noise_2", ge.OpMatcher("Add_2").input_ops("Const_2", matcher1))
    top = ge.select_ops("^AddNoise_2$", graph=graph)[0]
    self.assertTrue(matcher2(top))

  def test_copy_with_input_replacements(self):
    with self.graph.as_default():
      ten = constant_op.constant(10.0, shape=[10], name="Input")
      sgv, _ = ge.copy_with_input_replacements(self.o.op,
                                               {self.o.op.inputs[1]: ten})
      with session.Session() as sess:
        val = sess.run(sgv.outputs[0])
      self.assertNear(
          np.linalg.norm(val - np.array([11])), 0.0, ERROR_TOLERANCE)

  def test_graph_replace(self):
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = variables.Variable(1.0, name="b")
    eps = constant_op.constant(0.001, name="eps")
    c = array_ops.identity(a + b + eps, name="c")
    a_new = constant_op.constant(2.0, name="a_new")
    c_new = ge.graph_replace(c, {a: a_new})
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      c_val, c_new_val = sess.run([c, c_new])
    self.assertNear(c_val, 2.001, ERROR_TOLERANCE)
    self.assertNear(c_new_val, 3.001, ERROR_TOLERANCE)

  def test_graph_replace_dict(self):
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = variables.Variable(1.0, name="b")
    eps = constant_op.constant(0.001, name="eps")
    c = array_ops.identity(a + b + eps, name="c")
    a_new = constant_op.constant(2.0, name="a_new")
    c_new = ge.graph_replace({"c": c}, {a: a_new})
    self.assertTrue(isinstance(c_new, dict))
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      c_val, c_new_val = sess.run([c, c_new])
    self.assertTrue(isinstance(c_new_val, dict))
    self.assertNear(c_val, 2.001, ERROR_TOLERANCE)
    self.assertNear(c_new_val["c"], 3.001, ERROR_TOLERANCE)

  def test_graph_replace_ordered_dict(self):
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = variables.Variable(1.0, name="b")
    eps = constant_op.constant(0.001, name="eps")
    c = array_ops.identity(a + b + eps, name="c")
    a_new = constant_op.constant(2.0, name="a_new")
    c_new = ge.graph_replace(collections.OrderedDict({"c": c}), {a: a_new})
    self.assertTrue(isinstance(c_new, collections.OrderedDict))

  def test_graph_replace_named_tuple(self):
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = variables.Variable(1.0, name="b")
    eps = constant_op.constant(0.001, name="eps")
    c = array_ops.identity(a + b + eps, name="c")
    a_new = constant_op.constant(2.0, name="a_new")
    one_tensor = collections.namedtuple("OneTensor", ["t"])
    c_new = ge.graph_replace(one_tensor(c), {a: a_new})
    self.assertTrue(isinstance(c_new, one_tensor))

  def test_graph_replace_missing(self):
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = constant_op.constant(2.0, name="b")
    c = a + 2 * b
    d = constant_op.constant(2.0, name="d")
    res = ge.graph_replace([b, c], {a: d})
    self.assertEqual(res[0].name, "b:0")
    self.assertEqual(res[1].name, "add_1:0")


if __name__ == "__main__":
  test.main()
