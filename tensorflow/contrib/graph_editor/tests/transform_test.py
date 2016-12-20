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
import tensorflow as tf
from tensorflow.contrib import graph_editor as ge

# Precision tolerance for floating-point value tests.
ERROR_TOLERANCE = 1e-3


class TransformTest(tf.test.TestCase):

  def setUp(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      c0 = tf.constant(1.0, shape=[10], name="Const")
      c1 = tf.constant(1.0, shape=[10], name="Const")
      c2 = tf.constant(1.0, shape=[10], name="Const")
      i = tf.constant(1.0, shape=[10], name="Input")
      self.o = tf.add(c2, tf.add(c1, tf.add(c0, i)))

  def test_copy(self):
    graph = tf.Graph()
    _, info = ge.copy(self.graph, graph)
    self.assertEqual(set(op.name for op in self.graph.get_operations()),
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
    tf.reset_default_graph()
    a = tf.constant(1)
    b = tf.constant(1)
    eq = tf.equal(a, b)
    assert_op = tf.Assert(eq, [a, b])
    with tf.control_dependencies([assert_op]):
      _ = tf.add(a, b)
    sgv = ge.make_view([assert_op, eq.op, a.op, b.op])
    copier = ge.Transformer()
    copied_sgv, info = copier(sgv, sgv.graph, "", "")
    new_assert_op = info.transformed(assert_op)
    self.assertIsNotNone(new_assert_op)

  def test_transform(self):
    transformer = ge.Transformer()
    def my_transform_op_handler(info, op):
      add_noise = op.name.startswith("Add")
      op_ = ge.transform.copy_op_handler(info, op)
      if add_noise:
        # add some noise to op
        with info.graph_.as_default():
          t_ = tf.add(tf.constant(1.0, shape=[10], name="Noise"),
                      op_.outputs[0], name="AddNoise")
        # return the "noisy" op
        return t_.op
      else:
        return op_
    transformer.transform_op_handler = my_transform_op_handler

    graph = tf.Graph()
    transformer(self.graph, graph, "", "")
    matcher0 = ge.matcher("AddNoise").input_ops(
        "Noise", ge.matcher("Add").input_ops("Const", "Input"))
    matcher1 = ge.matcher("AddNoise_1").input_ops(
        "Noise_1", ge.matcher("Add_1").input_ops("Const_1", matcher0))
    matcher2 = ge.matcher("AddNoise_2").input_ops(
        "Noise_2", ge.matcher("Add_2").input_ops("Const_2", matcher1))
    top = ge.select_ops("^AddNoise_2$", graph=graph)[0]
    self.assertTrue(matcher2(top))

  def test_transform_in_place(self):
    transformer = ge.Transformer()
    def my_transform_op_handler_in_place(info, op):
      add_noise = op.name.startswith("Add")
      op = ge.transform.transform_op_in_place(info, op,
                                              detach_outputs=add_noise)
      if add_noise:
        # add some noise to op
        with info.graph_.as_default():
          t = tf.add(tf.constant(1.0, shape=[10], name="Noise"), op.outputs[0],
                     name="AddNoise")
        # return the "noisy" op
        return t.op
      else:
        return op
    transformer.transform_op_handler = my_transform_op_handler_in_place

    transformer(self.graph, self.graph, "", "")
    matcher0 = ge.matcher("AddNoise").input_ops(
        "Noise", ge.matcher("Add").input_ops("Const", "Input"))
    matcher1 = ge.matcher("AddNoise_1").input_ops(
        "Noise_1", ge.matcher("Add_1").input_ops("Const_1", matcher0))
    matcher2 = ge.matcher("AddNoise_2").input_ops(
        "Noise_2", ge.matcher("Add_2").input_ops("Const_2", matcher1))
    top = ge.select_ops("^AddNoise_2$", graph=self.graph)[0]
    self.assertTrue(matcher2(top))

  def test_copy_with_input_replacements(self):
    with self.graph.as_default():
      ten = tf.constant(10.0, shape=[10], name="Input")
      sgv, _ = ge.copy_with_input_replacements(self.o.op,
                                               {self.o.op.inputs[1]: ten})
      with tf.Session() as sess:
        val = sess.run(sgv.outputs[0])
      self.assertNear(np.linalg.norm(val - np.array([11])),
                      0.0, ERROR_TOLERANCE)

  def test_graph_replace(self):
    tf.reset_default_graph()
    a = tf.constant(1.0, name="a")
    b = tf.Variable(1.0, name="b")
    eps = tf.constant(0.001, name="eps")
    c = tf.identity(a + b + eps, name="c")
    a_new = tf.constant(2.0, name="a_new")
    c_new = ge.graph_replace(c, {a: a_new})
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      c_val, c_new_val = sess.run([c, c_new])
    self.assertNear(c_val, 2.001, ERROR_TOLERANCE)
    self.assertNear(c_new_val, 3.001, ERROR_TOLERANCE)

  def test_graph_replace_dict(self):
    tf.reset_default_graph()
    a = tf.constant(1.0, name="a")
    b = tf.Variable(1.0, name="b")
    eps = tf.constant(0.001, name="eps")
    c = tf.identity(a + b + eps, name="c")
    a_new = tf.constant(2.0, name="a_new")
    c_new = ge.graph_replace({"c": c}, {a: a_new})
    self.assertTrue(isinstance(c_new, dict))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      c_val, c_new_val = sess.run([c, c_new])
    self.assertTrue(isinstance(c_new_val, dict))
    self.assertNear(c_val, 2.001, ERROR_TOLERANCE)
    self.assertNear(c_new_val["c"], 3.001, ERROR_TOLERANCE)

  def test_graph_replace_ordered_dict(self):
    tf.reset_default_graph()
    a = tf.constant(1.0, name="a")
    b = tf.Variable(1.0, name="b")
    eps = tf.constant(0.001, name="eps")
    c = tf.identity(a + b + eps, name="c")
    a_new = tf.constant(2.0, name="a_new")
    c_new = ge.graph_replace(collections.OrderedDict({"c": c}), {a: a_new})
    self.assertTrue(isinstance(c_new, collections.OrderedDict))

  def test_graph_replace_named_tuple(self):
    tf.reset_default_graph()
    a = tf.constant(1.0, name="a")
    b = tf.Variable(1.0, name="b")
    eps = tf.constant(0.001, name="eps")
    c = tf.identity(a + b + eps, name="c")
    a_new = tf.constant(2.0, name="a_new")
    one_tensor = collections.namedtuple("OneTensor", ["t"])
    c_new = ge.graph_replace(one_tensor(c), {a: a_new})
    self.assertTrue(isinstance(c_new, one_tensor))

  def test_graph_replace_missing(self):
    tf.reset_default_graph()
    a = tf.constant(1.0, name="a")
    b = tf.constant(2.0, name="b")
    c = a + 2 * b
    d = tf.constant(2.0, name="d")
    res = ge.graph_replace([b, c], {a: d})
    self.assertEqual(res[0].name, "b:0")
    self.assertEqual(res[1].name, "add_1:0")


if __name__ == "__main__":
  tf.test.main()
