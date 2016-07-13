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

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge


class SubgraphTest(tf.test.TestCase):

  def setUp(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.a = tf.constant([1., 1.], shape=[2], name="a")
      with tf.name_scope("foo"):
        self.b = tf.constant([2., 2.], shape=[2], name="b")
        self.c = tf.add(self.a, self.b, name="c")
        self.d = tf.constant([3., 3.], shape=[2], name="d")
        with tf.name_scope("bar"):
          self.e = tf.add(self.c, self.d, name="e")
          self.f = tf.add(self.c, self.d, name="f")
          self.g = tf.add(self.c, self.a, name="g")
          with tf.control_dependencies([self.c.op]):
            self.h = tf.add(self.f, self.g, name="h")

  def test_subgraph(self):
    sgv = ge.sgv(self.graph)
    self.assertEqual(list(sgv.outputs), [self.e, self.h])
    self.assertEqual(list(sgv.inputs), [])
    self.assertEqual(len(sgv.ops), 8)

    sgv = ge.sgv(self.f.op, self.g.op)
    self.assertEqual(list(sgv.outputs), [self.f, self.g])
    self.assertEqual(list(sgv.inputs), [self.c, self.d, self.a])

    sgv = ge.sgv_scope("foo/bar", graph=self.graph)
    self.assertEqual(list(sgv.ops),
                     [self.e.op, self.f.op, self.g.op, self.h.op])

  def test_subgraph_remap(self):
    sgv = ge.sgv(self.c.op)
    self.assertEqual(list(sgv.outputs), [self.c])
    self.assertEqual(list(sgv.inputs), [self.a, self.b])

    sgv = sgv.remap_outputs_to_consumers()
    self.assertEqual(list(sgv.outputs), [self.c, self.c, self.c])
    sgv = sgv.remap_outputs_make_unique()
    self.assertEqual(list(sgv.outputs), [self.c])

    sgv = sgv.remap(new_input_indices=[], new_output_indices=[])
    self.assertEqual(len(sgv.inputs), 0)
    self.assertEqual(len(sgv.outputs), 0)
    sgv = sgv.remap_default()
    self.assertEqual(list(sgv.outputs), [self.c])
    self.assertEqual(list(sgv.inputs), [self.a, self.b])

  def test_remove_unused_ops(self):
    sgv = ge.sgv(self.graph)
    self.assertEqual(list(sgv.outputs), [self.e, self.h])
    self.assertEqual(len(sgv.ops), 8)

    sgv = sgv.remap_outputs(new_output_indices=[1]).remove_unused_ops()
    self.assertEqual(list(sgv.outputs), [self.h])
    self.assertEqual(len(sgv.ops), 7)


if __name__ == "__main__":
  tf.test.main()
