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
"""Tests the graph freezing tool."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.tools import strip_unused


class FreezeGraphTest(test_util.TensorFlowTestCase):

  def testFreezeGraph(self):
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"

    # We'll create an input graph that has a single constant containing 1.0,
    # and that then multiplies it by 2.
    with tf.Graph().as_default():
      constant_node = tf.constant(1.0, name="constant_node")
      wanted_input_node = tf.sub(constant_node, 3.0, name="wanted_input_node")
      output_node = tf.mul(wanted_input_node, 2.0, name="output_node")
      tf.add(output_node, 2.0, name="later_node")
      sess = tf.Session()
      output = sess.run(output_node)
      self.assertNear(-4.0, output, 0.00001)
      tf.train.write_graph(sess.graph.as_graph_def(), self.get_temp_dir(),
                           input_graph_name)

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
    input_binary = False
    input_node_names = "wanted_input_node"
    output_node_names = "output_node"
    output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)

    strip_unused.strip_unused(input_graph_path, input_binary, output_graph_path,
                              input_node_names, output_node_names,
                              tf.float32.as_datatype_enum)

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with tf.Graph().as_default():
      output_graph_def = tf.GraphDef()
      with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

      self.assertEqual(3, len(output_graph_def.node))
      for node in output_graph_def.node:
        self.assertNotEqual("Add", node.op)
        self.assertNotEqual("Sub", node.op)

      with tf.Session() as sess:
        input_node = sess.graph.get_tensor_by_name("wanted_input_node:0")
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = sess.run(output_node, feed_dict={input_node: [10.0]})
        self.assertNear(20.0, output, 0.00001)

if __name__ == "__main__":
  tf.test.main()
