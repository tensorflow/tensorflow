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

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib


class FreezeGraphTest(test_util.TensorFlowTestCase):

  def _testFreezeGraph(self, saver_write_version):

    checkpoint_prefix = os.path.join(self.get_temp_dir(), "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"

    # We'll create an input graph that has a single variable containing 1.0,
    # and that then multiplies it by 2.
    with ops.Graph().as_default():
      variable_node = variables.Variable(1.0, name="variable_node")
      output_node = math_ops.multiply(variable_node, 2.0, name="output_node")
      sess = session.Session()
      init = variables.global_variables_initializer()
      sess.run(init)
      output = sess.run(output_node)
      self.assertNear(2.0, output, 0.00001)
      saver = saver_lib.Saver(write_version=saver_write_version)
      checkpoint_path = saver.save(
          sess,
          checkpoint_prefix,
          global_step=0,
          latest_filename=checkpoint_state_name)
      graph_io.write_graph(sess.graph, self.get_temp_dir(), input_graph_name)

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "output_node"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)
    clear_devices = False

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, "")

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with ops.Graph().as_default():
      output_graph_def = graph_pb2.GraphDef()
      with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = importer.import_graph_def(output_graph_def, name="")

      self.assertEqual(4, len(output_graph_def.node))
      for node in output_graph_def.node:
        self.assertNotEqual("VariableV2", node.op)
        self.assertNotEqual("Variable", node.op)

      with session.Session() as sess:
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = sess.run(output_node)
        self.assertNear(2.0, output, 0.00001)

  def testFreezeGraphV1(self):
    self._testFreezeGraph(saver_pb2.SaverDef.V1)

  def testFreezeGraphV2(self):
    self._testFreezeGraph(saver_pb2.SaverDef.V2)


if __name__ == "__main__":
  test.main()
