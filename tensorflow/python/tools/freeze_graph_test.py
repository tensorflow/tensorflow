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
import re

from tensorflow.core.example import example_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib


class FreezeGraphTest(test_util.TensorFlowTestCase):

  def _testFreezeGraph(self, saver_write_version):

    checkpoint_prefix = os.path.join(self.get_temp_dir(), "saved_checkpoint")
    checkpoint_meta_graph_file = os.path.join(self.get_temp_dir(),
                                              "saved_checkpoint.meta")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"

    # We'll create an input graph that has a single variable containing 1.0,
    # and that then multiplies it by 2.
    with ops.Graph().as_default():
      variable_node = variables.VariableV1(1.0, name="variable_node")
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
    input_meta_graph = checkpoint_meta_graph_file

    freeze_graph.freeze_graph(
        input_graph_path,
        input_saver_def_path,
        input_binary,
        checkpoint_path,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        output_graph_path,
        clear_devices,
        "",
        "",
        input_meta_graph,
        checkpoint_version=saver_write_version)

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

  def _createTFExampleString(self, feature_name, feature_value):
    """Create a serialized tensorflow example."""
    example = example_pb2.Example()
    example.features.feature[feature_name].float_list.value.extend([
        feature_value])
    return example.SerializeToString()

  def _writeDummySavedModel(self, path, feature_name):
    """Writes a classifier with two input features to the given path."""
    with ops.Graph().as_default():
      examples = array_ops.placeholder(dtypes.string, name="input_node")
      feature_configs = {
          feature_name: parsing_ops.FixedLenFeature(shape=[],
                                                    dtype=dtypes.float32),
      }
      features = parsing_ops.parse_example(examples, feature_configs)
      feature = features[feature_name]

      variable_node = variables.VariableV1(1.0, name="variable_node")
      scores = math_ops.multiply(variable_node, feature, name="output_node")
      class_feature = array_ops.fill(array_ops.shape(feature),
                                     "class_%s" % feature_name)
      classes = array_ops.transpose(class_feature)

      with session.Session() as sess:
        sess.run(variables.global_variables_initializer())
        signature = (
            signature_def_utils.classification_signature_def(
                examples=examples,
                classes=classes,
                scores=scores,))
        builder = saved_model_builder.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature,
            },)
        builder.save(as_text=True)

  def testFreezeGraphV1(self):
    self._testFreezeGraph(saver_pb2.SaverDef.V1)

  def testFreezeGraphV2(self):
    self._testFreezeGraph(saver_pb2.SaverDef.V2)

  def testFreezeMetaGraph(self):
    tmp_dir = self.get_temp_dir()
    checkpoint_prefix = os.path.join(tmp_dir, "meta_graph_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    output_graph_filename = os.path.join(tmp_dir, "output_graph.pb")

    with ops.Graph().as_default():
      variable_node = variables.VariableV1(1.0, name="variable_node")
      output_node = math_ops.multiply(variable_node, 2.0, name="output_node")
      sess = session.Session()
      init = variables.global_variables_initializer()
      sess.run(init)
      output = sess.run(output_node)
      self.assertNear(2.0, output, 0.00001)
      saver = saver_lib.Saver()
      checkpoint_path = saver.save(
          sess,
          checkpoint_prefix,
          global_step=0,
          latest_filename=checkpoint_state_name)

    input_saver_def_path = ""
    input_binary = True
    output_node_names = "output_node"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = False
    input_meta_graph = checkpoint_path + ".meta"

    freeze_graph.freeze_graph(
        "", input_saver_def_path, input_binary, checkpoint_path,
        output_node_names, restore_op_name, filename_tensor_name,
        output_graph_filename, clear_devices, "", "", "", input_meta_graph)

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with ops.Graph().as_default():
      output_graph_def = graph_pb2.GraphDef()
      with open(output_graph_filename, "rb") as f:
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

  def testFreezeSavedModel(self):
    tmp_dir = self.get_temp_dir()
    saved_model_dir = os.path.join(tmp_dir, "saved_model_dir")
    feature_name = "feature"
    self._writeDummySavedModel(saved_model_dir, feature_name)
    output_graph_filename = os.path.join(tmp_dir, "output_graph.pb")

    input_saved_model_dir = saved_model_dir
    output_node_names = "output_node"
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = tag_constants.SERVING

    freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                              saved_model_tags)

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with ops.Graph().as_default():
      output_graph_def = graph_pb2.GraphDef()
      with open(output_graph_filename, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = importer.import_graph_def(output_graph_def, name="")

      self.assertEqual(8, len(output_graph_def.node))
      for node in output_graph_def.node:
        self.assertNotEqual("VariableV2", node.op)
        self.assertNotEqual("Variable", node.op)

      feature_value = 2.0
      example = self._createTFExampleString(feature_name, feature_value)
      with session.Session() as sess:
        input_node = sess.graph.get_tensor_by_name("input_node:0")
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = sess.run(output_node, feed_dict={input_node: [example]})
        self.assertNear(feature_value, output, 0.00001)

  def testSinglePartitionedVariable(self):
    """Ensures partitioned variables fail cleanly with freeze graph."""
    checkpoint_prefix = os.path.join(self.get_temp_dir(), "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"

    # Create a graph with partition variables. When weights are partitioned into
    # a single partition, the weights variable is followed by a identity ->
    # identity (an additional identity node).
    partitioner = partitioned_variables.fixed_size_partitioner(1)
    with ops.Graph().as_default():
      with variable_scope.variable_scope("part", partitioner=partitioner):
        batch_size, height, width, depth = 5, 128, 128, 3
        input1 = array_ops.zeros(
            (batch_size, height, width, depth), name="input1")
        input2 = array_ops.zeros(
            (batch_size, height, width, depth), name="input2")

        num_nodes = depth
        filter1 = variable_scope.get_variable("filter", [num_nodes, num_nodes])
        filter2 = array_ops.reshape(filter1, [1, 1, num_nodes, num_nodes])
        conv = nn.conv2d(
            input=input1, filter=filter2, strides=[1, 1, 1, 1], padding="SAME")
        node = math_ops.add(conv, input2, name="test/add")
        node = nn.relu6(node, name="test/relu6")

      # Save graph and checkpoints.
      sess = session.Session()
      sess.run(variables.global_variables_initializer())

      saver = saver_lib.Saver()
      checkpoint_path = saver.save(
          sess,
          checkpoint_prefix,
          global_step=0,
          latest_filename=checkpoint_state_name)
      graph_io.write_graph(sess.graph, self.get_temp_dir(), input_graph_name)

      # Ensure this graph has partition variables.
      self.assertTrue([
          tensor.name.split(":")[0]
          for op in sess.graph.get_operations()
          for tensor in op.values()
          if re.search(r"/part_\d+/", tensor.name)
      ])

    # Test freezing graph doesn't make it crash.
    output_node_names = "save/restore_all"
    output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)

    return_value = freeze_graph.freeze_graph_with_def_protos(
        input_graph_def=sess.graph_def,
        input_saver_def=None,
        input_checkpoint=checkpoint_path,
        output_node_names=output_node_names,
        restore_op_name="save/restore_all",  # default value
        filename_tensor_name="save/Const:0",  # default value
        output_graph=output_graph_path,
        clear_devices=False,
        initializer_nodes="")
    self.assertTrue(return_value, -1)


if __name__ == "__main__":
  test.main()
