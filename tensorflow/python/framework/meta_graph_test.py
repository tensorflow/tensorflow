# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for tensorflow.python.framework.meta_graph.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import random
import shutil
from itertools import permutations

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import queue_runner_impl


# pylint: disable=invalid-name
def _TestDir(test_name):
  test_dir = os.path.join(test.get_temp_dir(), test_name)
  if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
  gfile.MakeDirs(test_dir)
  return test_dir


# pylint: enable=invalid-name


class SimpleMetaGraphTest(test.TestCase):

  def testNoVariables(self):
    test_dir = _TestDir("no_variables")
    filename = os.path.join(test_dir, "metafile")

    input_feed_value = -10  # Arbitrary input value for feed_dict.

    orig_graph = ops.Graph()
    with self.test_session(graph=orig_graph) as sess:
      # Create a minimal graph with zero variables.
      input_tensor = array_ops.placeholder(
          dtypes.float32, shape=[], name="input")
      offset = constant_op.constant(42, dtype=dtypes.float32, name="offset")
      output_tensor = math_ops.add(input_tensor, offset, name="add_offset")

      # Add input and output tensors to graph collections.
      ops.add_to_collection("input_tensor", input_tensor)
      ops.add_to_collection("output_tensor", output_tensor)

      output_value = sess.run(output_tensor, {input_tensor: input_feed_value})
      self.assertEqual(output_value, 32)

      # Generates MetaGraphDef.
      meta_graph_def, var_list = meta_graph.export_scoped_meta_graph(
          filename=filename,
          graph_def=ops.get_default_graph().as_graph_def(add_shapes=True),
          collection_list=["input_tensor", "output_tensor"],
          saver_def=None)
      self.assertTrue(meta_graph_def.HasField("meta_info_def"))
      self.assertNotEqual(meta_graph_def.meta_info_def.tensorflow_version, "")
      self.assertNotEqual(meta_graph_def.meta_info_def.tensorflow_git_version,
                          "")
      self.assertEqual({}, var_list)

    # Create a clean graph and import the MetaGraphDef nodes.
    new_graph = ops.Graph()
    with self.test_session(graph=new_graph) as sess:
      # Import the previously export meta graph.
      meta_graph.import_scoped_meta_graph(filename)

      # Re-exports the current graph state for comparison to the original.
      new_meta_graph_def, _ = meta_graph.export_scoped_meta_graph(filename +
                                                                  "_new")
      self.assertProtoEquals(meta_graph_def, new_meta_graph_def)

      # Ensures that we can still get a reference to our graph collections.
      new_input_tensor = ops.get_collection("input_tensor")[0]
      new_output_tensor = ops.get_collection("output_tensor")[0]
      # Verifies that the new graph computes the same result as the original.
      new_output_value = sess.run(new_output_tensor,
                                  {new_input_tensor: input_feed_value})
      self.assertEqual(new_output_value, output_value)

  def testStrippedOpListNestedFunctions(self):
    with self.test_session():
      # Square two levels deep
      @function.Defun(dtypes.int32)
      def f0(x):
        return math_ops.square(x)

      @function.Defun(dtypes.int32)
      def f1(x):
        return f0(x)

      # At this point we've defined two functions but haven't called them, so
      # there should be no used ops.
      op_list = meta_graph.stripped_op_list_for_graph(ops.get_default_graph()
                                                      .as_graph_def())
      self.assertEqual(len(op_list.op), 0)

      # If we call the function on a constant, there should be two ops
      _ = f1(constant_op.constant(7))
      op_list = meta_graph.stripped_op_list_for_graph(ops.get_default_graph()
                                                      .as_graph_def())
      self.assertEqual(["Const", "Square"], [op.name for op in op_list.op])

  def testStrippedOpListRecursiveFunctions(self):
    # The function module doesn't support recursive functions, so we build a
    # recursive function situation by ourselves: A calls B calls A and Const.
    graph = graph_pb2.GraphDef()
    a = graph.library.function.add()
    b = graph.library.function.add()
    a.signature.name = "A"
    b.signature.name = "B"
    a.node_def.add().op = "B"
    b.node_def.add().op = "Const"
    b.node_def.add().op = "A"

    # Use A in the graph
    graph.node.add().op = "A"

    # The stripped op list should contain just Const.
    op_list = meta_graph.stripped_op_list_for_graph(graph)
    self.assertEqual(["Const"], [op.name for op in op_list.op])


class ScopedMetaGraphTest(test.TestCase):

  def _testScopedExport(self, test_dir, exported_filenames):
    graph = ops.Graph()
    with graph.as_default():
      # Creates an inference graph.
      # Hidden 1
      colocate_constraint = constant_op.constant(1.2, name="constraint")
      images = constant_op.constant(
          1.2, dtypes.float32, shape=[100, 28], name="images")
      with ops.name_scope("hidden1"):
        with graph.colocate_with(colocate_constraint.op):
          weights1 = variables.Variable(
              random_ops.truncated_normal(
                  [28, 128], stddev=1.0 / math.sqrt(float(28))),
              name="weights")
        # The use of control_flow_ops.cond here is purely for adding test
        # coverage the save and restore of control flow context (which doesn't
        # make any sense here from a machine learning perspective).  The typical
        # biases is a simple Variable without the conditions.
        biases1 = variables.Variable(
            control_flow_ops.cond(
                math_ops.less(random.random(), 0.5),
                lambda: array_ops.ones([128]), lambda: array_ops.zeros([128])),
            name="biases")
        hidden1 = nn_ops.relu(math_ops.matmul(images, weights1) + biases1)

      # Hidden 2
      with ops.name_scope("hidden2"):
        weights2 = variables.Variable(
            random_ops.truncated_normal(
                [128, 32], stddev=1.0 / math.sqrt(float(128))),
            name="weights")

        # The use of control_flow_ops.while_loop here is purely for adding test
        # coverage the save and restore of control flow context (which doesn't
        # make any sense here from a machine learning perspective).  The typical
        # biases is a simple Variable without the conditions.
        def loop_cond(it, _):
          return it < 2

        def loop_body(it, biases2):
          biases2 += constant_op.constant(0.1, shape=[32])
          return it + 1, biases2

        _, biases2 = control_flow_ops.while_loop(
            loop_cond,
            loop_body, [
                constant_op.constant(0), variables.Variable(
                    array_ops.zeros([32]), name="biases")
            ])
        hidden2 = nn_ops.relu(math_ops.matmul(hidden1, weights2) + biases2)
      # Linear
      with ops.name_scope("softmax_linear"):
        weights3 = variables.Variable(
            random_ops.truncated_normal(
                [32, 10], stddev=1.0 / math.sqrt(float(32))),
            name="weights")
        biases3 = variables.Variable(array_ops.zeros([10]), name="biases")
        logits = math_ops.matmul(hidden2, weights3) + biases3
        ops.add_to_collection("logits", logits)

      # Exports each sub-graph.
      # Exports the first one with unbound_inputs_col_name set to default.
      orig_meta_graph1, var_list = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filenames[0]),
          graph=ops.get_default_graph(),
          export_scope="hidden1")
      self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
      var_names = [v.name for _, v in var_list.items()]
      self.assertEqual(["hidden1/biases:0", "hidden1/weights:0"],
                       sorted(var_names))

      # Exports the rest with no unbound_inputs_col_name.
      orig_meta_graph2, _ = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filenames[1]),
          graph=ops.get_default_graph(),
          export_scope="hidden2",
          unbound_inputs_col_name=None)
      orig_meta_graph3, _ = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filenames[2]),
          graph=ops.get_default_graph(),
          export_scope="softmax_linear",
          unbound_inputs_col_name=None)

    return [orig_meta_graph1, orig_meta_graph2, orig_meta_graph3]

  def _testScopedImport(self, test_dir, exported_filenames):
    graph = ops.Graph()
    # Create all the missing inputs.
    with graph.as_default():
      new_image = constant_op.constant(
          1.2, dtypes.float32, shape=[100, 28], name="images")

    with self.assertRaisesRegexp(ValueError, "Graph contains unbound inputs"):
      meta_graph.import_scoped_meta_graph(
          os.path.join(test_dir, exported_filenames[0]),
          graph=graph,
          import_scope="new_hidden1")

    with self.assertRaisesRegexp(ValueError, "Graph contains unbound inputs"):
      meta_graph.import_scoped_meta_graph(
          os.path.join(test_dir, exported_filenames[0]),
          graph=graph,
          input_map={"image:0": new_image},
          import_scope="new_hidden1")

    # Verifies we can import the original "hidden1" into "new_hidden1".
    var_list = meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filenames[0]),
        graph=graph,
        input_map={"$unbound_inputs_images": new_image},
        import_scope="new_hidden1")

    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
    new_var_names = [v.name for _, v in var_list.items()]
    self.assertEqual(["new_hidden1/biases:0", "new_hidden1/weights:0"],
                     sorted(new_var_names))

    # Verifies we can import the original "hidden2" into "new_hidden2".
    hidden1 = array_ops.identity(
        graph.as_graph_element("new_hidden1/Relu:0"), name="hidden1/Relu")
    var_list = meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filenames[1]),
        graph=graph,
        input_map={"$unbound_inputs_hidden1/Relu": hidden1},
        import_scope="new_hidden2",
        unbound_inputs_col_name=None)

    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
    new_var_names = [v.name for _, v in var_list.items()]
    self.assertEqual(["new_hidden2/biases:0", "new_hidden2/weights:0"],
                     sorted(new_var_names))

    # Verifies we can import the original "softmax_linear" into
    # "new_softmax_linear".
    hidden2 = array_ops.identity(
        graph.as_graph_element("new_hidden2/Relu:0"), name="hidden2/Relu")
    var_list = meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filenames[2]),
        graph=graph,
        input_map={"$unbound_inputs_hidden2/Relu": hidden2},
        import_scope="new_softmax_linear",
        unbound_inputs_col_name=None)
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
    new_var_names = [v.name for _, v in var_list.items()]
    self.assertEqual(
        ["new_softmax_linear/biases:0", "new_softmax_linear/weights:0"],
        sorted(new_var_names))

    # Exports the scoped meta graphs again.
    new_meta_graph1, var_list = meta_graph.export_scoped_meta_graph(
        graph=graph, export_scope="new_hidden1")
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))

    new_meta_graph2, var_list = meta_graph.export_scoped_meta_graph(
        graph=graph, export_scope="new_hidden2", unbound_inputs_col_name=None)
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))

    new_meta_graph3, var_list = meta_graph.export_scoped_meta_graph(
        graph=graph,
        export_scope="new_softmax_linear",
        unbound_inputs_col_name=None)
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))

    return [new_meta_graph1, new_meta_graph2, new_meta_graph3]

  # Verifies that we can export the subgraph under each layer and import
  # them into new layers in a new graph.
  def testScopedExportAndImport(self):
    test_dir = _TestDir("scoped_export_import")
    filenames = [
        "exported_hidden1.pbtxt", "exported_hidden2.pbtxt",
        "exported_softmax_linear.pbtxt"
    ]
    orig_meta_graphs = self._testScopedExport(test_dir, filenames)
    new_meta_graphs = self._testScopedImport(test_dir, filenames)
    # Delete the unbound_inputs to allow directly calling ProtoEqual.
    del orig_meta_graphs[0].collection_def["unbound_inputs"]
    del new_meta_graphs[0].collection_def["unbound_inputs"]
    for a, b in zip(orig_meta_graphs, new_meta_graphs):
      self.assertProtoEquals(a, b)

  def _testScopedExportWithQueue(self, test_dir, exported_filename):
    graph = ops.Graph()
    with graph.as_default():
      with ops.name_scope("queue1"):
        input_queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
        enqueue = input_queue.enqueue((9876), name="enqueue")
        close = input_queue.close(name="close")
        qr = queue_runner_impl.QueueRunner(input_queue, [enqueue], close)
        queue_runner_impl.add_queue_runner(qr)
        input_queue.dequeue(name="dequeue")

      orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filename),
          graph=ops.get_default_graph(),
          export_scope="queue1")

    return orig_meta_graph

  def _testScopedImportWithQueue(self, test_dir, exported_filename,
                                 new_exported_filename):
    graph = ops.Graph()
    meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filename),
        graph=graph,
        import_scope="new_queue1")
    graph.as_graph_element("new_queue1/dequeue:0")
    graph.as_graph_element("new_queue1/close")
    with graph.as_default():
      new_meta_graph, _ = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, new_exported_filename),
          graph=graph,
          export_scope="new_queue1")

    return new_meta_graph

  # Verifies that we can export the subgraph containing a FIFOQueue under
  # "queue1" and import it into "new_queue1" in a new graph.
  def testScopedWithQueue(self):
    test_dir = _TestDir("scoped_with_queue")
    orig_meta_graph = self._testScopedExportWithQueue(test_dir,
                                                      "exported_queue1.pbtxt")
    new_meta_graph = self._testScopedImportWithQueue(
        test_dir, "exported_queue1.pbtxt", "exported_new_queue1.pbtxt")
    self.assertProtoEquals(orig_meta_graph, new_meta_graph)

  # Verifies that we can export a subgraph in a nested name scope containing a
  # "hidden1/hidden2" and import it into "new_hidden1/new_hidden2" in a new
  # graph.
  def testExportNestedNames(self):
    graph1 = ops.Graph()
    with graph1.as_default():
      with ops.name_scope("hidden1/hidden2/hidden3"):
        images = constant_op.constant(
            1.0, dtypes.float32, shape=[3, 2], name="images")
        weights1 = variables.Variable(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="weights")
        biases1 = variables.Variable([0.1] * 3, name="biases")
        nn_ops.relu(math_ops.matmul(images, weights1) + biases1, name="relu")

    orig_meta_graph, var_list = meta_graph.export_scoped_meta_graph(
        export_scope="hidden1/hidden2", graph=graph1)
    var_names = [v.name for _, v in var_list.items()]
    self.assertEqual(["hidden3/biases:0", "hidden3/weights:0"],
                     sorted(var_list.keys()))
    self.assertEqual([
        "hidden1/hidden2/hidden3/biases:0", "hidden1/hidden2/hidden3/weights:0"
    ], sorted(var_names))
    for node in orig_meta_graph.graph_def.node:
      self.assertTrue(node.name.startswith("hidden3"))

    graph2 = ops.Graph()
    new_var_list = meta_graph.import_scoped_meta_graph(
        orig_meta_graph, import_scope="new_hidden1/new_hidden2", graph=graph2)
    self.assertEqual(["hidden3/biases:0", "hidden3/weights:0"],
                     sorted(new_var_list.keys()))
    new_var_names = [v.name for _, v in new_var_list.items()]
    self.assertEqual([
        "new_hidden1/new_hidden2/hidden3/biases:0",
        "new_hidden1/new_hidden2/hidden3/weights:0"
    ], sorted(new_var_names))

    nodes = [
        "new_hidden1/new_hidden2/hidden3/biases/Assign",
        "new_hidden1/new_hidden2/hidden3/weights/Assign"
    ]
    expected = [
        b"loc:@new_hidden1/new_hidden2/hidden3/biases",
        b"loc:@new_hidden1/new_hidden2/hidden3/weights"
    ]
    for n, e in zip(nodes, expected):
      self.assertEqual([e], graph2.get_operation_by_name(n).get_attr("_class"))

  def testPotentialCycle(self):
    graph1 = ops.Graph()
    with graph1.as_default():
      a = constant_op.constant(1.0, shape=[2, 2])
      b = constant_op.constant(2.0, shape=[2, 2])
      matmul = math_ops.matmul(a, b)
      with ops.name_scope("hidden1"):
        c = nn_ops.relu(matmul)
        d = constant_op.constant(3.0, shape=[2, 2])
        matmul = math_ops.matmul(c, d)

    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
        export_scope="hidden1", graph=graph1)

    graph2 = ops.Graph()
    with graph2.as_default():
      with self.assertRaisesRegexp(ValueError, "Graph contains unbound inputs"):
        meta_graph.import_scoped_meta_graph(
            orig_meta_graph, import_scope="new_hidden1")

      meta_graph.import_scoped_meta_graph(
          orig_meta_graph,
          import_scope="new_hidden1",
          input_map={
              "$unbound_inputs_MatMul": constant_op.constant(
                  4.0, shape=[2, 2])
          })

  def testClearDevices(self):
    graph1 = ops.Graph()
    with graph1.as_default():
      with ops.device("/device:CPU:0"):
        a = variables.Variable(
            constant_op.constant(
                1.0, shape=[2, 2]), name="a")
      with ops.device("/job:ps/replica:0/task:0/gpu:0"):
        b = variables.Variable(
            constant_op.constant(
                2.0, shape=[2, 2]), name="b")
      with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
        math_ops.matmul(a, b, name="matmul")

    self.assertEqual("/device:CPU:0", str(graph1.as_graph_element("a").device))
    self.assertEqual("/job:ps/replica:0/task:0/device:GPU:0",
                     str(graph1.as_graph_element("b").device))
    self.assertEqual("/job:localhost/replica:0/task:0/device:CPU:0",
                     str(graph1.as_graph_element("matmul").device))

    # Verifies that devices are cleared on export.
    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
        graph=graph1, clear_devices=True)

    graph2 = ops.Graph()
    with graph2.as_default():
      meta_graph.import_scoped_meta_graph(orig_meta_graph, clear_devices=False)

    self.assertEqual("", str(graph2.as_graph_element("a").device))
    self.assertEqual("", str(graph2.as_graph_element("b").device))
    self.assertEqual("", str(graph2.as_graph_element("matmul").device))

    # Verifies that devices are cleared on export when passing in graph_def.
    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
        graph_def=graph1.as_graph_def(), clear_devices=True)

    graph2 = ops.Graph()
    with graph2.as_default():
      meta_graph.import_scoped_meta_graph(orig_meta_graph, clear_devices=False)

    self.assertEqual("", str(graph2.as_graph_element("a").device))
    self.assertEqual("", str(graph2.as_graph_element("b").device))
    self.assertEqual("", str(graph2.as_graph_element("matmul").device))

    # Verifies that devices are cleared on import.
    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
        graph=graph1, clear_devices=False)

    graph2 = ops.Graph()
    with graph2.as_default():
      meta_graph.import_scoped_meta_graph(orig_meta_graph, clear_devices=True)

    self.assertEqual("", str(graph2.as_graph_element("a").device))
    self.assertEqual("", str(graph2.as_graph_element("b").device))
    self.assertEqual("", str(graph2.as_graph_element("matmul").device))


class TestGetBackwardTensors(tf.test.TestCase):

  def testGetBackwardOpsChain(self):
    # a -> b -> c
    a = tf.placeholder(tf.float32)
    b = tf.sqrt(a)
    c = tf.square(b)
    for n in range(4):
      for seed_tensors in permutations([a, b, c], n):
        if c in seed_tensors:
          truth = [a.op, b.op, c.op]
        elif b in seed_tensors:
          truth = [a.op, b.op]
        elif a in seed_tensors:
          truth = [a.op]
        else:
          truth = []
        self.assertEqual(meta_graph._get_backward_ops(seed_tensors), truth)

    self.assertEqual(meta_graph._get_backward_ops([c], as_inputs=[b]), [c.op])
    self.assertEqual(
      meta_graph._get_backward_ops([b, c], as_inputs=[b]), [c.op])
    self.assertEqual(
      meta_graph._get_backward_ops([a, c], as_inputs=[b]), [a.op, c.op])


  def testGetBackwardOpsSplit(self):
    # a -> b -> c
    #       \-> d
    a = tf.placeholder(tf.float32)
    b = tf.exp(a)
    c = tf.log(b)
    d = tf.neg(b)
    self.assertEqual(meta_graph._get_backward_ops([d]), [a.op, b.op, d.op])
    self.assertEqual(meta_graph._get_backward_ops([c]), [a.op, b.op, c.op])
    self.assertEqual(
      meta_graph._get_backward_ops([c, d]), [a.op, b.op, c.op, d.op])
    self.assertEqual(meta_graph._get_backward_ops([b, d]), [a.op, b.op, d.op])
    self.assertEqual(meta_graph._get_backward_ops([a, d]), [a.op, b.op, d.op])

    self.assertEqual(
      meta_graph._get_backward_ops([c, d], as_inputs=[b]), [c.op, d.op])
    self.assertEqual(
      meta_graph._get_backward_ops([c], as_inputs=[d]), [a.op, b.op, c.op])


  def testGetBackwardOpsMerge(self):
    # a -> c -> d
    # b ->/
    a = tf.placeholder(tf.float32)
    b = tf.constant(0, dtype=tf.int32)
    c = tf.reduce_sum(a, reduction_indices=b)
    d = tf.stop_gradient(c)
    self.assertEqual(
      meta_graph._get_backward_ops([d]), [a.op, b.op, c.op, d.op])
    self.assertEqual(meta_graph._get_backward_ops([d], as_inputs=[c]), [d.op])
    self.assertEqual(
      meta_graph._get_backward_ops([d], as_inputs=[a]), [b.op, c.op, d.op])


  def testGetBackwardOpsBridge(self):
    # a -> b -> c -> d -> e
    #       \  ---  /
    a = tf.placeholder(tf.int32)
    b = tf.identity(a)
    c = tf.cast(b, tf.float32)
    d = tf.tile(c, b)
    e = tf.tanh(d)
    self.assertEqual(
      meta_graph._get_backward_ops([e]), [a.op, b.op, c.op, d.op, e.op])
    self.assertEqual(meta_graph._get_backward_ops([c]), [a.op, b.op, c.op])
    self.assertEqual(meta_graph._get_backward_ops([e], as_inputs=[c]),
                    [a.op, b.op, d.op, e.op])


  def testGetBackwardOpsControlDeps(self):
    # a -> b - \
    # c -> d - e
    #       \ /
    #        f
    a = tf.placeholder(tf.float32, name='a')
    b = tf.identity(a, name='b')
    c = tf.placeholder(tf.float32, name='c')
    d = tf.identity(c, name='d')
    with tf.control_dependencies([b, d]):
      e = tf.placeholder(tf.float32, name='e')
    with tf.control_dependencies([e, d]):
      f = tf.placeholder(tf.float32, name='f')
    self.assertEqual(meta_graph._get_backward_ops([f]),
                    [a.op, b.op, c.op, d.op, e.op, f.op])
    self.assertEqual(meta_graph._get_backward_ops([d, f]),
                    [c.op, d.op, a.op, b.op, e.op, f.op])

    self.assertEqual(meta_graph._get_backward_ops([f], as_inputs=[b]),
                    [a.op, b.op, c.op, d.op, e.op, f.op])
    self.assertEqual(meta_graph._get_backward_ops([f], as_inputs=[b, c]),
                    [a.op, b.op, d.op, e.op, f.op])
    self.assertEqual(meta_graph._get_backward_ops([f], as_inputs=[d, e]),
                    [a.op, b.op, c.op, d.op, e.op, f.op])
    self.assertEqual(meta_graph._get_backward_ops([d, f], as_inputs=[b]),
                    [c.op, d.op, a.op, b.op, e.op, f.op])


class TestClone(tf.test.TestCase):

  def testCloneChain(self):
    # a -> b -> c
    g = tf.Graph()
    with g.as_default():
      a = tf.constant(1., name="a")
      b = tf.sqrt(a, name="b")
      c = tf.square(b, name="c")

      a_new = tf.constant(4., name="a_new")
      b_new = tf.constant(2., name="b_new")

      # case 1
      c_out = meta_graph.clone(c, "copy1", replace={b: b_new})
      with tf.Session() as sess:
        self.assertNear(sess.run(c_out), 4., 1e-6)

      # case 2
      copies = meta_graph.clone([b, c], "copy2", replace={b: b_new})
      with tf.Session() as sess:
        b_out_, c_out_ = sess.run(copies)
      self.assertNear(b_out_, 2., 1e-6)
      self.assertNear(c_out_, 4., 1e-6)

      # case 3
      copies = meta_graph.clone([a, c], "copy3", replace={b: b_new})
      with tf.Session() as sess:
        a_out_, c_out_ = sess.run(copies)
      self.assertNear(a_out_, 1., 1e-6)
      self.assertNear(c_out_, 4., 1e-6)

      # case 4
      copies = meta_graph.clone([a, b, c], "copy4", replace={a: a_new})
      with tf.Session() as sess:
        a_out_, b_out_, c_out_ = sess.run(copies)
      self.assertNear(a_out_, 4., 1e-6)
      self.assertNear(b_out_, 2., 1e-6)
      self.assertNear(c_out_, 4., 1e-6)

  def testCloneSplit(self):
    # a -> b -> c
    #       \-> d
    g = tf.Graph()
    with g.as_default():
      a = tf.constant(1., name="a")
      b = tf.exp(a, name="b")
      c = tf.log(b, name="c")
      d = tf.neg(b, name="d")

      b_new = tf.constant(math.e ** 2, name="b_new")
      d_new = tf.constant(-math.e ** 2, name="d_new")

      # case 1
      d_out = meta_graph.clone(d, "copy1")
      self.assertEqual(d_out.name, "copy1/d:0")
      self.assertEqual(d_out.op.inputs[:], [b])

      # case 2
      copies = meta_graph.clone([c, d], "copy2")
      self.assertEqual(copies[0].op.inputs[:], [b])
      self.assertEqual(copies[1].op.inputs[:], [b])

      # case 3
      copies = meta_graph.clone([c, d], "copy3", replace={b: b_new})
      with tf.Session() as sess:
        c_out_, d_out_ = sess.run(copies)
      self.assertNear(c_out_, 2., 1e-6)
      self.assertNear(d_out_, -math.e ** 2, 1e-6)

      # case 4
      c_out = meta_graph.clone(c, "copy4", replace={d: d_new})
      self.assertEqual(c_out.op.inputs[:], [b])
      with tf.Session() as sess:
        self.assertNear(sess.run(c_out), 1., 1e-6)

  def testCloneMerge(self):
    # a -> c -> d
    # b ->/
    g = tf.Graph()
    with g.as_default():
      a = tf.constant(4., name='a')
      b = tf.constant(0., name='b')
      c = tf.add(a, b, name='c')
      d = tf.stop_gradient(c, name='d')

      a_new = tf.constant(10., name='a_new')
      b_new = tf.constant(1., name='b_new')
      c_new = tf.constant(-1., name='c_new')

      # case 1
      copies = meta_graph.clone([a, b, c, d], "copy1", replace={a: a_new})
      with tf.Session() as sess:
        a_out_, b_out_, c_out_, d_out_ = sess.run(copies)
      self.assertNear(a_out_, 10., 1e-6)
      self.assertNear(b_out_, 0., 1e-6)
      self.assertNear(c_out_, 10., 1e-6)
      self.assertNear(d_out_, 10., 1e-6)

      # case 2
      copies = meta_graph.clone([a, b, c, d], "copy2", replace={b: b_new})
      with tf.Session() as sess:
        a_out_, b_out_, c_out_, d_out_ = sess.run(copies)
      self.assertNear(a_out_, 4., 1e-6)
      self.assertNear(b_out_, 1., 1e-6)
      self.assertNear(c_out_, 5., 1e-6)
      self.assertNear(d_out_, 5., 1e-6)

      # case 3
      copies = meta_graph.clone([a, b, c, d], "copy3", replace={c: c_new})
      assert copies[0].name == "copy3/a:0"
      assert copies[1].name == "copy3/b:0"
      with tf.Session() as sess:
        a_out_, b_out_, c_out_, d_out_ = sess.run(copies)
      self.assertNear(a_out_, 4., 1e-6)
      self.assertNear(b_out_, 0., 1e-6)
      self.assertNear(c_out_, -1., 1e-6)
      self.assertNear(d_out_, -1., 1e-6)

  def testCloneBridge(self):
    # a -> b -> c -> d -> e
    #       \  ---  /
    g = tf.Graph()
    with g.as_default():
      a = tf.constant([2], dtype=tf.int32, name='a')
      b = tf.identity(a, name='b')
      c = tf.neg(b, name='c')
      d = tf.tile(c, b, name='d')
      e = tf.square(d, name='e')

      a_new = tf.constant([3], dtype=tf.int32, name='a_new')
      b_new = tf.constant([4], dtype=tf.int32, name='b_new')
      c_new = tf.constant([5], dtype=tf.int32, name='c_new')
      d_new = tf.constant([5, 5, 5], name='d_new')

      # case 1
      copies = meta_graph.clone([d, e], "copy1", replace={a: a_new, c: c_new})
      with tf.Session() as sess:
        d_out_, e_out_ = sess.run(copies)
      self.assertAllClose(d_out_, np.array([5, 5, 5]))
      self.assertAllClose(e_out_, np.array([25, 25, 25]))

      # case 2
      copies = meta_graph.clone([c, e], "copy2",
                                replace={a: a_new, b: b_new, d: d_new})
      with tf.Session() as sess:
        c_out_, e_out_ = sess.run(copies)
      self.assertAllClose(c_out_, [-4])
      self.assertAllClose(e_out_, np.array([25, 25, 25]))

  def testCloneOneToManyOp(self):
    # tf.unpack
    # a -.---- a0
    #     \ -- a1
    #      \ - a2 -> c
    # b ----------- /
    g = tf.Graph()
    with g.as_default():
      a = tf.zeros([3, 2, 1, 4], name="a")
      a0, a1, a2 = tf.unpack(a, axis=0)
      b = tf.ones([2, 4, 1], name="b")
      c = tf.batch_matmul(a2, b, name="c")

      a1_new = tf.ones([2, 1, 4], name="a1_new")
      a_new = tf.ones([3, 2, 1, 4], name="a_new")
      a2_new = tf.ones([2, 1, 4], name="a2_new") * 2

      # case 1
      copies = meta_graph.clone([a2, c], "copy1", replace={a1: a1_new})
      a2_out, c_out = copies
      self.assertEqual(a2_out.name, "copy1/unpack:2")
      self.assertEqual(c_out.name, "copy1/c:0")

      # case 2
      copies = meta_graph.clone([a0, a2, c], "copy2",
                                replace={a: a_new, a2: a2_new})
      with tf.Session() as sess:
        a0_out_, a2_out_, c_out_ = sess.run(copies)
      self.assertAllClose(a0_out_, np.ones([2, 1, 4]))
      self.assertAllClose(a2_out_, np.ones([2, 1, 4]) * 2)
      self.assertAllClose(c_out_, np.array([[[8]], [[8]]]))

  def testClonePlaceholderFeed(self):
    # a -> c -> c0
    # b - /    /
    #  \ ---- /
    g = tf.Graph()
    with g.as_default():
      a = tf.placeholder(tf.float32, name='a')
      b = tf.placeholder(tf.int32, name='b')
      c = tf.expand_dims(a, b, name='c')
      c0 = tf.split(b, 1, c)[0]

      b_new = tf.placeholder(tf.int32, name='b_new')
      c0_out = meta_graph.clone(c0, "copy", replace={b: b_new})
      with tf.Session() as sess:
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                     "You must feed a value for placeholder"):
          sess.run(c0_out, feed_dict={a: np.ones([2, 3]), b: 0})
        c0_out_ = sess.run(c0_out, feed_dict={a: np.ones([2, 3]),
                                              b_new: 0})
      self.assertAllClose(c0_out_, np.ones([1, 2, 3]))

  def testCloneControlDeps(self):
    # a -> b ---> e -----
    # c -> d --- /       \
    #       \ ----------- f
    g = tf.Graph()
    with g.as_default():
      a = tf.placeholder(tf.float32, name='a')
      b = tf.identity(a, name='b')
      c = tf.placeholder(tf.float32, name='c')
      d = tf.identity(c, name='d')
      with tf.control_dependencies([b, d]):
        e = tf.identity(a, name='e')
      with tf.control_dependencies([e, d]):
        f = tf.identity(a, name='f')

      d_new = tf.add(1., tf.ones([]), name='d_new')
      e_new = tf.add(1., tf.constant([2., 2.]), name='e_new')
      f_out = meta_graph.clone(f, "copy1", replace={d: d_new, e: e_new})
      self.assertEqual(f_out.op.name, "copy1/f")
      self.assertEqual(f_out.op.control_inputs[0].name, "e")
      self.assertEqual(f_out.op.control_inputs[1].name, "d")

  def testCloneAssertEqual(self):
    g = tf.Graph()
    with g.as_default():
      a = tf.placeholder(tf.float32, shape=(), name='a')
      b = tf.identity(a, name='b')
      c = tf.identity(a, name='c')
      _assert_equal = tf.assert_equal(b, c)
      with tf.control_dependencies([_assert_equal]):
        d = tf.add(b, c, name='d')

      a_new = tf.constant(1, dtype=tf.float32, name='a_new')
      d_out = meta_graph.clone(d, "copy1", replace={a: a_new})
      with tf.Session() as sess:
        self.assertNear(sess.run(d_out), 2., 1e-6)

  def testCloneVariable(self):
    # w -> y
    # x - /
    g = tf.Graph()
    with g.as_default():
      with tf.variable_scope("weights"):
        w = tf.get_variable("w", shape=[4, 5],
                            initializer=tf.random_normal_initializer())
      x = tf.ones([5, 2], name="x")
      y = tf.matmul(w, x, name="y")

      x_new = tf.zeros([5, 2], name="x_new")
      with tf.variable_scope("weights_new"):
        w_new = tf.get_variable("w_new", shape=[4, 5],
                                initializer=tf.random_normal_initializer())

      # case 1
      y_out = meta_graph.clone(y, "copy1", replace={x: x_new})
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_out_ = sess.run(y_out)
        self.assertAllClose(y_out_, np.zeros((4, 2)))

      # case 2
      with self.assertRaisesRegexp(TypeError, "consist of Tensor pairs"):
        meta_graph.clone(y, "copy2", replace={w: w_new})

  def testCloneFullyConnected(self):
    g = tf.Graph()
    with g.as_default():
      x = tf.ones([3, 4], name='x')
      y = layers.fully_connected(x, 10)

      x_new = tf.zeros([3, 4], name='x_new')
      y_out = meta_graph.clone(y, "fully_connected/copy1",
                               from_scope="fully_connected",
                               replace={x: x_new})
      self.assertEqual(y_out.op.inputs[0].op.inputs[1].name,
                       "fully_connected/biases/read:0")
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_out_ = sess.run(y_out)
      self.assertAllClose(y_out_, np.zeros((3, 10)))

  def testCloneConvolution(self):
    g = tf.Graph()
    with g.as_default():
      x = tf.ones([2, 5, 5, 3], name='x')
      y = layers.conv2d(x, 2, [3, 3])

      x_new = tf.zeros([2, 5, 5, 3], name='x_new')
      y_out = meta_graph.clone(y, "Conv/copy1", from_scope="Conv",
                               replace={x: x_new})
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_out_ = sess.run(y_out)
      self.assertEqual(y_out.op.inputs[0].op.inputs[1].name,
                       "Conv/biases/read:0")
      self.assertAllClose(y_out_, np.zeros((2, 5, 5, 2)))

  def testCloneBatchNorm(self):
    g = tf.Graph()
    with g.as_default():
      np.random.seed(1234)
      x_value = np.random.random([2, 5, 5, 3])
      w_value = np.random.random([3, 3, 3, 2])
      is_training_t = tf.placeholder(tf.bool, name='is_training_t')
      x_t = tf.constant(x_value, dtype=tf.float32, name='x_t')
      y_t = layers.conv2d(x_t, 2, [3, 3], normalizer_fn=layers.batch_norm,
                          normalizer_params={'is_training': is_training_t,
                                             'updates_collections': None},
                          weights_initializer=tf.constant_initializer(w_value))
      optimizer_t = tf.train.AdamOptimizer()
      optimize_t = optimizer_t.minimize(tf.reduce_sum(y_t))
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_test_1 = sess.run(y_t, feed_dict={is_training_t: False})
        sess.run(optimize_t, feed_dict={is_training_t: True})
        y_test_2 = sess.run(y_t, feed_dict={is_training_t: False})

      is_training = tf.placeholder(tf.bool, name='is_training')
      x = tf.constant(np.zeros([2, 5, 5, 3]), dtype=tf.float32, name='x')
      y = layers.conv2d(x, 2, [3, 3], normalizer_fn=layers.batch_norm,
                        normalizer_params={'is_training': is_training,
                                           'updates_collections': None},
                        weights_initializer=tf.constant_initializer(
                          w_value))
      x_new = tf.constant(x_value, dtype=tf.float32, name='x')
      y_out = meta_graph.clone(y, "copy", replace={x: x_new})
      optimizer = tf.train.AdamOptimizer()
      optimize = optimizer.minimize(tf.reduce_sum(y_out))
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_out_1 = sess.run(y_out, feed_dict={is_training: False})
        y_out_2 = sess.run(y_out, feed_dict={is_training: False})
        sess.run(optimize, feed_dict={is_training: True})
        y_out_3 = sess.run(y_out, feed_dict={is_training: False})
      self.assertAllClose(y_out_1, y_out_2)
      self.assertTrue(np.abs(y_out_1 - y_out_3).max() > 1e-6)
      self.assertAllClose(y_test_1, y_out_1)
      self.assertAllClose(y_test_2, y_out_3)


if __name__ == "__main__":
  test.main()
