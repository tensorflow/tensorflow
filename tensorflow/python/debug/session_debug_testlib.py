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
# ==============================================================================
"""Tests for debugger functionalities in tf.Session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil
import tempfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug import debug_data
from tensorflow.python.debug import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class SessionDebugTestBase(test_util.TensorFlowTestCase):
  """Base class for unit tests of tfdbg running with tf.Session."""

  @classmethod
  def setUpClass(cls):
    if test.is_gpu_available():
      cls._expected_partition_graph_count = 2
      cls._expected_num_devices = 2
      cls._main_device = "/job:localhost/replica:0/task:0/gpu:0"
    else:
      cls._expected_partition_graph_count = 1
      cls._expected_num_devices = 1
      cls._main_device = "/job:localhost/replica:0/task:0/cpu:0"

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    self._dump_root = tempfile.mkdtemp()

  def tearDown(self):
    ops.reset_default_graph()

    # Tear down temporary dump directory.
    if os.path.isdir(self._dump_root):
      shutil.rmtree(self._dump_root)

  def _debug_urls(self, run_number=None):
    raise NotImplementedError(
        "_debug_urls() method is not implemented in the base test class.")

  def _debug_dump_dir(self, run_number=None):
    raise NotImplementedError(
        "_debug_dump_dir() method is not implemented in the base test class.")

  def _generate_dump_from_simple_addition_graph(self):
    with session.Session() as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      v_init_val = np.array([[2.0], [-1.0]])

      # Use node names with overlapping namespace (i.e., parent directory) to
      # test concurrent, non-racing directory creation.
      u_name = "u"
      v_name = "v"
      w_name = "w"

      u_init = constant_op.constant(u_init_val, shape=[2, 2])
      u = variables.Variable(u_init, name=u_name)
      v_init = constant_op.constant(v_init_val, shape=[2, 1])
      v = variables.Variable(v_init, name=v_name)

      w = math_ops.matmul(u, v, name=w_name)

      u.initializer.run()
      v.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = "file://%s" % self._dump_root

      # Add debug tensor watch for u.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % u_name, 0, debug_urls=debug_urls)
      # Add debug tensor watch for v.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % v_name, 0, debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()

      # Invoke Session.run().
      sess.run(w, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

    simple_add_results = collections.namedtuple("SimpleAddResults", [
        "u_init_val", "v_init_val", "u", "v", "w", "u_name", "v_name", "w_name",
        "dump"
    ])
    return simple_add_results(u_init_val, v_init_val, u, v, w, u_name, v_name,
                              w_name, dump)

  def testConcurrentDumpingToPathsWithOverlappingParentDirsWorks(self):
    results = self._generate_dump_from_simple_addition_graph()
    self.assertTrue(results.dump.loaded_partition_graphs())

    # Verify the dumped tensor values for u and v.
    self.assertEqual(2, results.dump.size)

    self.assertAllClose([results.u_init_val],
                        results.dump.get_tensors("%s/read" % results.u_name, 0,
                                                 "DebugIdentity"))
    self.assertAllClose([results.v_init_val],
                        results.dump.get_tensors("%s/read" % results.v_name, 0,
                                                 "DebugIdentity"))

    self.assertGreaterEqual(
        results.dump.get_rel_timestamps("%s/read" % results.u_name, 0,
                                        "DebugIdentity")[0], 0)
    self.assertGreaterEqual(
        results.dump.get_rel_timestamps("%s/read" % results.v_name, 0,
                                        "DebugIdentity")[0], 0)

    self.assertGreater(
        results.dump.get_dump_sizes_bytes("%s/read" % results.u_name, 0,
                                          "DebugIdentity")[0], 0)
    self.assertGreater(
        results.dump.get_dump_sizes_bytes("%s/read" % results.v_name, 0,
                                          "DebugIdentity")[0], 0)

  def testGetOpTypeWorks(self):
    results = self._generate_dump_from_simple_addition_graph()

    self.assertEqual(results.u.op.type,
                     results.dump.node_op_type(results.u_name))
    self.assertIn(results.v.op.type, results.dump.node_op_type(results.v_name))
    self.assertIn(results.w.op.type, results.dump.node_op_type(results.w_name))

    with self.assertRaisesRegexp(
        ValueError, "Node 'foo_bar' does not exist in partition graphs."):
      results.dump.node_op_type("foo_bar")

  def testDumpStringTensorsWorks(self):
    with session.Session() as sess:
      str1_init_val = np.array(b"abc")
      str2_init_val = np.array(b"def")

      str1_init = constant_op.constant(str1_init_val)
      str2_init = constant_op.constant(str2_init_val)

      str1_name = "str1"
      str2_name = "str2"
      str1 = variables.Variable(str1_init, name=str1_name)
      str2 = variables.Variable(str2_init, name=str2_name)
      # Concatenate str1 and str2
      str_concat = math_ops.add(str1, str2, name="str_concat")

      str1.initializer.run()
      str2.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = self._debug_urls()

      # Add debug tensor watch for u.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % str1_name, 0, debug_urls=debug_urls)
      # Add debug tensor watch for v.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % str2_name, 0, debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()
      sess.run(str_concat, options=run_options, run_metadata=run_metadata)

      # String ops are located on CPU.
      self.assertEqual(1, len(run_metadata.partition_graphs))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      self.assertIn(str1_name, dump.nodes())
      self.assertIn(str2_name, dump.nodes())

      self.assertEqual(2, dump.size)

      self.assertEqual([str1_init_val],
                       dump.get_tensors("%s/read" % str1_name, 0,
                                        "DebugIdentity"))
      self.assertEqual([str2_init_val],
                       dump.get_tensors("%s/read" % str2_name, 0,
                                        "DebugIdentity"))

      self.assertGreaterEqual(
          dump.get_rel_timestamps("%s/read" % str1_name, 0, "DebugIdentity")[0],
          0)
      self.assertGreaterEqual(
          dump.get_rel_timestamps("%s/read" % str2_name, 0, "DebugIdentity")[0],
          0)

      self.assertGreater(
          dump.get_dump_sizes_bytes("%s/read" % str1_name, 0,
                                    "DebugIdentity")[0], 0)
      self.assertGreater(
          dump.get_dump_sizes_bytes("%s/read" % str2_name, 0,
                                    "DebugIdentity")[0], 0)

  def testDumpUninitializedVariable(self):
    op_namespace = "testDumpUninitializedVariable"
    with session.Session() as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      s_init_val = b"str1"

      u_name = "%s/u" % op_namespace
      s_name = "%s/s" % op_namespace

      u_init = constant_op.constant(u_init_val, shape=[2, 2])
      u = variables.Variable(u_init, name=u_name)
      s_init = constant_op.constant(s_init_val)
      s = variables.Variable(s_init, name=s_name)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = self._debug_urls()

      # Add debug tensor watch for u.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s" % u_name, 0, debug_urls=debug_urls)
      debug_utils.add_debug_tensor_watch(
          run_options, "%s" % s_name, 0, debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()

      # Initialize u and s.
      sess.run(variables.global_variables_initializer(),
               options=run_options,
               run_metadata=run_metadata)

      # Verify the dump file for the uninitialized value of u.
      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      self.assertEqual(2, dump.size)
      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      # Verify that the variable is properly initialized by the run() call.
      u_vals = dump.get_tensors(u_name, 0, "DebugIdentity")
      s_vals = dump.get_tensors(s_name, 0, "DebugIdentity")
      self.assertEqual(1, len(u_vals))
      self.assertIsNone(u_vals[0])
      self.assertEqual(1, len(s_vals))
      self.assertIsNone(s_vals[0])

      # Call run() again, to check that u is initialized properly.
      self.assertAllClose(u_init_val, sess.run(u))
      self.assertEqual(s_init_val, sess.run(s))

  def testDebugWhileLoopGeneratesMultipleDumps(self):
    with session.Session() as sess:
      num_iter = 10

      # "u" is the Variable being updated in the loop.
      u_name = "testDumpToFileWhileLoop/u"
      u_namespace = u_name.split("/")[0]

      u_init_val = np.array(11.0)
      u_init = constant_op.constant(u_init_val)
      u = variables.Variable(u_init, name=u_name)

      # "v" is the increment.
      v_name = "testDumpToFileWhileLoop/v"
      v_namespace = v_name.split("/")[0]

      v_init_val = np.array(2.0)
      v_init = constant_op.constant(v_init_val)
      v = variables.Variable(v_init, name=v_name)

      u.initializer.run()
      v.initializer.run()

      i = constant_op.constant(0, name="testDumpToFileWhileLoop/i")

      def cond(i):
        return math_ops.less(i, num_iter)

      def body(i):
        new_u = state_ops.assign_add(u, v)
        new_i = math_ops.add(i, 1)
        op = control_flow_ops.group(new_u)
        new_i = control_flow_ops.with_dependencies([op], new_i)
        return [new_i]

      loop = control_flow_ops.while_loop(cond, body, [i], parallel_iterations=1)

      # Create RunOptions for debug-watching tensors
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_urls = self._debug_urls()

      # Add debug tensor watch for u.
      debug_utils.add_debug_tensor_watch(
          run_options, u_name, 0, debug_urls=debug_urls)
      # Add debug tensor watch for v.
      debug_utils.add_debug_tensor_watch(
          run_options, "%s/read" % v_name, 0, debug_urls=debug_urls)
      # Add debug tensor watch for while/Identity.
      debug_utils.add_debug_tensor_watch(
          run_options, "while/Identity", 0, debug_urls=debug_urls)
      # Add debug tensor watch for while/Add/y.
      debug_utils.add_debug_tensor_watch(
          run_options, "while/Add/y", 0, debug_urls=debug_urls)

      run_metadata = config_pb2.RunMetadata()
      r = sess.run(loop, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      self.assertEqual(num_iter, r)

      u_val_final = sess.run(u)
      self.assertAllClose(u_init_val + num_iter * v_init_val, u_val_final)

      # Verify dump files
      self.assertTrue(os.path.isdir(self._dump_root))

      self.assertTrue(os.path.isdir(os.path.join(self._dump_root, u_namespace)))
      self.assertTrue(
          os.path.isdir(os.path.join(self._dump_root, v_namespace, "v")))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Expected dumped tensors: u, v/read, 10 iterations of while/Identity,
      # and 10 iterations of while/Add/y.
      self.assertEqual(1 + 1 + num_iter + num_iter, dump.size)

      # Verify tensor values.
      self.assertAllClose([u_init_val],
                          dump.get_tensors(u_name, 0, "DebugIdentity"))
      self.assertAllClose([v_init_val],
                          dump.get_tensors("%s/read" % v_name, 0,
                                           "DebugIdentity"))

      while_id_tensors = dump.get_tensors("while/Identity", 0, "DebugIdentity")
      self.assertEqual(10, len(while_id_tensors))
      for k in xrange(len(while_id_tensors)):
        self.assertAllClose(np.array(k), while_id_tensors[k])

      # Verify ascending timestamps from the while loops.
      while_id_rel_timestamps = dump.get_rel_timestamps("while/Identity", 0,
                                                        "DebugIdentity")
      while_id_dump_sizes_bytes = dump.get_dump_sizes_bytes("while/Identity", 0,
                                                            "DebugIdentity")
      self.assertEqual(10, len(while_id_rel_timestamps))
      prev_rel_time = 0
      prev_dump_size_bytes = while_id_dump_sizes_bytes[0]
      for rel_time, dump_size_bytes in zip(while_id_rel_timestamps,
                                           while_id_dump_sizes_bytes):
        self.assertGreaterEqual(rel_time, prev_rel_time)
        self.assertEqual(dump_size_bytes, prev_dump_size_bytes)
        prev_rel_time = rel_time
        prev_dump_size_bytes = dump_size_bytes

      # Test querying debug watch keys from node name.
      watch_keys = dump.debug_watch_keys("while/Identity")
      self.assertEqual(["while/Identity:0:DebugIdentity"], watch_keys)

      # Test querying debug datum instances from debug watch key.
      self.assertEqual(10, len(dump.watch_key_to_data(watch_keys[0])))
      self.assertEqual([], dump.watch_key_to_data("foo"))

  def testFindNodesWithBadTensorValues(self):
    with session.Session() as sess:
      u_name = "testFindNodesWithBadTensorValues/u"
      v_name = "testFindNodesWithBadTensorValues/v"
      w_name = "testFindNodesWithBadTensorValues/w"
      x_name = "testFindNodesWithBadTensorValues/x"
      y_name = "testFindNodesWithBadTensorValues/y"
      z_name = "testFindNodesWithBadTensorValues/z"

      u_init = constant_op.constant([2.0, 4.0])
      u = variables.Variable(u_init, name=u_name)
      v_init = constant_op.constant([2.0, 1.0])
      v = variables.Variable(v_init, name=v_name)

      # Expected output: [0.0, 3.0]
      w = math_ops.subtract(u, v, name=w_name)

      # Expected output: [inf, 1.3333]
      x = math_ops.div(u, w, name=x_name)

      # Expected output: [nan, 4.0]
      y = math_ops.multiply(w, x, name=y_name)

      z = math_ops.multiply(y, y, name=z_name)

      u.initializer.run()
      v.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      sess.run(z, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      def has_bad_value(_, tensor):
        return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))

      # Find all "offending tensors".
      bad_data = dump.find(has_bad_value)

      # Verify that the nodes with bad values are caught through running find
      # on the debug dump.
      self.assertEqual(3, len(bad_data))
      self.assertEqual(x_name, bad_data[0].node_name)
      self.assertEqual(y_name, bad_data[1].node_name)
      self.assertEqual(z_name, bad_data[2].node_name)

      # Test first_n kwarg of find(): Find the first offending tensor.
      first_bad_datum = dump.find(has_bad_value, first_n=1)

      self.assertEqual(1, len(first_bad_datum))
      self.assertEqual(x_name, first_bad_datum[0].node_name)

  def _session_run_for_graph_structure_lookup(self):
    with session.Session() as sess:
      u_name = "testDumpGraphStructureLookup/u"
      v_name = "testDumpGraphStructureLookup/v"
      w_name = "testDumpGraphStructureLookup/w"

      u_init = constant_op.constant([2.0, 4.0])
      u = variables.Variable(u_init, name=u_name)
      v = math_ops.add(u, u, name=v_name)
      w = math_ops.add(v, v, name=w_name)

      u.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      sess.run(w, options=run_options, run_metadata=run_metadata)

    self.assertEqual(self._expected_partition_graph_count,
                     len(run_metadata.partition_graphs))

    dump = debug_data.DebugDumpDir(
        self._dump_root, partition_graphs=run_metadata.partition_graphs)

    return u_name, v_name, w_name, dump

  def testGraphStructureLookupGivesDevicesAndNodesInfo(self):
    u_name, _, _, dump = self._session_run_for_graph_structure_lookup()

    # Test num_devices().
    self.assertEqual(self._expected_num_devices, len(dump.devices()))

    # Test node_device().
    self.assertEqual(self._main_device, dump.node_device(u_name))

    with self.assertRaisesRegexp(ValueError,
                                 "does not exist in partition graphs"):
      dump.node_device(u_name + "foo")

    # Test node_exists().
    self.assertTrue(dump.node_exists(u_name))
    self.assertTrue(dump.node_exists(u_name + "/read"))
    self.assertFalse(dump.node_exists(u_name + "/read" + "/foo"))

  def testGraphStructureLookupGivesNodesAndAttributes(self):
    u_name, _, _, dump = self._session_run_for_graph_structure_lookup()

    u_read_name = u_name + "/read"

    # Test node name list lookup of the DebugDumpDir object.
    node_names = dump.nodes()
    self.assertTrue(u_name in node_names)
    self.assertTrue(u_read_name in node_names)

    # Test querying node attributes.
    u_attr = dump.node_attributes(u_name)
    self.assertEqual(dtypes.float32, u_attr["dtype"].type)
    self.assertEqual(1, len(u_attr["shape"].shape.dim))
    self.assertEqual(2, u_attr["shape"].shape.dim[0].size)

    with self.assertRaisesRegexp(ValueError, "No node named \"foo\" exists"):
      dump.node_attributes("foo")

  def testGraphStructureLookupGivesDebugWatchKeys(self):
    u_name, v_name, w_name, dump = (
        self._session_run_for_graph_structure_lookup())

    # Test querying the debug watch keys with node names.
    self.assertEqual(["%s:0:DebugIdentity" % u_name],
                     dump.debug_watch_keys(u_name))
    self.assertEqual(["%s:0:DebugIdentity" % v_name],
                     dump.debug_watch_keys(v_name))
    self.assertEqual(["%s:0:DebugIdentity" % w_name],
                     dump.debug_watch_keys(w_name))
    self.assertEqual([], dump.debug_watch_keys("foo"))

    # Test querying debug datum instances from debug watch.
    u_data = dump.watch_key_to_data(dump.debug_watch_keys(u_name)[0])
    self.assertEqual(1, len(u_data))
    self.assertEqual(u_name, u_data[0].node_name)
    self.assertEqual(0, u_data[0].output_slot)
    self.assertEqual("DebugIdentity", u_data[0].debug_op)
    self.assertGreaterEqual(u_data[0].timestamp, 0)

    self.assertEqual([], dump.watch_key_to_data("foo"))

  def testGraphStructureLookupGivesNodeInputsAndRecipients(self):
    u_name, v_name, w_name, dump = (
        self._session_run_for_graph_structure_lookup())

    u_read_name = u_name + "/read"

    # Test the inputs lookup of the DebugDumpDir object.
    self.assertEqual([], dump.node_inputs(u_name))
    self.assertEqual([u_name], dump.node_inputs(u_read_name))
    self.assertEqual([u_read_name] * 2, dump.node_inputs(v_name))
    self.assertEqual([v_name] * 2, dump.node_inputs(w_name))

    self.assertEqual([], dump.node_inputs(u_name, is_control=True))
    self.assertEqual([], dump.node_inputs(u_read_name, is_control=True))
    self.assertEqual([], dump.node_inputs(v_name, is_control=True))
    self.assertEqual([], dump.node_inputs(w_name, is_control=True))

    # Test the outputs recipient lookup of the DebugDumpDir object.
    self.assertTrue(u_read_name in dump.node_recipients(u_name))
    self.assertEqual(2, dump.node_recipients(u_read_name).count(v_name))
    self.assertEqual(2, dump.node_recipients(v_name).count(w_name))

    self.assertEqual([], dump.node_recipients(u_name, is_control=True))
    self.assertEqual([], dump.node_recipients(u_read_name, is_control=True))
    self.assertEqual([], dump.node_recipients(v_name, is_control=True))
    self.assertEqual([], dump.node_recipients(w_name, is_control=True))

    # Test errors raised on invalid node names.
    with self.assertRaisesRegexp(ValueError,
                                 "does not exist in partition graphs"):
      dump.node_inputs(u_name + "foo")

    with self.assertRaisesRegexp(ValueError,
                                 "does not exist in partition graphs"):
      dump.node_recipients(u_name + "foo")

    # Test transitive_inputs().
    self.assertEqual([], dump.transitive_inputs(u_name))
    self.assertEqual([u_name], dump.transitive_inputs(u_read_name))
    self.assertEqual(
        set([u_name, u_read_name]), set(dump.transitive_inputs(v_name)))
    self.assertEqual(
        set([u_name, u_read_name, v_name]), set(dump.transitive_inputs(w_name)))

    with self.assertRaisesRegexp(ValueError,
                                 "does not exist in partition graphs"):
      dump.transitive_inputs(u_name + "foo")

  def testGraphStructureLookupWithoutPartitionGraphsDoesNotErrorOut(self):
    _, _, _, dump = self._session_run_for_graph_structure_lookup()

    # Now load the dump again, without the partition graphs, so we can check
    # errors are not raised because the partition graphs are loaded from the
    # dump directory.
    dump = debug_data.DebugDumpDir(self._dump_root, validate=False)
    self.assertTrue(dump.loaded_partition_graphs())

  def testCausalityCheckOnDumpsDetectsWrongTemporalOrder(self):
    with session.Session() as sess:
      u_name = "testDumpCausalityCheck/u"
      v_name = "testDumpCausalityCheck/v"
      w_name = "testDumpCausalityCheck/w"

      u_init = constant_op.constant([2.0, 4.0])
      u = variables.Variable(u_init, name=u_name)
      v = math_ops.add(u, u, name=v_name)
      w = math_ops.add(v, v, name=w_name)

      u.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      sess.run(w, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      # First, loading the original dump without supplying the
      # partition_graphs should not cause a LookupError, validation occurs
      # only with partition_graphs loaded.
      debug_data.DebugDumpDir(self._dump_root)

      # Now, loading the original dump with partition graphs supplied should
      # succeed. The validation should pass quietly.
      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Get the dump file names and compute their timestamps.
      self.assertEqual(
          1, len(dump.get_tensor_file_paths(u_name, 0, "DebugIdentity")))
      u_file_path = dump.get_tensor_file_paths(u_name, 0, "DebugIdentity")[0]

      self.assertEqual(
          1, len(dump.get_tensor_file_paths(v_name, 0, "DebugIdentity")))
      v_file_path = dump.get_tensor_file_paths(v_name, 0, "DebugIdentity")[0]

      u_timestamp = int(u_file_path[u_file_path.rindex("_") + 1:])
      v_timestamp = int(v_file_path[v_file_path.rindex("_") + 1:])

      # Swap the time stamps
      new_u_file_path = u_file_path[:u_file_path.rindex(
          "_")] + "_%d" % v_timestamp
      new_v_file_path = v_file_path[:v_file_path.rindex(
          "_")] + "_%d" % u_timestamp

      os.rename(u_file_path, new_u_file_path)
      os.rename(v_file_path, new_v_file_path)

      # Load the dump directory again. Now a ValueError is expected to be
      # raised due to the timestamp swap.
      with self.assertRaisesRegexp(ValueError, "Causality violated"):
        dump = debug_data.DebugDumpDir(
            self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Loading the dump directory with kwarg "validate" set explicitly to
      # False should get rid of the error.
      dump = debug_data.DebugDumpDir(
          self._dump_root,
          partition_graphs=run_metadata.partition_graphs,
          validate=False)

  def testWatchingOnlyOneOfTwoOutputSlotsDoesNotLeadToCausalityFailure(self):
    with session.Session() as sess:
      x_name = "oneOfTwoSlots/x"
      u_name = "oneOfTwoSlots/u"
      v_name = "oneOfTwoSlots/v"
      w_name = "oneOfTwoSlots/w"
      y_name = "oneOfTwoSlots/y"

      x = variables.Variable([1, 3, 3, 7], dtype=dtypes.int32, name=x_name)
      sess.run(x.initializer)

      unique_x, indices, _ = array_ops.unique_with_counts(x, name=u_name)

      v = math_ops.add(unique_x, unique_x, name=v_name)
      w = math_ops.add(indices, indices, name=w_name)
      y = math_ops.add(w, w, name=y_name)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      # Watch only the first output slot of u, even though it has two output
      # slots.
      debug_utils.add_debug_tensor_watch(
          run_options, u_name, 0, debug_urls=self._debug_urls())
      debug_utils.add_debug_tensor_watch(
          run_options, w_name, 0, debug_urls=self._debug_urls())
      debug_utils.add_debug_tensor_watch(
          run_options, y_name, 0, debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      sess.run([v, y], options=run_options, run_metadata=run_metadata)

      dump = debug_data.DebugDumpDir(
          self._dump_root,
          partition_graphs=run_metadata.partition_graphs,
          validate=True)

      self.assertAllClose([1, 3, 7],
                          dump.get_tensors(u_name, 0, "DebugIdentity")[0])

  def testOutputSlotWithoutOutgoingEdgeCanBeWatched(self):
    """Test watching output slots not attached to any outgoing edges."""

    with session.Session() as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      u = constant_op.constant(u_init_val, shape=[2, 2], name="u")

      # Create a control edge from a node with an output: From u to z.
      # Node u will get executed only because of the control edge. The output
      # tensor u:0 is not attached to any outgoing edge in the graph. This test
      # checks that the debugger can watch such a tensor.
      with ops.control_dependencies([u]):
        z = control_flow_ops.no_op(name="z")

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      sess.run(z, options=run_options, run_metadata=run_metadata)

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Assert that the DebugIdentity watch on u works properly.
      self.assertEqual(1, len(dump.dumped_tensor_data))
      datum = dump.dumped_tensor_data[0]
      self.assertEqual("u", datum.node_name)
      self.assertEqual(0, datum.output_slot)
      self.assertEqual("DebugIdentity", datum.debug_op)
      self.assertAllClose([[5.0, 3.0], [-1.0, 0.0]], datum.get_tensor())

  def testWatchingVariableUpdateOpsSeesUpdatedValues(self):
    """Watch output slots on Variable-updating ops, with no emitted edges."""

    with session.Session() as sess:
      u_init = constant_op.constant(10.0)
      u = variables.Variable(u_init, name="gdo/u")
      v_init = constant_op.constant(20.0)
      v = variables.Variable(v_init, name="gdo/v")

      w = math_ops.multiply(u, v, name="gdo/w")
      # gdo stands for GradientDescentOptimizer.

      train_op = gradient_descent.GradientDescentOptimizer(
          learning_rate=0.1).minimize(
              w, name="gdo/train")

      u.initializer.run()
      v.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      sess.run(train_op, options=run_options, run_metadata=run_metadata)

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      update_u_data = dump.watch_key_to_data(
          "gdo/train/update_gdo/u/ApplyGradientDescent:0:DebugIdentity")
      self.assertEqual(1, len(update_u_data))

      # Gradient descent on u: w = u * v, so dw / du = v.
      # Updated value of u should be:
      #   10.0 - learning_rate * v = 10.0 - 0.1 * 20.0 = 8.0
      self.assertAllClose(8.0, update_u_data[0].get_tensor())

      update_v_data = dump.watch_key_to_data(
          "gdo/train/update_gdo/v/ApplyGradientDescent:0:DebugIdentity")
      self.assertEqual(1, len(update_v_data))

      # Gradient descent on u: w = u * v, so dw / dv = u.
      # Updated value of u should be:
      #   20.0 - learning_rate * u = 20.0 - 0.1 * 10.0 = 19.0
      self.assertAllClose(19.0, update_v_data[0].get_tensor())

      # Verify that the Variables u and v are updated properly.
      self.assertAllClose(8.0, sess.run(u))
      self.assertAllClose(19.0, sess.run(v))

  def testAllowsWatchingUnconnectedOutputTensor(self):
    """Watch an output slot not emitting any edges.

    (Not even control edges from the node.)
    """

    with session.Session() as sess:
      x_init = constant_op.constant([2, 2, 3, 5, 5])
      x = variables.Variable(x_init, name="unconnected/x")

      # The UniqueOp (tf.unique) has two output slots. Use only slot 0 in the
      # graph. Let the debugger watch the unused slot 1.
      unique_x, _ = array_ops.unique(x, name="unconnected/unique_x")
      y = math_ops.add(unique_x, [0, 1, 2], name="unconnected/y")

      x.initializer.run()

      # Verify that only slot 0 of unique_x has recipients, while slot 1 of the
      # same node does not have recipients.
      unique_x_slot_0_recipients = []
      unique_x_slot_1_recipients = []
      for op in sess.graph.get_operations():
        for inp in op.inputs:
          if inp.name == "unconnected/unique_x:0":
            unique_x_slot_0_recipients.append(op.name)
          elif inp.name == "unconnected/unique_x:1":
            unique_x_slot_1_recipients.append(op.name)

      self.assertEqual(["unconnected/y"], unique_x_slot_0_recipients)
      self.assertEqual([], unique_x_slot_1_recipients)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      run_metadata = config_pb2.RunMetadata()
      result = sess.run(y, options=run_options, run_metadata=run_metadata)
      self.assertAllClose([2, 4, 7], result)

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Assert that the connected slot (slot 0) is dumped properly.
      unique_x_slot_0_dumps = dump.watch_key_to_data(
          "unconnected/unique_x:0:DebugIdentity")
      self.assertEqual(1, len(unique_x_slot_0_dumps))
      self.assertEqual("unconnected/unique_x",
                       unique_x_slot_0_dumps[0].node_name)
      self.assertEqual(0, unique_x_slot_0_dumps[0].output_slot)
      self.assertAllClose([2, 3, 5], unique_x_slot_0_dumps[0].get_tensor())

      # Assert that the unconnected slot (slot 1) is dumped properly.
      unique_x_slot_1_dumps = dump.watch_key_to_data(
          "unconnected/unique_x:1:DebugIdentity")
      self.assertEqual(1, len(unique_x_slot_1_dumps))
      self.assertEqual("unconnected/unique_x",
                       unique_x_slot_1_dumps[0].node_name)
      self.assertEqual(1, unique_x_slot_1_dumps[0].output_slot)
      self.assertAllClose([0, 0, 1, 2, 2],
                          unique_x_slot_1_dumps[0].get_tensor())

  def testDebuggingDuringOpError(self):
    """Test the debug tensor dumping when error occurs in graph runtime."""

    with session.Session() as sess:
      ph = array_ops.placeholder(dtypes.float32, name="mismatch/ph")
      x = array_ops.transpose(ph, name="mismatch/x")
      m = constant_op.constant(
          np.array(
              [[1.0, 2.0]], dtype=np.float32), name="mismatch/m")
      y = math_ops.matmul(m, x, name="mismatch/y")

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=self._debug_urls())

      with self.assertRaises(errors.OpError):
        sess.run(y,
                 options=run_options,
                 feed_dict={ph: np.array([[-3.0], [0.0]])})

      dump = debug_data.DebugDumpDir(self._dump_root)

      # Despite the fact that the run() call errored out and partition_graphs
      # are not available via run_metadata, the partition graphs should still
      # have been loaded from the dump directory.
      self.assertTrue(dump.loaded_partition_graphs())

      m_dumps = dump.watch_key_to_data("mismatch/m:0:DebugIdentity")
      self.assertEqual(1, len(m_dumps))
      self.assertAllClose(np.array([[1.0, 2.0]]), m_dumps[0].get_tensor())

      x_dumps = dump.watch_key_to_data("mismatch/x:0:DebugIdentity")
      self.assertEqual(1, len(x_dumps))
      self.assertAllClose(np.array([[-3.0, 0.0]]), x_dumps[0].get_tensor())

  def testDebugNumericSummaryOnInitializedTensorGivesCorrectResult(self):
    with session.Session() as sess:
      a = variables.Variable(
          [
              np.nan, np.nan, 0.0, 0.0, 0.0, -1.0, -3.0, 3.0, 7.0, -np.inf,
              -np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.nan, np.nan
          ],
          dtype=np.float32,
          name="numeric_summary/a")
      b = variables.Variable(
          [0.0] * 18, dtype=np.float32, name="numeric_summary/b")
      c = math_ops.add(a, b, name="numeric_summary/c")

      sess.run(variables.global_variables_initializer())

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugNumericSummary"],
          debug_urls=self._debug_urls())

      sess.run(c, options=run_options, run_metadata=run_metadata)

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)
      self.assertTrue(dump.loaded_partition_graphs())

      self.assertAllClose([[
          1.0, 18.0, 2.0, 2.0, 3.0, 2.0, 5.0, 4.0, -3.0, 7.0, 0.85714286,
          8.97959184
      ]], dump.get_tensors("numeric_summary/a/read", 0, "DebugNumericSummary"))

  def testDebugNumericSummaryOnUninitializedTensorGivesCorrectResult(self):
    with session.Session() as sess:
      a = variables.Variable(
          [42], dtype=np.float32, name="numeric_summary_uninit/a")

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugNumericSummary"],
          debug_urls=self._debug_urls())

      sess.run(a.initializer, options=run_options, run_metadata=run_metadata)

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)
      self.assertTrue(dump.loaded_partition_graphs())

      # DebugNumericSummary output should reflect the uninitialized state of
      # the watched tensor.
      numeric_summary = dump.get_tensors("numeric_summary_uninit/a", 0,
                                         "DebugNumericSummary")[0]
      self.assertAllClose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          numeric_summary[0:8])
      self.assertTrue(np.isinf(numeric_summary[8]))
      self.assertGreater(numeric_summary[8], 0.0)
      self.assertTrue(np.isinf(numeric_summary[9]))
      self.assertLess(numeric_summary[9], 0.0)
      self.assertTrue(np.isnan(numeric_summary[10]))
      self.assertTrue(np.isnan(numeric_summary[11]))

  def testLookUpNodePythonTracebackWorks(self):
    with session.Session() as sess:
      u_init = constant_op.constant(10.0)
      u = variables.Variable(u_init, name="traceback/u")
      v_init = constant_op.constant(20.0)
      v = variables.Variable(v_init, name="traceback/v")

      w = math_ops.multiply(u, v, name="traceback/w")

      sess.run(variables.global_variables_initializer())

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options, sess.graph, debug_urls=self._debug_urls())

      sess.run(w, options=run_options, run_metadata=run_metadata)
      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Prior to setting the Python graph, attempts to do traceback lookup
      # should lead to exceptions.
      with self.assertRaisesRegexp(
          LookupError, "Python graph is not available for traceback lookup"):
        dump.node_traceback("traceback/w")

      dump.set_python_graph(sess.graph)

      # After setting the Python graph, attempts to look up nonexistent nodes
      # should lead to exceptions.
      with self.assertRaisesRegexp(KeyError,
                                   r"Cannot find node \"foo\" in Python graph"):
        dump.node_traceback("foo")

      # Lookup should work with node name input.
      traceback = dump.node_traceback("traceback/w")
      self.assertIsInstance(traceback, list)
      self.assertGreater(len(traceback), 0)
      for trace in traceback:
        self.assertIsInstance(trace, tuple)

      # Lookup should also work with tensor name input.
      traceback = dump.node_traceback("traceback/w:0")
      self.assertIsInstance(traceback, list)
      self.assertGreater(len(traceback), 0)
      for trace in traceback:
        self.assertIsInstance(trace, tuple)


if __name__ == "__main__":
  googletest.main()
