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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class SessionDebugTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._dump_root = tempfile.mkdtemp()

    if test.is_gpu_available():
      self._expected_partition_graph_count = 2
      self._expected_num_devices = 2
      self._main_device = "/job:localhost/replica:0/task:0/gpu:0"
    else:
      self._expected_partition_graph_count = 1
      self._expected_num_devices = 1
      self._main_device = "/job:localhost/replica:0/task:0/cpu:0"

  def tearDown(self):
    # Tear down temporary dump directory.
    shutil.rmtree(self._dump_root)

  def _addDebugTensorWatch(self,
                           run_opts,
                           node_name,
                           output_slot,
                           debug_op="DebugIdentity",
                           debug_urls=None):
    watch_opts = run_opts.debug_tensor_watch_opts

    # Add debug tensor watch for u.
    watch = watch_opts.add()
    watch.node_name = node_name
    watch.output_slot = 0
    watch.debug_ops.append(debug_op)

    if debug_urls:
      for debug_url in debug_urls:
        watch.debug_urls.append(debug_url)

  def testDumpToFileOverlappingParentDir(self):
    with session.Session() as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      v_init_val = np.array([[2.0], [-1.0]])

      # Use node names with overlapping namespace (i.e., parent directory) to
      # test concurrent, non-racing directory creation.
      u_name = "testDumpToFile/u"
      v_name = "testDumpToFile/v"

      u_init = constant_op.constant(u_init_val, shape=[2, 2])
      u = variables.Variable(u_init, name=u_name)
      v_init = constant_op.constant(v_init_val, shape=[2, 1])
      v = variables.Variable(v_init, name=v_name)

      w = math_ops.matmul(u, v, name="testDumpToFile/matmul")

      u.initializer.run()
      v.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_url = "file://%s" % self._dump_root

      # Add debug tensor watch for u.
      self._addDebugTensorWatch(
          run_options, "%s/read" % u_name, 0, debug_urls=[debug_url])
      # Add debug tensor watch for v.
      self._addDebugTensorWatch(
          run_options, "%s/read" % v_name, 0, debug_urls=[debug_url])

      run_metadata = config_pb2.RunMetadata()

      # Invoke Session.run().
      sess.run(w, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      # Verify the dumped tensor values for u and v.
      self.assertEqual(2, dump.size)

      self.assertAllClose([u_init_val], dump.get_tensors("%s/read" % u_name, 0,
                                                         "DebugIdentity"))
      self.assertAllClose([v_init_val], dump.get_tensors("%s/read" % v_name, 0,
                                                         "DebugIdentity"))

      self.assertGreaterEqual(
          dump.get_rel_timestamps("%s/read" % u_name, 0, "DebugIdentity")[0], 0)
      self.assertGreaterEqual(
          dump.get_rel_timestamps("%s/read" % v_name, 0, "DebugIdentity")[0], 0)

  def testDumpStringTensorsToFileSystem(self):
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
      debug_url = "file://%s" % self._dump_root

      # Add debug tensor watch for u.
      self._addDebugTensorWatch(
          run_options, "%s/read" % str1_name, 0, debug_urls=[debug_url])
      # Add debug tensor watch for v.
      self._addDebugTensorWatch(
          run_options, "%s/read" % str2_name, 0, debug_urls=[debug_url])

      run_metadata = config_pb2.RunMetadata()
      sess.run(str_concat, options=run_options, run_metadata=run_metadata)

      # String ops are located on CPU.
      self.assertEqual(1, len(run_metadata.partition_graphs))

      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

      self.assertEqual(2, dump.size)

      self.assertEqual([str1_init_val], dump.get_tensors("%s/read" % str1_name,
                                                         0, "DebugIdentity"))
      self.assertEqual([str2_init_val], dump.get_tensors("%s/read" % str2_name,
                                                         0, "DebugIdentity"))

      self.assertGreaterEqual(
          dump.get_rel_timestamps("%s/read" % str1_name, 0, "DebugIdentity")[0],
          0)
      self.assertGreaterEqual(
          dump.get_rel_timestamps("%s/read" % str2_name, 0, "DebugIdentity")[0],
          0)

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
      debug_url = "file://%s" % self._dump_root

      # Add debug tensor watch for u.
      self._addDebugTensorWatch(
          run_options, "%s" % u_name, 0, debug_urls=[debug_url])
      self._addDebugTensorWatch(
          run_options, "%s" % s_name, 0, debug_urls=[debug_url])

      run_metadata = config_pb2.RunMetadata()

      # Initialize u and s.
      sess.run(variables.initialize_all_variables(),
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

  def testDumpToFileWhileLoop(self):
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
      debug_url = "file://%s" % self._dump_root

      # Add debug tensor watch for u.
      self._addDebugTensorWatch(run_options, u_name, 0, debug_urls=[debug_url])
      # Add debug tensor watch for v.
      self._addDebugTensorWatch(
          run_options, "%s/read" % v_name, 0, debug_urls=[debug_url])
      # Add debug tensor watch for while/Identity.
      self._addDebugTensorWatch(
          run_options, "while/Identity", 0, debug_urls=[debug_url])
      # Add debug tensor watch for while/Add/y.
      self._addDebugTensorWatch(
          run_options, "while/Add/y", 0, debug_urls=[debug_url])

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
      self.assertAllClose([u_init_val], dump.get_tensors(u_name, 0,
                                                         "DebugIdentity"))
      self.assertAllClose([v_init_val], dump.get_tensors("%s/read" % v_name, 0,
                                                         "DebugIdentity"))

      while_id_tensors = dump.get_tensors("while/Identity", 0, "DebugIdentity")
      self.assertEqual(10, len(while_id_tensors))
      for k in xrange(len(while_id_tensors)):
        self.assertAllClose(np.array(k), while_id_tensors[k])

      # Verify ascending timestamps from the while loops.
      while_id_rel_timestamps = dump.get_rel_timestamps("while/Identity", 0,
                                                        "DebugIdentity")
      self.assertEqual(10, len(while_id_rel_timestamps))
      prev_rel_time = 0
      for rel_time in while_id_rel_timestamps:
        self.assertGreaterEqual(rel_time, prev_rel_time)
        prev_rel_time = rel_time

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
      w = math_ops.sub(u, v, name=w_name)

      # Expected output: [inf, 1.3333]
      x = math_ops.div(u, w, name=x_name)

      # Expected output: [nan, 4.0]
      y = math_ops.mul(w, x, name=y_name)

      z = math_ops.mul(y, y, name=z_name)

      u.initializer.run()
      v.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls="file://%s" % self._dump_root)

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

  def testDumpGraphStructureLookup(self):
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
          debug_urls="file://%s" % self._dump_root)

      run_metadata = config_pb2.RunMetadata()
      sess.run(w, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))
      dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs)

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
          set([u_name, u_read_name, v_name]),
          set(dump.transitive_inputs(w_name)))

      with self.assertRaisesRegexp(ValueError,
                                   "does not exist in partition graphs"):
        dump.transitive_inputs(u_name + "foo")

      # Test num_devices().
      self.assertEqual(self._expected_num_devices, len(dump.devices()))

      # Test node_device().
      self.assertEqual(self._main_device, dump.node_device(u_name))

      with self.assertRaisesRegexp(ValueError,
                                   "does not exist in partition graphs"):
        dump.node_device(u_name + "foo")

      # Test node_op_type().
      self.assertEqual("Variable", dump.node_op_type(u_name))
      self.assertEqual("Identity", dump.node_op_type(u_name + "/read"))
      self.assertEqual("Add", dump.node_op_type(v_name))
      self.assertEqual("Add", dump.node_op_type(w_name))

      with self.assertRaisesRegexp(ValueError,
                                   "does not exist in partition graphs"):
        dump.node_op_type(u_name + "foo")

      # Now load the dump again, without the parition graphs, so we can check
      # the errors raised for no partition graphs loaded.
      dump = debug_data.DebugDumpDir(self._dump_root, validate=False)

      with self.assertRaisesRegexp(RuntimeError,
                                   "No partition graphs have been loaded"):
        dump.partition_graphs()

      with self.assertRaisesRegexp(
          RuntimeError, "Node inputs are not loaded from partiton graphs yet"):
        dump.node_inputs(u_name)

      with self.assertRaisesRegexp(RuntimeError,
                                   "No partition graphs have been loaded"):
        dump.nodes()

      with self.assertRaisesRegexp(
          RuntimeError,
          "Node recipients are not loaded from partiton graphs yet"):
        dump.node_recipients(u_name)

      with self.assertRaisesRegexp(
          RuntimeError, "Node inputs are not loaded from partiton graphs yet"):
        dump.transitive_inputs(u_name)

      with self.assertRaisesRegexp(
          RuntimeError, "Devices are not loaded from partiton graphs yet"):
        dump.devices()

      with self.assertRaisesRegexp(
          RuntimeError, "Node devices are not loaded from partiton graphs yet"):
        dump.node_device(u_name)

      with self.assertRaisesRegexp(
          RuntimeError,
          "Node op types are not loaded from partiton graphs yet"):
        dump.node_op_type(u_name)

  def testDumpCausalityCheck(self):
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
          debug_urls="file://%s" % self._dump_root)

      run_metadata = config_pb2.RunMetadata()
      sess.run(w, options=run_options, run_metadata=run_metadata)

      self.assertEqual(self._expected_partition_graph_count,
                       len(run_metadata.partition_graphs))

      # First, loading the original dump without supplying the
      # partition_graphs should not cause a RuntimeError, validation occurs
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


if __name__ == "__main__":
  googletest.main()
