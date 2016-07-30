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
"""Tests for debugger functionalities in tf.Session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import shutil
import tempfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class SessionDebugTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.dump_root_ = tempfile.mkdtemp()

  def tearDown(self):
    # Tear down temporary dump directory.
    shutil.rmtree(self.dump_root_)

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

  def _verifyTensorDumpFile(self, dump_file, expected_tensor_name, debug_op,
                            wall_time_lower_bound, expected_tensor_val):
    """Helper method: Verify a Tensor debug dump file and its content.

    Args:
      dump_file: Path to the dump file.
      expected_tensor_name: Expected name of the tensor, e.g., node_a:0.
      debug_op: Name of the debug Op, e.g., DebugIdentity.
      wall_time_lower_bound: Lower bound of the wall time.
      expected_tensor_val: Expected tensor value, as a numpy array.
    """
    self.assertTrue(os.path.isfile(dump_file))

    event = event_pb2.Event()
    f = open(dump_file, "rb")
    event.ParseFromString(f.read())

    wall_time = event.wall_time
    debg_node_name = event.summary.value[0].node_name

    tensor_value = tensor_util.MakeNdarray(event.summary.value[0].tensor)

    self.assertGreater(wall_time, wall_time_lower_bound)
    self.assertEqual("%s:%s" % (expected_tensor_name, debug_op), debg_node_name)

    if expected_tensor_val.dtype.type is np.string_:
      self.assertEqual(str(expected_tensor_val), str(tensor_value))
    else:
      self.assertAllClose(expected_tensor_val, tensor_value)

  def testDumpToFileOverlaoppinpParentDir(self):
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

      run_options = config_pb2.RunOptions()
      debug_url = "file://%s" % self.dump_root_

      # Add debug tensor watch for u.
      self._addDebugTensorWatch(
          run_options, "%s/read" % u_name, 0, debug_urls=[debug_url])
      # Add debug tensor watch for v.
      self._addDebugTensorWatch(
          run_options, "%s/read" % v_name, 0, debug_urls=[debug_url])

      run_metadata = config_pb2.RunMetadata()

      # Invoke Session.run().
      sess.run(w, options=run_options, run_metadata=run_metadata)

      # Verify the dump file for u.
      dump_files = os.listdir(os.path.join(self.dump_root_, u_name))
      self.assertEqual(1, len(dump_files))
      self.assertTrue(dump_files[0].startswith("read_0_"))

      dump_file = os.path.join(self.dump_root_, u_name, dump_files[0])
      self._verifyTensorDumpFile(dump_file, "%s/read:0" % u_name,
                                 "DebugIdentity", 0, u_init_val)

      # Verify the dump file for v.
      dump_files = os.listdir(os.path.join(self.dump_root_, v_name))
      self.assertEqual(1, len(dump_files))
      self.assertTrue(dump_files[0].startswith("read_0_"))

      dump_file = os.path.join(self.dump_root_, v_name, dump_files[0])
      self._verifyTensorDumpFile(dump_file, "%s/read:0" % v_name,
                                 "DebugIdentity", 0, v_init_val)

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

      run_options = config_pb2.RunOptions()
      debug_url = "file://%s" % self.dump_root_

      # Add debug tensor watch for u.
      self._addDebugTensorWatch(
          run_options, "%s/read" % str1_name, 0, debug_urls=[debug_url])
      # Add debug tensor watch for v.
      self._addDebugTensorWatch(
          run_options, "%s/read" % str2_name, 0, debug_urls=[debug_url])

      run_metadata = config_pb2.RunMetadata()

      # Invoke Session.run().
      sess.run(str_concat, options=run_options, run_metadata=run_metadata)

      # Verify the dump file for str1.
      dump_files = os.listdir(os.path.join(self.dump_root_, str1_name))
      self.assertEqual(1, len(dump_files))
      self.assertTrue(dump_files[0].startswith("read_0_"))
      dump_file = os.path.join(self.dump_root_, str1_name, dump_files[0])
      self._verifyTensorDumpFile(dump_file, "%s/read:0" % str1_name,
                                 "DebugIdentity", 0, str1_init_val)

      # Verify the dump file for str2.
      dump_files = os.listdir(os.path.join(self.dump_root_, str2_name))
      self.assertEqual(1, len(dump_files))
      self.assertTrue(dump_files[0].startswith("read_0_"))
      dump_file = os.path.join(self.dump_root_, str2_name, dump_files[0])
      self._verifyTensorDumpFile(dump_file, "%s/read:0" % str2_name,
                                 "DebugIdentity", 0, str2_init_val)

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
      run_options = config_pb2.RunOptions()
      debug_url = "file://%s" % self.dump_root_

      # Add debug tensor watch for u.
      self._addDebugTensorWatch(run_options, u_name, 0, debug_urls=[debug_url])
      # Add debug tensor watch for v.
      self._addDebugTensorWatch(
          run_options, "%s/read" % v_name, 0, debug_urls=[debug_url])
      # Add debug tensor watch for while/Identity.
      self._addDebugTensorWatch(
          run_options, "while/Identity", 0, debug_urls=[debug_url])

      run_metadata = config_pb2.RunMetadata()

      r = sess.run(loop, options=run_options, run_metadata=run_metadata)

      self.assertEqual(num_iter, r)

      u_val_final = sess.run(u)
      self.assertAllClose(u_init_val + num_iter * v_init_val, u_val_final)

      # Verify dump files
      self.assertTrue(os.path.isdir(self.dump_root_))

      self.assertTrue(os.path.isdir(os.path.join(self.dump_root_, u_namespace)))
      self.assertTrue(
          os.path.isdir(os.path.join(self.dump_root_, v_namespace, "v")))

      # Verify the dump file for tensor "u".
      dump_files = glob.glob(
          os.path.join(self.dump_root_, u_namespace, "u_0_*"))
      self.assertEqual(1, len(dump_files))
      dump_file = os.path.join(self.dump_root_, u_namespace, dump_files[0])
      self.assertTrue(os.path.isfile(dump_file))
      self._verifyTensorDumpFile(dump_file, "%s:0" % u_name, "DebugIdentity", 0,
                                 u_init_val)

      # Verify the dump file for tensor "v".
      dump_files = os.listdir(os.path.join(self.dump_root_, v_name))
      self.assertEqual(1, len(dump_files))
      self.assertTrue(dump_files[0].startswith("read_0_"))

      dump_file = os.path.join(self.dump_root_, v_name, dump_files[0])
      self._verifyTensorDumpFile(dump_file, "%s/read:0" % v_name,
                                 "DebugIdentity", 0, v_init_val)

      # Verify the dump files for tensor while/Identity
      while_identity_dump_files = sorted(
          os.listdir(os.path.join(self.dump_root_, "while")))
      self.assertEqual(num_iter, len(while_identity_dump_files))

      # Verify the content of the individual
      for k in xrange(len(while_identity_dump_files)):
        dump_file_path = os.path.join(self.dump_root_, "while",
                                      while_identity_dump_files[k])
        self._verifyTensorDumpFile(dump_file_path, "while/Identity:0",
                                   "DebugIdentity", 0, np.array(k))


if __name__ == "__main__":
  googletest.main()
