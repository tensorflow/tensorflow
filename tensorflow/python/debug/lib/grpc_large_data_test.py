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
"""Tests for sending large-size data through tfdbg grpc channels.

"Large-size data" includes large GraphDef protos and large Tensor protos.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.debug.lib import grpc_debug_test_server
from tensorflow.python.debug.lib import session_debug_testlib
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class LargeGraphAndLargeTensorsDebugTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    (cls.debug_server_port, cls.debug_server_url, _, cls.debug_server_thread,
     cls.debug_server
    ) = grpc_debug_test_server.start_server_on_separate_thread(
        dump_to_filesystem=False)
    tf_logging.info("debug server url: %s", cls.debug_server_url)

  @classmethod
  def tearDownClass(cls):
    cls.debug_server.stop_server().wait()
    cls.debug_server_thread.join()

  def tearDown(self):
    ops.reset_default_graph()
    self.debug_server.clear_data()

  def testSendingLargeGraphDefsWorks(self):
    with self.test_session(
        use_gpu=True,
        config=session_debug_testlib.no_rewrite_session_config()) as sess:
      u = variables.Variable(42.0, name="original_u")
      for _ in xrange(50 * 1000):
        u = array_ops.identity(u)
      sess.run(variables.global_variables_initializer())

      def watch_fn(fetches, feeds):
        del fetches, feeds
        return framework.WatchOptions(
            debug_ops=["DebugIdentity"],
            node_name_regex_whitelist=r"original_u")
      sess = grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % self.debug_server_port, watch_fn=watch_fn)
      self.assertAllClose(42.0, sess.run(u))

      self.assertAllClose(
          [42.0],
          self.debug_server.debug_tensor_values["original_u:0:DebugIdentity"])
      self.assertEqual(2 if test.is_gpu_available() else 1,
                       len(self.debug_server.partition_graph_defs))
      max_graph_def_size = max([
          len(graph_def.SerializeToString())
          for graph_def in self.debug_server.partition_graph_defs])
      self.assertGreater(max_graph_def_size, 4 * 1024 * 1024)

  def testSendingLargeFloatTensorWorks(self):
    with self.test_session(
        use_gpu=True,
        config=session_debug_testlib.no_rewrite_session_config()) as sess:
      u_init_val_array = list(xrange(1200 * 1024))
      # Size: 4 * 1200 * 1024 = 4800k > 4M

      u_init = constant_op.constant(
          u_init_val_array, dtype=dtypes.float32, name="u_init")
      u = variables.Variable(u_init, name="u")

      def watch_fn(fetches, feeds):
        del fetches, feeds  # Unused by this watch_fn.
        return framework.WatchOptions(
            debug_ops=["DebugIdentity"],
            node_name_regex_whitelist=r"u_init")
      sess = grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % self.debug_server_port, watch_fn=watch_fn)
      sess.run(u.initializer)

      self.assertAllEqual(
          u_init_val_array,
          self.debug_server.debug_tensor_values["u_init:0:DebugIdentity"][0])

  def testSendingStringTensorWithAlmostTooLargeStringsWorks(self):
    with self.test_session(
        use_gpu=True,
        config=session_debug_testlib.no_rewrite_session_config()) as sess:
      u_init_val = [
          b"", b"spam", b"A" * 2500 * 1024, b"B" * 2500 * 1024, b"egg", b""]
      u_init = constant_op.constant(
          u_init_val, dtype=dtypes.string, name="u_init")
      u = variables.Variable(u_init, name="u")

      def watch_fn(fetches, feeds):
        del fetches, feeds
        return framework.WatchOptions(
            debug_ops=["DebugIdentity"],
            node_name_regex_whitelist=r"u_init")
      sess = grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % self.debug_server_port, watch_fn=watch_fn)
      sess.run(u.initializer)

      self.assertAllEqual(
          u_init_val,
          self.debug_server.debug_tensor_values["u_init:0:DebugIdentity"][0])

  def testSendingLargeStringTensorWorks(self):
    with self.test_session(
        use_gpu=True,
        config=session_debug_testlib.no_rewrite_session_config()) as sess:
      strs_total_size_threshold = 5000 * 1024
      cum_size = 0
      u_init_val_array = []
      while cum_size < strs_total_size_threshold:
        strlen = np.random.randint(200)
        u_init_val_array.append(b"A" * strlen)
        cum_size += strlen

      u_init = constant_op.constant(
          u_init_val_array, dtype=dtypes.string, name="u_init")
      u = variables.Variable(u_init, name="u")

      def watch_fn(fetches, feeds):
        del fetches, feeds
        return framework.WatchOptions(
            debug_ops=["DebugIdentity"],
            node_name_regex_whitelist=r"u_init")
      sess = grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % self.debug_server_port, watch_fn=watch_fn)
      sess.run(u.initializer)

      self.assertAllEqual(
          u_init_val_array,
          self.debug_server.debug_tensor_values["u_init:0:DebugIdentity"][0])

  def testSendingEmptyFloatTensorWorks(self):
    with self.test_session(
        use_gpu=True,
        config=session_debug_testlib.no_rewrite_session_config()) as sess:
      u_init = constant_op.constant(
          [], dtype=dtypes.float32, shape=[0], name="u_init")
      u = variables.Variable(u_init, name="u")

      def watch_fn(fetches, feeds):
        del fetches, feeds
        return framework.WatchOptions(
            debug_ops=["DebugIdentity"],
            node_name_regex_whitelist=r"u_init")
      sess = grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % self.debug_server_port, watch_fn=watch_fn)
      sess.run(u.initializer)

      u_init_value = self.debug_server.debug_tensor_values[
          "u_init:0:DebugIdentity"][0]
      self.assertEqual(np.float32, u_init_value.dtype)
      self.assertEqual(0, len(u_init_value))

  def testSendingEmptyStringTensorWorks(self):
    with self.test_session(
        use_gpu=True,
        config=session_debug_testlib.no_rewrite_session_config()) as sess:
      u_init = constant_op.constant(
          [], dtype=dtypes.string, shape=[0], name="u_init")
      u = variables.Variable(u_init, name="u")

      def watch_fn(fetches, feeds):
        del fetches, feeds
        return framework.WatchOptions(
            debug_ops=["DebugIdentity"],
            node_name_regex_whitelist=r"u_init")
      sess = grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % self.debug_server_port, watch_fn=watch_fn)
      sess.run(u.initializer)

      u_init_value = self.debug_server.debug_tensor_values[
          "u_init:0:DebugIdentity"][0]
      self.assertEqual(np.object, u_init_value.dtype)
      self.assertEqual(0, len(u_init_value))


if __name__ == "__main__":
  googletest.main()
