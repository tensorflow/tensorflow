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
"""Tests for debugger functionalities in tf.Session with grpc:// URLs.

This test file focuses on the grpc:// debugging of local (non-distributed)
tf.Sessions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_test_server
from tensorflow.python.debug.lib import session_debug_testlib
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.debug.wrappers import hooks
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session


def no_rewrite_session_config():
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


class GrpcDebugServerTest(test_util.TensorFlowTestCase):

  def testRepeatedRunServerRaisesException(self):
    (_, _, _, server_thread,
     server) = grpc_debug_test_server.start_server_on_separate_thread(
         poll_server=True)
    # The server is started asynchronously. It needs to be polled till its state
    # has become started.

    with self.assertRaisesRegexp(
        ValueError, "Server has already started running"):
      server.run_server()

    server.stop_server().wait()
    server_thread.join()

  def testRepeatedStopServerRaisesException(self):
    (_, _, _, server_thread,
     server) = grpc_debug_test_server.start_server_on_separate_thread(
         poll_server=True)
    server.stop_server().wait()
    server_thread.join()

    with self.assertRaisesRegexp(ValueError, "Server has already stopped"):
      server.stop_server().wait()

  def testRunServerAfterStopRaisesException(self):
    (_, _, _, server_thread,
     server) = grpc_debug_test_server.start_server_on_separate_thread(
         poll_server=True)
    server.stop_server().wait()
    server_thread.join()

    with self.assertRaisesRegexp(ValueError, "Server has already stopped"):
      server.run_server()

  def testStartServerWithoutBlocking(self):
    (_, _, _, server_thread,
     server) = grpc_debug_test_server.start_server_on_separate_thread(
         poll_server=True, blocking=False)
    # The thread that starts the server shouldn't block, so we should be able to
    # join it before stopping the server.
    server_thread.join()
    server.stop_server().wait()


class SessionDebugGrpcTest(session_debug_testlib.SessionDebugTestBase):

  @classmethod
  def setUpClass(cls):
    session_debug_testlib.SessionDebugTestBase.setUpClass()
    (cls._server_port, cls._debug_server_url, cls._server_dump_dir,
     cls._server_thread,
     cls._server) = grpc_debug_test_server.start_server_on_separate_thread()

  @classmethod
  def tearDownClass(cls):
    # Stop the test server and join the thread.
    cls._server.stop_server().wait()
    cls._server_thread.join()

    session_debug_testlib.SessionDebugTestBase.tearDownClass()

  def setUp(self):
    # Override the dump root as the test server's dump directory.
    self._dump_root = self._server_dump_dir

  def tearDown(self):
    if os.path.isdir(self._server_dump_dir):
      shutil.rmtree(self._server_dump_dir)
    session_debug_testlib.SessionDebugTestBase.tearDown(self)

  def _debug_urls(self, run_number=None):
    return ["grpc://localhost:%d" % self._server_port]

  def _debug_dump_dir(self, run_number=None):
    if run_number is None:
      return self._dump_root
    else:
      return os.path.join(self._dump_root, "run_%d" % run_number)

  def testConstructGrpcDebugWrapperSessionWithInvalidTypeRaisesException(self):
    sess = session.Session(config=no_rewrite_session_config())
    with self.assertRaisesRegexp(
        TypeError, "Expected type str or list in grpc_debug_server_addresses"):
      grpc_wrapper.GrpcDebugWrapperSession(sess, 1337)

  def testConstructGrpcDebugWrapperSessionWithInvalidTypeRaisesException2(self):
    sess = session.Session(config=no_rewrite_session_config())
    with self.assertRaisesRegexp(
        TypeError, "Expected type str in list grpc_debug_server_addresses"):
      grpc_wrapper.GrpcDebugWrapperSession(sess, ["localhost:1337", 1338])

  def testUseInvalidWatchFnTypeWithGrpcDebugWrapperSessionRaisesException(self):
    sess = session.Session(config=no_rewrite_session_config())
    with self.assertRaises(TypeError):
      grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % self._server_port, watch_fn="foo")

  def testGrpcDebugWrapperSessionWithoutWatchFnWorks(self):
    u = variables.Variable(2.1, name="u")
    v = variables.Variable(20.0, name="v")
    w = math_ops.multiply(u, v, name="w")

    sess = session.Session(config=no_rewrite_session_config())
    sess.run(u.initializer)
    sess.run(v.initializer)

    sess = grpc_wrapper.GrpcDebugWrapperSession(
        sess, "localhost:%d" % self._server_port)
    w_result = sess.run(w)
    self.assertAllClose(42.0, w_result)

    dump = debug_data.DebugDumpDir(self._dump_root)
    self.assertEqual(5, dump.size)
    self.assertAllClose([2.1], dump.get_tensors("u", 0, "DebugIdentity"))
    self.assertAllClose([2.1], dump.get_tensors("u/read", 0, "DebugIdentity"))
    self.assertAllClose([20.0], dump.get_tensors("v", 0, "DebugIdentity"))
    self.assertAllClose([20.0], dump.get_tensors("v/read", 0, "DebugIdentity"))
    self.assertAllClose([42.0], dump.get_tensors("w", 0, "DebugIdentity"))

  def testGrpcDebugWrapperSessionWithWatchFnWorks(self):
    def watch_fn(feeds, fetch_keys):
      del feeds, fetch_keys
      return ["DebugIdentity", "DebugNumericSummary"], r".*/read", None

    u = variables.Variable(2.1, name="u")
    v = variables.Variable(20.0, name="v")
    w = math_ops.multiply(u, v, name="w")

    sess = session.Session(config=no_rewrite_session_config())
    sess.run(u.initializer)
    sess.run(v.initializer)

    sess = grpc_wrapper.GrpcDebugWrapperSession(
        sess, "localhost:%d" % self._server_port, watch_fn=watch_fn)
    w_result = sess.run(w)
    self.assertAllClose(42.0, w_result)

    dump = debug_data.DebugDumpDir(self._dump_root)
    self.assertEqual(4, dump.size)
    self.assertAllClose([2.1], dump.get_tensors("u/read", 0, "DebugIdentity"))
    self.assertEqual(
        14, len(dump.get_tensors("u/read", 0, "DebugNumericSummary")[0]))
    self.assertAllClose([20.0], dump.get_tensors("v/read", 0, "DebugIdentity"))
    self.assertEqual(
        14, len(dump.get_tensors("v/read", 0, "DebugNumericSummary")[0]))

  def testGrpcDebugHookWithStatelessWatchFnWorks(self):
    # Perform some set up. Specifically, construct a simple TensorFlow graph and
    # create a watch function for certain ops.
    def watch_fn(feeds, fetch_keys):
      del feeds, fetch_keys
      return framework.WatchOptions(
          debug_ops=["DebugIdentity", "DebugNumericSummary"],
          node_name_regex_whitelist=r".*/read",
          op_type_regex_whitelist=None,
          tolerate_debug_op_creation_failures=True)

    u = variables.Variable(2.1, name="u")
    v = variables.Variable(20.0, name="v")
    w = math_ops.multiply(u, v, name="w")

    sess = session.Session(config=no_rewrite_session_config())
    sess.run(u.initializer)
    sess.run(v.initializer)

    # Create a hook. One could use this hook with say a tflearn Estimator.
    # However, we use a HookedSession in this test to avoid depending on the
    # internal implementation of Estimators.
    grpc_debug_hook = hooks.GrpcDebugHook(
        ["localhost:%d" % self._server_port], watch_fn=watch_fn)
    sess = monitored_session._HookedSession(sess, [grpc_debug_hook])

    # Run the hooked session. This should stream tensor data to the GRPC
    # endpoints.
    w_result = sess.run(w)

    # Verify that the hook monitored the correct tensors.
    self.assertAllClose(42.0, w_result)
    dump = debug_data.DebugDumpDir(self._dump_root)
    self.assertEqual(4, dump.size)
    self.assertAllClose([2.1], dump.get_tensors("u/read", 0, "DebugIdentity"))
    self.assertEqual(
        14, len(dump.get_tensors("u/read", 0, "DebugNumericSummary")[0]))
    self.assertAllClose([20.0], dump.get_tensors("v/read", 0, "DebugIdentity"))
    self.assertEqual(
        14, len(dump.get_tensors("v/read", 0, "DebugNumericSummary")[0]))

  def testConstructGrpcDebugHookWithGrpcInUrlRaisesValueError(self):
    """Tests that the hook raises an error if the URL starts with grpc://."""
    with self.assertRaises(ValueError):
      hooks.GrpcDebugHook(["grpc://foo:42"])


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
        use_gpu=True, config=no_rewrite_session_config()) as sess:
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
        use_gpu=True, config=no_rewrite_session_config()) as sess:
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
        use_gpu=True, config=no_rewrite_session_config()) as sess:
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
        use_gpu=True, config=no_rewrite_session_config()) as sess:
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
        use_gpu=True, config=no_rewrite_session_config()) as sess:
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
        use_gpu=True, config=no_rewrite_session_config()) as sess:
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


class SessionDebugConcurrentTest(
    session_debug_testlib.DebugConcurrentRunCallsTest):

  @classmethod
  def setUpClass(cls):
    session_debug_testlib.SessionDebugTestBase.setUpClass()
    (cls._server_port, cls._debug_server_url, cls._server_dump_dir,
     cls._server_thread,
     cls._server) = grpc_debug_test_server.start_server_on_separate_thread()

  @classmethod
  def tearDownClass(cls):
    # Stop the test server and join the thread.
    cls._server.stop_server().wait()
    cls._server_thread.join()
    session_debug_testlib.SessionDebugTestBase.tearDownClass()

  def setUp(self):
    self._num_concurrent_runs = 3
    self._dump_roots = []
    for i in range(self._num_concurrent_runs):
      self._dump_roots.append(
          os.path.join(self._server_dump_dir, "thread%d" % i))

  def tearDown(self):
    ops.reset_default_graph()
    if os.path.isdir(self._server_dump_dir):
      shutil.rmtree(self._server_dump_dir)

  def _get_concurrent_debug_urls(self):
    urls = []
    for i in range(self._num_concurrent_runs):
      urls.append(self._debug_server_url + "/thread%d" % i)
    return urls


class SessionDebugGrpcGatingTest(test_util.TensorFlowTestCase):
  """Test server gating of debug ops."""

  @classmethod
  def setUpClass(cls):
    (cls._server_port_1, cls._debug_server_url_1, _, cls._server_thread_1,
     cls._server_1) = grpc_debug_test_server.start_server_on_separate_thread(
         dump_to_filesystem=False)
    (cls._server_port_2, cls._debug_server_url_2, _, cls._server_thread_2,
     cls._server_2) = grpc_debug_test_server.start_server_on_separate_thread(
         dump_to_filesystem=False)

  @classmethod
  def tearDownClass(cls):
    cls._server_1.stop_server().wait()
    cls._server_thread_1.join()
    cls._server_2.stop_server().wait()
    cls._server_thread_2.join()

  def tearDown(self):
    ops.reset_default_graph()
    self._server_1.clear_data()
    self._server_2.clear_data()

  def testToggleEnableTwoDebugWatchesNoCrosstalkBetweenDebugNodes(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
      v = variables.Variable(50.0, name="v")
      delta = constant_op.constant(5.0, name="delta")
      inc_v = state_ops.assign_add(v, delta, name="inc_v")

      sess.run(v.initializer)

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity(gated_grpc=true)",
                     "DebugNumericSummary(gated_grpc=true)"],
          debug_urls=[self._debug_server_url_1])

      for i in xrange(4):
        self._server_1.clear_data()

        # N.B.: These requests will be fulfilled not in this debugged
        # Session.run() invocation, but in the next one.
        if i % 2 == 0:
          self._server_1.request_watch("delta", 0, "DebugIdentity")
          self._server_1.request_unwatch("delta", 0, "DebugNumericSummary")
        else:
          self._server_1.request_unwatch("delta", 0, "DebugIdentity")
          self._server_1.request_watch("delta", 0, "DebugNumericSummary")

        sess.run(inc_v, options=run_options, run_metadata=run_metadata)

        if i == 0:
          self.assertEqual(0, len(self._server_1.debug_tensor_values))
        else:
          self.assertEqual(1, len(self._server_1.debug_tensor_values))
          if i % 2 == 1:
            self.assertAllClose(
                [5.0],
                self._server_1.debug_tensor_values["delta:0:DebugIdentity"])
          else:
            self.assertAllClose(
                [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0,
                  0.0, 1.0, 0.0]],
                self._server_1.debug_tensor_values[
                    "delta:0:DebugNumericSummary"])

  def testToggleEnableTwoDebugWatchesNoCrosstalkBetweenServers(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
      v = variables.Variable(50.0, name="v")
      delta = constant_op.constant(5.0, name="delta")
      inc_v = state_ops.assign_add(v, delta, name="inc_v")

      sess.run(v.initializer)

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity(gated_grpc=true)"],
          debug_urls=[self._debug_server_url_1, self._debug_server_url_2])

      for i in xrange(4):
        self._server_1.clear_data()
        self._server_2.clear_data()

        # N.B.: These requests will be fulfilled not in this debugged
        # Session.run() invocation, but in the next one.
        if i % 2 == 0:
          self._server_1.request_watch("delta", 0, "DebugIdentity")
          self._server_2.request_watch("v", 0, "DebugIdentity")
        else:
          self._server_1.request_unwatch("delta", 0, "DebugIdentity")
          self._server_2.request_unwatch("v", 0, "DebugIdentity")

        sess.run(inc_v, options=run_options, run_metadata=run_metadata)

        if i % 2 == 0:
          self.assertEqual(0, len(self._server_1.debug_tensor_values))
          self.assertEqual(0, len(self._server_2.debug_tensor_values))
        else:
          self.assertEqual(1, len(self._server_1.debug_tensor_values))
          self.assertEqual(1, len(self._server_2.debug_tensor_values))
          self.assertAllClose(
              [5.0],
              self._server_1.debug_tensor_values["delta:0:DebugIdentity"])
          self.assertAllClose(
              [50 + 5.0 * i],
              self._server_2.debug_tensor_values["v:0:DebugIdentity"])

  def testToggleBreakpointWorks(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
      v = variables.Variable(50.0, name="v")
      delta = constant_op.constant(5.0, name="delta")
      inc_v = state_ops.assign_add(v, delta, name="inc_v")

      sess.run(v.initializer)

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity(gated_grpc=true)"],
          debug_urls=[self._debug_server_url_1])

      for i in xrange(4):
        self._server_1.clear_data()

        # N.B.: These requests will be fulfilled not in this debugged
        # Session.run() invocation, but in the next one.
        if i in (0, 2):
          # Enable breakpoint at delta:0:DebugIdentity in runs 0 and 2.
          self._server_1.request_watch(
              "delta", 0, "DebugIdentity", breakpoint=True)
        else:
          # Disable the breakpoint in runs 1 and 3.
          self._server_1.request_unwatch("delta", 0, "DebugIdentity")

        output = sess.run(inc_v, options=run_options, run_metadata=run_metadata)
        self.assertAllClose(50.0 + 5.0 * (i + 1), output)

        if i in (0, 2):
          # After the end of runs 0 and 2, the server has received the requests
          # to enable the breakpoint at delta:0:DebugIdentity. So the server
          # should keep track of the correct breakpoints.
          self.assertSetEqual({("delta", 0, "DebugIdentity")},
                              self._server_1.breakpoints)
        else:
          # During runs 1 and 3, the server should have received the published
          # debug tensor delta:0:DebugIdentity. The breakpoint should have been
          # unblocked by EventReply reponses from the server.
          self.assertAllClose(
              [5.0],
              self._server_1.debug_tensor_values["delta:0:DebugIdentity"])
          # After the runs, the server should have properly removed the
          # breakpoints due to the request_unwatch calls.
          self.assertSetEqual(set(), self._server_1.breakpoints)

  def testGetGrpcDebugWatchesReturnsCorrectAnswer(self):
    with session.Session() as sess:
      v = variables.Variable(50.0, name="v")
      delta = constant_op.constant(5.0, name="delta")
      inc_v = state_ops.assign_add(v, delta, name="inc_v")

      sess.run(v.initializer)

      # Before any debugged runs, the server should be aware of no debug
      # watches.
      self.assertEqual([], self._server_1.gated_grpc_debug_watches())

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.add_debug_tensor_watch(
          run_options, "delta", output_slot=0,
          debug_ops=["DebugNumericSummary(gated_grpc=true)"],
          debug_urls=[self._debug_server_url_1])
      debug_utils.add_debug_tensor_watch(
          run_options, "v", output_slot=0,
          debug_ops=["DebugIdentity"],
          debug_urls=[self._debug_server_url_1])
      sess.run(inc_v, options=run_options, run_metadata=run_metadata)

      # After the first run, the server should have noted the debug watches
      # for which gated_grpc == True, but not the ones with gated_grpc == False.
      self.assertEqual(1, len(self._server_1.gated_grpc_debug_watches()))
      debug_watch = self._server_1.gated_grpc_debug_watches()[0]
      self.assertEqual("delta", debug_watch.node_name)
      self.assertEqual(0, debug_watch.output_slot)
      self.assertEqual("DebugNumericSummary", debug_watch.debug_op)


class DelayedDebugServerTest(test_util.TensorFlowTestCase):

  def testDebuggedSessionRunWorksWithDelayedDebugServerStartup(self):
    """Test debugged Session.run() tolerates delayed debug server startup."""
    ops.reset_default_graph()

    # Start a debug server asynchronously, with a certain amount of delay.
    (debug_server_port, _, _, server_thread,
     debug_server) = grpc_debug_test_server.start_server_on_separate_thread(
         server_start_delay_sec=2.0, dump_to_filesystem=False)

    with self.test_session() as sess:
      a_init = constant_op.constant(42.0, name="a_init")
      a = variables.Variable(a_init, name="a")

      def watch_fn(fetches, feeds):
        del fetches, feeds
        return framework.WatchOptions(debug_ops=["DebugIdentity"])

      sess = grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % debug_server_port, watch_fn=watch_fn)
      sess.run(a.initializer)
      self.assertAllClose(
          [42.0], debug_server.debug_tensor_values["a_init:0:DebugIdentity"])

    debug_server.stop_server().wait()
    server_thread.join()


if __name__ == "__main__":
  googletest.main()
