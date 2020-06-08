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
"""Tests for debugger functionalities in tf.compat.v1.Session with grpc:// URLs.

This test focus on grpc:// debugging of distributed (gRPC) sessions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import subprocess
import sys
import time

import portpicker
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_test_server
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


@test_util.run_v1_only("b/120545219")
class DistributedSessionDebugTest(test_util.TensorFlowTestCase):
  """Test the debugging of distributed sessions."""

  PER_PROC_GPU_MEMORY_FRACTION = 0.1
  POLLING_INTERVAL_SEC = 0.025

  @classmethod
  def setUpClass(cls):
    gpu_memory_fraction_opt = (
        "--gpu_memory_fraction=%f" % cls.PER_PROC_GPU_MEMORY_FRACTION)

    worker_port = portpicker.pick_unused_port()
    cluster_spec = "worker|localhost:%d" % worker_port
    tf_logging.info("cluster_spec: %s", cluster_spec)

    server_bin = test.test_src_dir_path(
        "python/debug/grpc_tensorflow_server.par")

    cls.server_target = "grpc://localhost:%d" % worker_port

    cls.server_procs = {}
    cls.server_procs["worker"] = subprocess.Popen(
        [
            server_bin,
            "--logtostderr",
            "--cluster_spec=%s" % cluster_spec,
            "--job_name=worker",
            "--task_id=0",
            gpu_memory_fraction_opt,
        ],
        stdout=sys.stdout,
        stderr=sys.stderr)

    # Start debug server in-process, on separate thread.
    (cls.debug_server_port, cls.debug_server_url, _, cls.debug_server_thread,
     cls.debug_server
    ) = grpc_debug_test_server.start_server_on_separate_thread(
        dump_to_filesystem=False)
    tf_logging.info("debug server url: %s", cls.debug_server_url)

    cls.session_config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(
            per_process_gpu_memory_fraction=cls.PER_PROC_GPU_MEMORY_FRACTION))

  @classmethod
  def tearDownClass(cls):
    for key in cls.server_procs:
      cls.server_procs[key].terminate()
    try:
      cls.debug_server.stop_server().wait()
    except ValueError:
      pass
    cls.debug_server_thread.join()

  def setUp(self):
    pass

  def tearDown(self):
    self.debug_server.clear_data()

  def _pollingAssertDebugTensorValuesAllClose(self, expected_values,
                                              debug_tensor_name):
    """Poll debug_server till tensor appears and matches expected values."""
    while (debug_tensor_name not in self.debug_server.debug_tensor_values or
           len(self.debug_server.debug_tensor_values) < len(expected_values)):
      time.sleep(self.POLLING_INTERVAL_SEC)
    self.assertAllClose(
        expected_values,
        self.debug_server.debug_tensor_values[debug_tensor_name])

  def _createGraph(self):
    """Create graph for testing.

    Returns:
      Python Graph object.
    """
    with ops.Graph().as_default() as graph:
      with ops.device("/job:worker/task:0/cpu:0"):
        self.a = variables.VariableV1(10.0, name="a")
        self.b = variables.VariableV1(100.0, name="b")
        self.inc_a = state_ops.assign_add(self.a, 2.0, name="inc_a")
        self.dec_b = state_ops.assign_add(self.b, -5.0, name="dec_b")
        self.p = math_ops.multiply(self.inc_a, self.dec_b, name="p")
        self.q = math_ops.negative(self.p, name="q")
    return graph

  def testDistributedRunWithGatedGrpcCommunicatesWithDebugServerCorrectly(self):
    graph = self._createGraph()
    with session.Session(
        config=self.session_config, graph=graph,
        target=self.server_target) as sess:
      sess.run(self.a.initializer)
      sess.run(self.b.initializer)

      run_options = config_pb2.RunOptions()
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          node_name_regex_whitelist=r"a",
          debug_ops=["DebugIdentity"],
          debug_urls=[self.debug_server_url])

      # Test gated_grpc for an op located on the worker, i.e., on the same
      # host as where MasterSession is.
      # TODO(cais): gRPC gating of debug ops does not work on partition graphs
      # not located on MasterSession hosts (e.g., parameter servers) yet. Make
      # it work.
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          node_name_regex_whitelist=r"p",
          debug_ops=["DebugIdentity(gated_grpc=True)"],
          debug_urls=[self.debug_server_url])

      for i in xrange(4):
        if i % 2 == 0:
          self.debug_server.request_watch("p", 0, "DebugIdentity")
        else:
          self.debug_server.request_unwatch("p", 0, "DebugIdentity")

        expected_p = (10.0 + 2.0 * (i + 1)) * (100.0 - 5.0 * (i + 1))
        self.assertAllClose(-expected_p, sess.run(self.q, options=run_options))

        self.assertEqual(1, len(self.debug_server.core_metadata_json_strings))
        core_metadata = json.loads(
            self.debug_server.core_metadata_json_strings[0])
        self.assertEqual([], core_metadata["input_names"])
        self.assertEqual(["q:0"], core_metadata["output_names"])
        self.assertEqual(i, core_metadata["executor_step_index"])

        if i == 0:
          self.assertEqual(1, len(self.debug_server.partition_graph_defs))

        # Tensor "a" is from a PS. It may take longer to arrive due to the fact
        # that the stream connection between the PS and the debug server is
        # persistent and not torn down at the end of each Session.run()
        self._pollingAssertDebugTensorValuesAllClose([10.0 + 2.0 * i],
                                                     "a:0:DebugIdentity")

        # Due to the gRPC gating of the debug op for "p", the debug tensor
        # should be available on odd-indexed runs.
        if i % 2 == 0:
          self.assertAllClose(
              [expected_p],
              self.debug_server.debug_tensor_values["p:0:DebugIdentity"])
        else:
          self.assertNotIn("p:0:DebugIdentity",
                           self.debug_server.debug_tensor_values)

        self.assertNotIn("b:0:DebugIdentity",
                         self.debug_server.debug_tensor_values)
        self.debug_server.clear_data()

  def testDistributedRunWithGrpcDebugWrapperWorks(self):
    graph = self._createGraph()
    with session.Session(
        config=self.session_config, graph=graph,
        target=self.server_target) as sess:
      sess.run(self.a.initializer)
      sess.run(self.b.initializer)

      def watch_fn(feeds, fetch_keys):
        del feeds, fetch_keys
        return framework.WatchOptions(
            debug_ops=["DebugIdentity"],
            node_name_regex_whitelist=r"p")
      sess = grpc_wrapper.GrpcDebugWrapperSession(
          sess, "localhost:%d" % self.debug_server_port, watch_fn=watch_fn)

      for i in xrange(4):
        expected_p = (10.0 + 2.0 * (i + 1)) * (100.0 - 5.0 * (i + 1))
        self.assertAllClose(-expected_p, sess.run(self.q))

        if i == 0:
          self.assertEqual(1, len(self.debug_server.partition_graph_defs))

        self.assertAllClose(
            [expected_p],
            self.debug_server.debug_tensor_values["p:0:DebugIdentity"])
        self.assertNotIn("b:0:DebugIdentity",
                         self.debug_server.debug_tensor_values)
        self.debug_server.clear_data()


if __name__ == "__main__":
  googletest.main()
