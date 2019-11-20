# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for source_remote."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback

import grpc

from tensorflow.core.debug import debug_service_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import grpc_debug_test_server
from tensorflow.python.debug.lib import source_remote
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def line_number_above():
  return tf_inspect.stack()[1][2] - 1


class SendTracebacksTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    test_util.TensorFlowTestCase.setUpClass()
    (cls._server_port, cls._debug_server_url, cls._server_dump_dir,
     cls._server_thread,
     cls._server) = grpc_debug_test_server.start_server_on_separate_thread(
         poll_server=True)
    cls._server_address = "localhost:%d" % cls._server_port
    (cls._server_port_2, cls._debug_server_url_2, cls._server_dump_dir_2,
     cls._server_thread_2,
     cls._server_2) = grpc_debug_test_server.start_server_on_separate_thread()
    cls._server_address_2 = "localhost:%d" % cls._server_port_2
    cls._curr_file_path = os.path.normpath(os.path.abspath(__file__))

  @classmethod
  def tearDownClass(cls):
    # Stop the test server and join the thread.
    cls._server.stop_server().wait()
    cls._server_thread.join()
    cls._server_2.stop_server().wait()
    cls._server_thread_2.join()
    test_util.TensorFlowTestCase.tearDownClass()

  def tearDown(self):
    ops.reset_default_graph()
    self._server.clear_data()
    self._server_2.clear_data()
    super(SendTracebacksTest, self).tearDown()

  def _findFirstTraceInsideTensorFlowPyLibrary(self, op):
    """Find the first trace of an op that belongs to the TF Python library."""
    for trace in op.traceback:
      if source_utils.guess_is_tensorflow_py_library(trace.filename):
        return trace

  def testSendGraphTracebacksToSingleDebugServer(self):
    this_func_name = "testSendGraphTracebacksToSingleDebugServer"
    with session.Session() as sess:
      a = variables.Variable(21.0, name="a")
      a_lineno = line_number_above()
      b = variables.Variable(2.0, name="b")
      b_lineno = line_number_above()
      math_ops.add(a, b, name="x")
      x_lineno = line_number_above()

      send_stack = traceback.extract_stack()
      send_lineno = line_number_above()
      source_remote.send_graph_tracebacks(
          self._server_address, "dummy_run_key", send_stack, sess.graph)

      tb = self._server.query_op_traceback("a")
      self.assertIn((self._curr_file_path, a_lineno, this_func_name), tb)
      tb = self._server.query_op_traceback("b")
      self.assertIn((self._curr_file_path, b_lineno, this_func_name), tb)
      tb = self._server.query_op_traceback("x")
      self.assertIn((self._curr_file_path, x_lineno, this_func_name), tb)

      self.assertIn(
          (self._curr_file_path, send_lineno, this_func_name),
          self._server.query_origin_stack()[-1])

      self.assertEqual(
          "      a = variables.Variable(21.0, name=\"a\")",
          self._server.query_source_file_line(__file__, a_lineno))
      # Files in the TensorFlow code base shouldn not have been sent.
      tf_trace = self._findFirstTraceInsideTensorFlowPyLibrary(a.op)
      tf_trace_file_path = tf_trace.filename
      with self.assertRaises(ValueError):
        self._server.query_source_file_line(tf_trace_file_path, 0)
      self.assertEqual([debug_service_pb2.CallTraceback.GRAPH_EXECUTION],
                       self._server.query_call_types())
      self.assertEqual(["dummy_run_key"], self._server.query_call_keys())
      self.assertEqual(
          [sess.graph.version], self._server.query_graph_versions())

  def testSendGraphTracebacksToTwoDebugServers(self):
    this_func_name = "testSendGraphTracebacksToTwoDebugServers"
    with session.Session() as sess:
      a = variables.Variable(21.0, name="two/a")
      a_lineno = line_number_above()
      b = variables.Variable(2.0, name="two/b")
      b_lineno = line_number_above()
      x = math_ops.add(a, b, name="two/x")
      x_lineno = line_number_above()

      send_traceback = traceback.extract_stack()
      send_lineno = line_number_above()

      with test.mock.patch.object(
          grpc, "insecure_channel",
          wraps=grpc.insecure_channel) as mock_grpc_channel:
        source_remote.send_graph_tracebacks(
            [self._server_address, self._server_address_2],
            "dummy_run_key", send_traceback, sess.graph)
        mock_grpc_channel.assert_called_with(
            test.mock.ANY,
            options=[("grpc.max_receive_message_length", -1),
                     ("grpc.max_send_message_length", -1)])

      servers = [self._server, self._server_2]
      for server in servers:
        tb = server.query_op_traceback("two/a")
        self.assertIn((self._curr_file_path, a_lineno, this_func_name), tb)
        tb = server.query_op_traceback("two/b")
        self.assertIn((self._curr_file_path, b_lineno, this_func_name), tb)
        tb = server.query_op_traceback("two/x")
        self.assertIn((self._curr_file_path, x_lineno, this_func_name), tb)

        self.assertIn(
            (self._curr_file_path, send_lineno, this_func_name),
            server.query_origin_stack()[-1])

        self.assertEqual(
            "      x = math_ops.add(a, b, name=\"two/x\")",
            server.query_source_file_line(__file__, x_lineno))
        tf_trace = self._findFirstTraceInsideTensorFlowPyLibrary(a.op)
        tf_trace_file_path = tf_trace.filename
        with self.assertRaises(ValueError):
          server.query_source_file_line(tf_trace_file_path, 0)
        self.assertEqual([debug_service_pb2.CallTraceback.GRAPH_EXECUTION],
                         server.query_call_types())
        self.assertEqual(["dummy_run_key"], server.query_call_keys())
        self.assertEqual([sess.graph.version], server.query_graph_versions())

  def testSendEagerTracebacksToSingleDebugServer(self):
    this_func_name = "testSendEagerTracebacksToSingleDebugServer"
    send_traceback = traceback.extract_stack()
    send_lineno = line_number_above()
    source_remote.send_eager_tracebacks(self._server_address, send_traceback)

    self.assertEqual([debug_service_pb2.CallTraceback.EAGER_EXECUTION],
                     self._server.query_call_types())
    self.assertIn((self._curr_file_path, send_lineno, this_func_name),
                  self._server.query_origin_stack()[-1])

  def testGRPCServerMessageSizeLimit(self):
    """Assert gRPC debug server is started with unlimited message size."""
    with test.mock.patch.object(
        grpc, "server", wraps=grpc.server) as mock_grpc_server:
      (_, _, _, server_thread,
       server) = grpc_debug_test_server.start_server_on_separate_thread(
           poll_server=True)
      mock_grpc_server.assert_called_with(
          test.mock.ANY,
          options=[("grpc.max_receive_message_length", -1),
                   ("grpc.max_send_message_length", -1)])
    server.stop_server().wait()
    server_thread.join()


if __name__ == "__main__":
  googletest.main()
