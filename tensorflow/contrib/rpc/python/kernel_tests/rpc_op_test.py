# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for RpcOp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes as ct
import os

import grpc
from grpc.framework.foundation import logging_pool
import portpicker

from tensorflow.contrib.rpc.python.kernel_tests import rpc_op_test_base
from tensorflow.contrib.rpc.python.kernel_tests import rpc_op_test_servicer
from tensorflow.contrib.rpc.python.kernel_tests import test_example_pb2_grpc
from tensorflow.python.platform import test


class RpcOpTest(test.TestCase, rpc_op_test_base.RpcOpTestBase):
  _protocol = 'grpc'

  invalid_method_string = 'Method not found'
  connect_failed_string = 'Connect Failed'

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    super(RpcOpTest, self).__init__(methodName)
    lib = os.path.join(os.path.dirname(__file__), 'libtestexample.so')
    if os.path.isfile(lib):
      ct.cdll.LoadLibrary(lib)

  def get_method_name(self, suffix):
    return '/tensorflow.contrib.rpc.TestCaseService/%s' % suffix

  def setUp(self):
    super(RpcOpTest, self).setUp()

    service_port = portpicker.pick_unused_port()

    server = grpc.server(logging_pool.pool(max_workers=25))
    servicer = rpc_op_test_servicer.RpcOpTestServicer()
    test_example_pb2_grpc.add_TestCaseServiceServicer_to_server(
        servicer, server)
    self._address = 'localhost:%d' % service_port
    server.add_insecure_port(self._address)
    server.start()
    self._server = server

  def tearDown(self):
    self._server.stop(grace=None)
    super(RpcOpTest, self).tearDown()


if __name__ == '__main__':
  test.main()
