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

"""Test servicer for RpcOp tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

import grpc

from tensorflow.contrib.rpc.python.kernel_tests import rpc_op_test_base
from tensorflow.contrib.rpc.python.kernel_tests import test_example_pb2_grpc


class RpcOpTestServicer(test_example_pb2_grpc.TestCaseServiceServicer):
  """Test servicer for RpcOp tests."""

  def Increment(self, request, context):
    """Increment the entries in the `values` attribute of request.

    Args:
      request: input TestCase.
      context: the rpc context.

    Returns:
      output TestCase.
    """
    for i in range(len(request.values)):
      request.values[i] += 1
    return request

  def AlwaysFailWithInvalidArgument(self, request, context):
    """Always fails with an InvalidArgument status.

    Args:
      request: input TestCase.
      context: the rpc context.

    Returns:
      output TestCase.
    """
    del request
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(rpc_op_test_base.I_WARNED_YOU)

  def SometimesFailWithInvalidArgument(self, request, context):
    """Sometimes fails with an InvalidArgument status.

    Args:
      request: input TestCase.
      context: the rpc context.

    Returns:
      output TestCase.
    """
    if random.randint(0, 1) == 1:
      context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
      context.set_details(rpc_op_test_base.I_WARNED_YOU)
    return request

  def SleepForever(self, request, context):
    """Sleeps forever.

    Args:
      request: input TestCase.
      context: the rpc context.

    Returns:
      output TestCase.
    """
    # TODO(ebrevdo): Make this async wait like the stubby version.
    time.sleep(5)

  def SometimesSleepForever(self, request, context):
    """Sometimes sleeps forever.

    Args:
      request: input TestCase.
      context: the rpc context.

    Returns:
      output TestCase.
    """
    if random.randint(0, 1) == 1:
      time.sleep(5)
    return request
