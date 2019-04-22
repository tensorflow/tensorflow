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

"""Base class for RpcOp tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.contrib.rpc.python.kernel_tests import test_example_pb2
from tensorflow.contrib.rpc.python.ops import rpc_op
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import proto_ops

__all__ = ['I_WARNED_YOU', 'RpcOpTestBase']

I_WARNED_YOU = 'I warned you!'


class RpcOpTestBase(object):
  # pylint: disable=missing-docstring,invalid-name
  """Base class for RpcOp tests."""

  def get_method_name(self, suffix):
    raise NotImplementedError

  def rpc(self, *args, **kwargs):
    return rpc_op.rpc(*args, protocol=self._protocol, **kwargs)

  def try_rpc(self, *args, **kwargs):
    return rpc_op.try_rpc(*args, protocol=self._protocol, **kwargs)

  def testScalarHostPortRpc(self):
    with self.cached_session() as sess:
      request_tensors = (
          test_example_pb2.TestCase(values=[1, 2, 3]).SerializeToString())
      response_tensors = self.rpc(
          method=self.get_method_name('Increment'),
          address=self._address,
          request=request_tensors)
      self.assertEqual(response_tensors.shape, ())
      response_values = sess.run(response_tensors)
    response_message = test_example_pb2.TestCase()
    self.assertTrue(response_message.ParseFromString(response_values))
    self.assertAllEqual([2, 3, 4], response_message.values)

  def testScalarHostPortTryRpc(self):
    with self.cached_session() as sess:
      request_tensors = (
          test_example_pb2.TestCase(values=[1, 2, 3]).SerializeToString())
      response_tensors, status_code, status_message = self.try_rpc(
          method=self.get_method_name('Increment'),
          address=self._address,
          request=request_tensors)
      self.assertEqual(status_code.shape, ())
      self.assertEqual(status_message.shape, ())
      self.assertEqual(response_tensors.shape, ())
      response_values, status_code_values, status_message_values = (
          sess.run((response_tensors, status_code, status_message)))
    response_message = test_example_pb2.TestCase()
    self.assertTrue(response_message.ParseFromString(response_values))
    self.assertAllEqual([2, 3, 4], response_message.values)
    # For the base Rpc op, don't expect to get error status back.
    self.assertEqual(errors.OK, status_code_values)
    self.assertEqual(b'', status_message_values)

  def testEmptyHostPortRpc(self):
    with self.cached_session() as sess:
      request_tensors = []
      response_tensors = self.rpc(
          method=self.get_method_name('Increment'),
          address=self._address,
          request=request_tensors)
      self.assertAllEqual(response_tensors.shape, [0])
      response_values = sess.run(response_tensors)
    self.assertAllEqual(response_values.shape, [0])

  def testInvalidMethod(self):
    for method in [
        '/InvalidService.Increment',
        self.get_method_name('InvalidMethodName')
    ]:
      with self.cached_session() as sess:
        with self.assertRaisesOpError(self.invalid_method_string):
          sess.run(self.rpc(method=method, address=self._address, request=''))

        _, status_code_value, status_message_value = sess.run(
            self.try_rpc(method=method, address=self._address, request=''))
        self.assertEqual(errors.UNIMPLEMENTED, status_code_value)
        self.assertTrue(
            self.invalid_method_string in status_message_value.decode('ascii'))

  def testInvalidAddress(self):
    # This covers the case of address='' and address='localhost:293874293874'
    address = 'unix:/tmp/this_unix_socket_doesnt_exist_97820348!!@'
    with self.cached_session() as sess:
      with self.assertRaises(errors.UnavailableError):
        sess.run(
            self.rpc(
                method=self.get_method_name('Increment'),
                address=address,
                request=''))
      _, status_code_value, status_message_value = sess.run(
          self.try_rpc(
              method=self.get_method_name('Increment'),
              address=address,
              request=''))
      self.assertEqual(errors.UNAVAILABLE, status_code_value)
      self.assertTrue(
          self.connect_failed_string in status_message_value.decode('ascii'))

  def testAlwaysFailingMethod(self):
    with self.cached_session() as sess:
      response_tensors = self.rpc(
          method=self.get_method_name('AlwaysFailWithInvalidArgument'),
          address=self._address,
          request='')
      self.assertEqual(response_tensors.shape, ())
      with self.assertRaisesOpError(I_WARNED_YOU):
        sess.run(response_tensors)

      response_tensors, status_code, status_message = self.try_rpc(
          method=self.get_method_name('AlwaysFailWithInvalidArgument'),
          address=self._address,
          request='')
      self.assertEqual(response_tensors.shape, ())
      self.assertEqual(status_code.shape, ())
      self.assertEqual(status_message.shape, ())
      status_code_value, status_message_value = sess.run((status_code,
                                                          status_message))
      self.assertEqual(errors.INVALID_ARGUMENT, status_code_value)
      self.assertTrue(I_WARNED_YOU in status_message_value.decode('ascii'))

  def testSometimesFailingMethodWithManyRequests(self):
    with self.cached_session() as sess:
      # Fail hard by default.
      response_tensors = self.rpc(
          method=self.get_method_name('SometimesFailWithInvalidArgument'),
          address=self._address,
          request=[''] * 20)
      self.assertEqual(response_tensors.shape, (20,))
      with self.assertRaisesOpError(I_WARNED_YOU):
        sess.run(response_tensors)

      # Don't fail hard, use TryRpc - return the failing status instead.
      response_tensors, status_code, status_message = self.try_rpc(
          method=self.get_method_name('SometimesFailWithInvalidArgument'),
          address=self._address,
          request=[''] * 20)
      self.assertEqual(response_tensors.shape, (20,))
      self.assertEqual(status_code.shape, (20,))
      self.assertEqual(status_message.shape, (20,))
      status_code_values, status_message_values = sess.run((status_code,
                                                            status_message))
      self.assertTrue([
          x in (errors.OK, errors.INVALID_ARGUMENT) for x in status_code_values
      ])
      expected_message_values = np.where(
          status_code_values == errors.INVALID_ARGUMENT,
          I_WARNED_YOU.encode('ascii'), b'')
      for msg, expected in zip(status_message_values, expected_message_values):
        self.assertTrue(expected in msg,
                        '"%s" did not contain "%s"' % (msg, expected))

  def testVecHostPortRpc(self):
    with self.cached_session() as sess:
      request_tensors = [
          test_example_pb2.TestCase(
              values=[i, i + 1, i + 2]).SerializeToString() for i in range(20)
      ]
      response_tensors = self.rpc(
          method=self.get_method_name('Increment'),
          address=self._address,
          request=request_tensors)
      self.assertEqual(response_tensors.shape, (20,))
      response_values = sess.run(response_tensors)
    self.assertEqual(response_values.shape, (20,))
    for i in range(20):
      response_message = test_example_pb2.TestCase()
      self.assertTrue(response_message.ParseFromString(response_values[i]))
      self.assertAllEqual([i + 1, i + 2, i + 3], response_message.values)

  def testVecHostPortManyParallelRpcs(self):
    with self.cached_session() as sess:
      request_tensors = [
          test_example_pb2.TestCase(
              values=[i, i + 1, i + 2]).SerializeToString() for i in range(20)
      ]
      many_response_tensors = [
          self.rpc(
              method=self.get_method_name('Increment'),
              address=self._address,
              request=request_tensors) for _ in range(10)
      ]
      # Launch parallel 10 calls to the RpcOp, each containing 20 rpc requests.
      many_response_values = sess.run(many_response_tensors)
    self.assertEqual(10, len(many_response_values))
    for response_values in many_response_values:
      self.assertEqual(response_values.shape, (20,))
      for i in range(20):
        response_message = test_example_pb2.TestCase()
        self.assertTrue(response_message.ParseFromString(response_values[i]))
        self.assertAllEqual([i + 1, i + 2, i + 3], response_message.values)

  def testVecHostPortRpcUsingEncodeAndDecodeProto(self):
    with self.cached_session() as sess:
      request_tensors = proto_ops.encode_proto(
          message_type='tensorflow.contrib.rpc.TestCase',
          field_names=['values'],
          sizes=[[3]] * 20,
          values=[
              [[i, i + 1, i + 2] for i in range(20)],
          ])
      response_tensor_strings = self.rpc(
          method=self.get_method_name('Increment'),
          address=self._address,
          request=request_tensors)
      _, (response_shape,) = proto_ops.decode_proto(
          bytes=response_tensor_strings,
          message_type='tensorflow.contrib.rpc.TestCase',
          field_names=['values'],
          output_types=[dtypes.int32])
      response_shape_values = sess.run(response_shape)
    self.assertAllEqual([[i + 1, i + 2, i + 3]
                         for i in range(20)], response_shape_values)

  def testVecHostPortRpcCancelsUponSessionTimeOutWhenSleepingForever(self):
    with self.cached_session() as sess:
      request_tensors = [''] * 25  # This will launch 25 RPC requests.
      response_tensors = self.rpc(
          method=self.get_method_name('SleepForever'),
          address=self._address,
          request=request_tensors)
      for timeout_ms in [1, 500, 1000]:
        options = config_pb2.RunOptions(timeout_in_ms=timeout_ms)
        with self.assertRaises((errors.UnavailableError,
                                errors.DeadlineExceededError)):
          sess.run(response_tensors, options=options)

  def testVecHostPortRpcCancelsUponConfiguredTimeOutWhenSleepingForever(self):
    with self.cached_session() as sess:
      request_tensors = [''] * 25  # This will launch 25 RPC requests.
      response_tensors = self.rpc(
          method=self.get_method_name('SleepForever'),
          address=self._address,
          timeout_in_ms=1000,
          request=request_tensors)
      with self.assertRaises(errors.DeadlineExceededError):
        sess.run(response_tensors)

  def testTryRpcPropagatesDeadlineErrorWithSometimesTimingOutRequests(self):
    with self.cached_session() as sess:
      response_tensors, status_code, status_message = self.try_rpc(
          method=self.get_method_name('SometimesSleepForever'),
          timeout_in_ms=1000,
          address=self._address,
          request=[''] * 20)
      self.assertEqual(response_tensors.shape, (20,))
      self.assertEqual(status_code.shape, (20,))
      self.assertEqual(status_message.shape, (20,))
      status_code_values = sess.run(status_code)
      self.assertTrue([
          x in (errors.OK, errors.DEADLINE_EXCEEDED) for x in status_code_values
      ])

  def testTryRpcWithMultipleAddressesSingleRequest(self):
    flatten = lambda x: list(itertools.chain.from_iterable(x))
    with self.cached_session() as sess:
      addresses = flatten([[
          self._address, 'unix:/tmp/this_unix_socket_doesnt_exist_97820348!!@'
      ] for _ in range(10)])
      request = test_example_pb2.TestCase(values=[0, 1, 2]).SerializeToString()
      response_tensors, status_code, _ = self.try_rpc(
          method=self.get_method_name('Increment'),
          address=addresses,
          request=request)
      response_tensors_values, status_code_values = sess.run((response_tensors,
                                                              status_code))
      self.assertAllEqual(
          flatten([errors.OK, errors.UNAVAILABLE] for _ in range(10)),
          status_code_values)
      for i in range(10):
        self.assertTrue(response_tensors_values[2 * i])
        self.assertFalse(response_tensors_values[2 * i + 1])

  def testTryRpcWithMultipleMethodsSingleRequest(self):
    flatten = lambda x: list(itertools.chain.from_iterable(x))
    with self.cached_session() as sess:
      methods = flatten(
          [[self.get_method_name('Increment'), 'InvalidMethodName']
           for _ in range(10)])
      request = test_example_pb2.TestCase(values=[0, 1, 2]).SerializeToString()
      response_tensors, status_code, _ = self.try_rpc(
          method=methods, address=self._address, request=request)
      response_tensors_values, status_code_values = sess.run((response_tensors,
                                                              status_code))
      self.assertAllEqual(
          flatten([errors.OK, errors.UNIMPLEMENTED] for _ in range(10)),
          status_code_values)
      for i in range(10):
        self.assertTrue(response_tensors_values[2 * i])
        self.assertFalse(response_tensors_values[2 * i + 1])

  def testTryRpcWithMultipleAddressesAndRequests(self):
    flatten = lambda x: list(itertools.chain.from_iterable(x))
    with self.cached_session() as sess:
      addresses = flatten([[
          self._address, 'unix:/tmp/this_unix_socket_doesnt_exist_97820348!!@'
      ] for _ in range(10)])
      requests = [
          test_example_pb2.TestCase(
              values=[i, i + 1, i + 2]).SerializeToString() for i in range(20)
      ]
      response_tensors, status_code, _ = self.try_rpc(
          method=self.get_method_name('Increment'),
          address=addresses,
          request=requests)
      response_tensors_values, status_code_values = sess.run((response_tensors,
                                                              status_code))
      self.assertAllEqual(
          flatten([errors.OK, errors.UNAVAILABLE] for _ in range(10)),
          status_code_values)
      for i in range(20):
        if i % 2 == 1:
          self.assertFalse(response_tensors_values[i])
        else:
          response_message = test_example_pb2.TestCase()
          self.assertTrue(
              response_message.ParseFromString(response_tensors_values[i]))
          self.assertAllEqual([i + 1, i + 2, i + 3], response_message.values)
