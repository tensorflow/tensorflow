# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Collective Operations that require GPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class CollectiveOpGPUTest(test.TestCase):

  def _configure(self, group_size, set_config_proto_nccl=True):
    """Set environment variables and return `ConfigProto` for NCCL execution."""
    # Configure virtual GPU devices
    virtual_devices = [config_pb2.GPUOptions.Experimental.VirtualDevices(
        memory_limit_mb=([1 << 10] * group_size))]  # 1 GB per virtual GPU
    gpu_options = config_pb2.GPUOptions(
        visible_device_list='0',
        experimental=config_pb2.GPUOptions.Experimental(
            virtual_devices=virtual_devices))
    # Configure NCCL
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'
    experimental = config_pb2.ConfigProto.Experimental()
    if set_config_proto_nccl:
      experimental.collective_nccl = True
    return config_pb2.ConfigProto(gpu_options=gpu_options,
                                  experimental=experimental)

  @test_util.run_deprecated_v1
  def testBasicNcclAllReduce(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_size = len(inputs)
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(group_size)]

    with self.session(config=self._configure(group_size)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_reduce(
              t, group_size, group_key, instance_key, 'Add', 'Div'))
      results = sess.run(collectives)
    for result in results:
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testInt32Error(self):
    inputs = [[0, 1], [2, 3]]
    group_size = len(inputs)
    group_key = 1
    instance_key = 50
    devices = ['/GPU:{}'.format(i) for i in range(group_size)]

    with self.session(config=self._configure(group_size)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i], dtype=dtypes.int32)
          collectives.append(collective_ops.all_reduce(
              t, group_size, group_key, instance_key, 'Add', 'Div'))
      with self.assertRaisesRegexp(
          errors.InternalError,
          'does not support datatype DT_INT32 on DEVICE_GPU'):
        sess.run(collectives)

  @test_util.run_deprecated_v1
  def testFp16Reduce(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_size = len(inputs)
    group_key = 1
    instance_key = 100
    devices = ['/GPU:{}'.format(i) for i in range(group_size)]

    with self.session(config=self._configure(group_size)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i], dtype=dtypes.float16)
          collectives.append(collective_ops.all_reduce(
              t, group_size, group_key, instance_key, 'Add', 'Div'))
      results = sess.run(collectives)
    for result in results:
      logging.info('i {} result {} expected {}'.format(i, results[i], expected))
      self.assertAllClose(result, expected, rtol=1e-3, atol=1e-3)

  @test_util.run_deprecated_v1
  def testNcclHintAllReduce(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_size = len(inputs)
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(group_size)]

    with self.session(
        config=self._configure(group_size,
                               set_config_proto_nccl=False)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_reduce(
              t, group_size, group_key, instance_key, 'Add', 'Div',
              communication_hint='nccl'))
      results = sess.run(collectives)
    for result in results:
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testBasicNcclBroadcast(self):
    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_size = 2
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(group_size)]

    with self.session(config=self._configure(group_size)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      with ops.device(devices[0]):
        t = constant_op.constant(tensor_value)
        collectives.append(collective_ops.broadcast_send(
            t, t.shape, t.dtype, group_size, group_key, instance_key))
      with ops.device(devices[1]):
        t = constant_op.constant(tensor_value)
        collectives.append(collective_ops.broadcast_recv(
            t.shape, t.dtype, group_size, group_key, instance_key))
      results = sess.run(collectives)
    for result in results:
      self.assertAllClose(result, tensor_value, rtol=1e-5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testNcclBroadcastDoubleRecv(self):
    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_size = 2
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(group_size)]

    with self.session(config=self._configure(group_size)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for device in devices:
        with ops.device(device):
          t = constant_op.constant(tensor_value)
          collectives.append(collective_ops.broadcast_recv(
              t.shape, t.dtype, group_size, group_key, instance_key))
      with self.assertRaisesRegexp(errors.InternalError, 'found no source'):
        sess.run(collectives)

  @test_util.run_deprecated_v1
  def testNcclBroadcastDoubleSend(self):
    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_size = 2
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(group_size)]

    with self.session(config=self._configure(group_size)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for device in devices:
        with ops.device(device):
          t = constant_op.constant(tensor_value)
          collectives.append(collective_ops.broadcast_send(
              t, t.shape, t.dtype, group_size, group_key, instance_key))
      with self.assertRaisesRegexp(errors.InternalError, 'already has source'):
        sess.run(collectives)

  @test_util.run_deprecated_v1
  def testBasicNcclAllGather(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1,
                0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]
    group_size = len(inputs)
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(group_size)]

    with self.session(config=self._configure(group_size)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_gather(t, group_size,
                                                       group_key, instance_key))
      results = sess.run(collectives)
    for result in results:
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  test.main()
