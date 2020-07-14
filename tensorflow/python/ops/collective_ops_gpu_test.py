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
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class CollectiveOpGPUTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Set group_size = num_gpus = 2 for all tests in this class."""
    super(CollectiveOpGPUTest, cls).setUpClass()
    # Group size is the number of devices in a group communicating collectively.
    # This will be passed into the collective ops in the tests below.
    cls._group_size = 2
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'

  def _configure(self, set_config_proto_nccl=True):
    """Return `ConfigProto` for NCCL execution."""
    experimental = config_pb2.ConfigProto.Experimental()
    if set_config_proto_nccl:
      experimental.collective_nccl = True
    return config_pb2.ConfigProto(experimental=experimental)

  def _ensure_context_initialized(self):
    gpus = config.list_physical_devices('GPU')
    if len(gpus) < 2:
      self.skipTest('Expected at least 2 GPUs but found {} GPUs'.format(
          len(gpus)))
    context.ensure_initialized()

  def testBasicNcclAllReduce(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(self._group_size)]

    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default(), self.session(
        config=self._configure()) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(self._group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div'))
      results = sess.run(collectives)
    for result in results:
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testInt32Error(self):
    inputs = [[0, 1], [2, 3]]
    group_key = 1
    instance_key = 50
    devices = ['/GPU:{}'.format(i) for i in range(self._group_size)]

    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default(), self.session(
        config=self._configure()) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(self._group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i], dtype=dtypes.int32)
          collectives.append(collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div'))
      with self.assertRaisesRegex(
          errors.InternalError,
          'does not support datatype DT_INT32 on DEVICE_GPU'):
        sess.run(collectives)

  def testFp16Reduce(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 100
    devices = ['/GPU:{}'.format(i) for i in range(self._group_size)]

    with ops.Graph().as_default(), self.session(
        config=self._configure()) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(self._group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i], dtype=dtypes.float16)
          collectives.append(collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div'))
      results = sess.run(collectives)
    for result in results:
      logging.info('i {} result {} expected {}'.format(i, results[i], expected))
      self.assertAllClose(result, expected, rtol=1e-3, atol=1e-3)

  def testNcclHintAllReduce(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(self._group_size)]

    with ops.Graph().as_default(), self.session(
        config=self._configure(set_config_proto_nccl=False)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(self._group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div',
              communication_hint='nccl'))
      results = sess.run(collectives)
    for result in results:
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testBasicNcclBroadcast(self):
    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(self._group_size)]

    with ops.Graph().as_default(), self.session(
        config=self._configure()) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      with ops.device(devices[0]):
        t = constant_op.constant(tensor_value)
        collectives.append(collective_ops.broadcast_send(
            t, t.shape, t.dtype, self._group_size, group_key, instance_key))
      with ops.device(devices[1]):
        t = constant_op.constant(tensor_value)
        collectives.append(collective_ops.broadcast_recv(
            t.shape, t.dtype, self._group_size, group_key, instance_key))
      results = sess.run(collectives)
    for result in results:
      self.assertAllClose(result, tensor_value, rtol=1e-5, atol=1e-5)

  def testNcclBroadcastDoubleRecv(self):
    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(self._group_size)]

    with ops.Graph().as_default(), self.session(
        config=self._configure()) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for device in devices:
        with ops.device(device):
          t = constant_op.constant(tensor_value)
          collectives.append(collective_ops.broadcast_recv(
              t.shape, t.dtype, self._group_size, group_key, instance_key))
      with self.assertRaisesRegex(errors.InternalError, 'found no source'):
        sess.run(collectives)

  def testNcclBroadcastDoubleSend(self):
    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(self._group_size)]

    with ops.Graph().as_default(), self.session(
        config=self._configure()) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for device in devices:
        with ops.device(device):
          t = constant_op.constant(tensor_value)
          collectives.append(collective_ops.broadcast_send(
              t, t.shape, t.dtype, self._group_size, group_key, instance_key))
      with self.assertRaisesRegex(errors.InternalError, 'already has source'):
        sess.run(collectives)

  def testBasicNcclAllGather(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1,
                0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]
    group_key = 1
    instance_key = 1
    devices = ['/GPU:{}'.format(i) for i in range(self._group_size)]

    with ops.Graph().as_default(), self.session(
        config=self._configure()) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      collectives = []
      for i in range(self._group_size):
        with ops.device(devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_gather(t, self._group_size,
                                                       group_key, instance_key))
      results = sess.run(collectives)
    for result in results:
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testCollectiveDeviceMismatch(self):
    group_key = 10
    instance_key = 20
    t0 = [1, 2, 3, 4]
    t1 = [5, 6, 7, 8]

    with ops.Graph().as_default(), self.session(
        config=self._configure(set_config_proto_nccl=False)) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      with ops.device('/CPU:0'):
        in0 = constant_op.constant(t0)
        c0 = collective_ops.all_reduce(in0, self._group_size, group_key,
                                       instance_key, 'Add', 'Id')
      with ops.device('/GPU:0'):
        in1 = constant_op.constant(t1)
        c1 = collective_ops.all_reduce(in1, self._group_size, group_key,
                                       instance_key, 'Add', 'Id')
      run_options = config_pb2.RunOptions()
      run_options.experimental.collective_graph_key = 100
      with self.assertRaisesRegex(errors.InternalError,
                                  'but that group has type'):
        sess.run([c0, c1], options=run_options)

  @test_util.run_v2_only
  def testCollectiveReduceMinMax(self):
    self._ensure_context_initialized()

    @def_function.function
    def run_all_reduce(group_key, instance_key, merge_op):
      t0 = [1., 20., 3., 40., 5.]
      t1 = [10., 2., 30., 4., 50.]
      os.environ['NCCL_DEBUG'] = 'INFO'
      os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'
      with ops.device('/GPU:0'):
        in0 = constant_op.constant(t0)
        c0 = collective_ops.all_reduce(
            in0, self._group_size, group_key, instance_key, merge_op,
            final_op='Id', communication_hint='nccl')
      with ops.device('/GPU:1'):
        in1 = constant_op.constant(t1)
        c1 = collective_ops.all_reduce(
            in1, self._group_size, group_key, instance_key, merge_op,
            final_op='Id', communication_hint='nccl')
      return c0, c1

    for combination in [('Max', [10., 20., 30., 40., 50.]),
                        ('Min', [1., 2., 3., 4., 5.])]:
      merge_op = combination[0]
      results = run_all_reduce(group_key=10, instance_key=20, merge_op=merge_op)
      expected = combination[1]
      for result in results:
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  @test_util.run_v2_only
  def testCollectiveGroupSizeOne(self):
    self._ensure_context_initialized()

    group_size = 1
    group_key = 100
    instance_key = 100
    in_value = [1., 2., 3., 4.]
    in_tensor = constant_op.constant(in_value)

    with ops.device('/GPU:0'):
      reduced_tensor = collective_ops.all_reduce(
          in_tensor, group_size, group_key, instance_key, 'Add', 'Id',
          communication_hint='nccl')
    self.assertAllEqual(in_value, reduced_tensor.numpy())

    with ops.device('/GPU:0'):
      gathered_tensor = collective_ops.all_gather(
          in_tensor, group_size, group_key, instance_key)
    self.assertAllEqual(in_value, gathered_tensor.numpy())


if __name__ == '__main__':
  test.main()
