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

import os
import threading
import time

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


class CollectiveOpGPUTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Set group_size = num_gpus = 2 for all tests in this class."""
    super(CollectiveOpGPUTest, cls).setUpClass()
    # Group size is the number of devices in a group communicating collectively.
    # This will be passed into the collective ops in the tests below.
    cls._group_size = 2
    cls._devices = ['/device:GPU:{}'.format(i) for i in range(2)]
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'

  def _setup_context(self, num_gpus=2):
    context._reset_context()
    gpus = config.list_physical_devices('GPU')
    if len(gpus) < num_gpus:
      self.skipTest('Expected at least {} GPUs but found {} GPUs'.format(
          num_gpus, len(gpus)))
    context.ensure_initialized()

  def testBasicNcclAllReduce(self):
    self._setup_context()

    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_basic_all_reduce():
      collectives = []
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div'))
      return collectives

    for result in run_basic_all_reduce():
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testInt32Error(self):
    self._setup_context()

    inputs = [[0, 1], [2, 3]]
    group_key = 1
    instance_key = 50

    @def_function.function
    def run_int32_error():
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i], dtype=dtypes.int32)
          collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div')

    with self.assertRaisesRegex(
        errors.InternalError,
        'does not support datatype DT_INT32 on DEVICE_GPU'):
      run_int32_error()

  def testFp16Reduce(self):
    self._setup_context()

    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 100

    @def_function.function
    def run_fp16_reduce():
      collectives = []
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i], dtype=dtypes.float16)
          collectives.append(collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div'))
      return collectives

    for result in run_fp16_reduce():
      self.assertAllClose(result, expected, rtol=1e-3, atol=1e-3)

  def testNcclHintAllReduce(self):
    self._setup_context()

    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_nccl_hint_all_reduce():
      collectives = []
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div',
              communication_hint='nccl'))
      return collectives

    for result in run_nccl_hint_all_reduce():
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testBasicNcclBroadcast(self):
    self._setup_context()

    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_basic_nccl_broadcast():
      collectives = []
      with ops.device(self._devices[0]):
        t = constant_op.constant(tensor_value)
        collectives.append(collective_ops.broadcast_send(
            t, t.shape, t.dtype, self._group_size, group_key, instance_key))
      with ops.device(self._devices[1]):
        t = constant_op.constant(tensor_value)
        collectives.append(collective_ops.broadcast_recv(
            t.shape, t.dtype, self._group_size, group_key, instance_key))
      return collectives

    for result in run_basic_nccl_broadcast():
      self.assertAllClose(result, tensor_value, rtol=1e-5, atol=1e-5)

  def testNcclBroadcastDoubleRecv(self):
    self._setup_context()

    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_nccl_broadcast_double_recv():
      for device in self._devices:
        with ops.device(device):
          t = constant_op.constant(tensor_value)
          collective_ops.broadcast_recv(
              t.shape, t.dtype, self._group_size, group_key, instance_key)

    with self.assertRaisesRegex(errors.InternalError, 'found no source'):
      run_nccl_broadcast_double_recv()

  def testNcclBroadcastDoubleSend(self):
    self._setup_context()

    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_nccl_broadcast_double_send():
      for device in self._devices:
        with ops.device(device):
          t = constant_op.constant(tensor_value)
          collective_ops.broadcast_send(
              t, t.shape, t.dtype, self._group_size, group_key, instance_key)

    with self.assertRaisesRegex(errors.InternalError, 'already has source'):
      run_nccl_broadcast_double_send()

  def testBasicNcclAllGather(self):
    self._setup_context()

    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1,
                0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_basic_nccl_all_gather():
      collectives = []
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(collective_ops.all_gather(t, self._group_size,
                                                       group_key, instance_key))
      return collectives

    for result in run_basic_nccl_all_gather():
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testCollectiveDeviceMismatch(self):
    self._setup_context()

    group_key = 10
    instance_key = 20
    t0 = [1, 2, 3, 4]
    t1 = [5, 6, 7, 8]

    @def_function.function
    def run_collective_device_mismatch():
      with ops.device('/CPU:0'):
        in0 = constant_op.constant(t0)
        collective_ops.all_reduce(in0, self._group_size, group_key,
                                  instance_key, 'Add', 'Id')
      with ops.device('/GPU:0'):
        in1 = constant_op.constant(t1)
        collective_ops.all_reduce(in1, self._group_size, group_key,
                                  instance_key, 'Add', 'Id')

    with self.assertRaisesRegex(errors.InternalError,
                                'but that group has type'):
      run_collective_device_mismatch()

  def testCollectiveReduceMinMax(self):
    self._setup_context()

    @def_function.function
    def run_all_reduce(group_key, instance_key, merge_op):
      t0 = [1., 20., 3., 40., 5.]
      t1 = [10., 2., 30., 4., 50.]
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

  def testNcclStress(self):
    self._setup_context(num_gpus=1)

    num_iters = 1000
    for _ in range(num_iters):
      with ops.device('/device:GPU:0'):
        collective_ops.all_reduce(
            [1.], group_size=1, group_key=0, instance_key=0, merge_op='Add',
            final_op='Id', communication_hint='NCCL')

  @test_util.run_v2_only
  def testAbortNccl(self):
    self._setup_context(num_gpus=2)

    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant(1.)

    # First perform a normal collective to finish resolution.
    def collective_fn():
      for device in ['GPU:0', 'GPU:1']:
        with ops.device(device):
          collective_ops.all_reduce(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              'Add',
              'Id',
              communication_hint='nccl')

    def_function.function(collective_fn)()

    # Launch a collective that hangs, and abort the collective executor after
    # the launch.
    def abort_fn():
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          'Add',
          'Id',
          communication_hint='nccl')

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          'Add',
          'Id',
          communication_hint='nccl')

    t.join()
    # Reset the context in order to reset the collective executor.
    context._reset_context()  # pylint: disable=protected-access
    def_function.function(collective_fn)()


if __name__ == '__main__':
  test.main()
