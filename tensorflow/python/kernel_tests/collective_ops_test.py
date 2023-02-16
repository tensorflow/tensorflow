# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for V2 Collective Operations."""

import os
import threading
import time
from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.experimental.ops import testing as dataset_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import collective_ops as _collective_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


def create_ordering_token():
  return resource_variable_ops.ResourceVariable(1.0).handle


class CollectiveOpsV1(object):

  @staticmethod
  def all_reduce(t, group_size, group_key, instance_key, *args, **kwargs):
    kwargs.pop('ordering_token', None)
    return _collective_ops.all_reduce(t, group_size, group_key, instance_key,
                                      *args, **kwargs)

  @staticmethod
  def all_gather(t, group_size, group_key, instance_key, *args, **kwargs):
    kwargs.pop('ordering_token', None)
    return _collective_ops.all_gather(t, group_size, group_key, instance_key,
                                      *args, **kwargs)

  broadcast_send = _collective_ops.broadcast_send
  broadcast_recv = _collective_ops.broadcast_recv


class CollectiveOpsV2(object):

  @staticmethod
  def all_reduce(t, group_size, group_key, instance_key, *args, **kwargs):
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    return _collective_ops.all_reduce_v2(t, group_size, group_key, instance_key,
                                         *args, **kwargs)

  @staticmethod
  def all_gather(t, group_size, group_key, instance_key, *args, **kwargs):
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    return _collective_ops.all_gather_v2(t, group_size, group_key, instance_key,
                                         *args, **kwargs)

  @staticmethod
  def broadcast_send(t, shape, dtype, group_size, group_key, instance_key,
                     *args, **kwargs):
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    return _collective_ops.broadcast_send_v2(t, group_size, group_key,
                                             instance_key, *args, **kwargs)

  @staticmethod
  def broadcast_recv(shape, dtype, group_size, group_key, instance_key, *args,
                     **kwargs):
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    shape = array_ops.identity(shape)
    return _collective_ops.broadcast_recv_v2(shape, dtype, group_size,
                                             group_key, instance_key, *args,
                                             **kwargs)

  @staticmethod
  def all_to_all(t, group_size, group_key, instance_key, *args, **kwargs):
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    return _collective_ops.all_to_all_v2(
        t, group_size, group_key, instance_key, *args, **kwargs
    )


device_combination = (
    combinations.combine(device='CPU', communication='RING', required_gpus=0) +
    combinations.combine(
        device='GPU', communication=['RING', 'NCCL'], required_gpus=2))

collective_op_combinations = combinations.combine(collective_op=[
    combinations.NamedObject('all_reduce', CollectiveOpsV1.all_reduce),
    combinations.NamedObject('all_gather', CollectiveOpsV1.all_gather),
    combinations.NamedObject('all_reduce_v2', CollectiveOpsV2.all_reduce),
    combinations.NamedObject('all_gather_v2', CollectiveOpsV2.all_gather)
])


@combinations.generate(
    combinations.times(
        combinations.combine(
            collective_ops=[
                combinations.NamedObject('v1', CollectiveOpsV1),
                combinations.NamedObject('v2', CollectiveOpsV2)
            ],
            mode='eager'), device_combination))
class CollectiveOpsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context(num_devices=16)
    super().setUp()

  def testReduce(self, collective_ops, device, communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    tokens = {}
    for dev in [dev0, dev1]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_reduce_1device():
      with ops.device(dev0):
        in_value = constant_op.constant([1.])
        group_size = 1
        group_key = 1
        instance_key = 1
        return collective_ops.all_reduce(
            in_value,
            group_size,
            group_key,
            instance_key,
            communication_hint=communication,
            ordering_token=tokens[dev0])

    @def_function.function
    def run_all_reduce_2devices():
      in_value = constant_op.constant([1.])
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.all_reduce(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev0],
                communication_hint=communication))
      with ops.device(dev1):
        collectives.append(
            collective_ops.all_reduce(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev1],
                communication_hint=communication))
      return collectives

    self.assertAllClose(run_all_reduce_1device(), [1.], rtol=1e-5, atol=1e-5)
    for result in run_all_reduce_2devices():
      self.assertAllClose(result, [2.], rtol=1e-5, atol=1e-5)

  def testGather(self, collective_ops, device, communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    tokens = {}
    for dev in [dev0, dev1]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_gather_1device():
      with ops.device(dev0):
        in_value = constant_op.constant([1.])
        group_size = 1
        group_key = 1
        instance_key = 1
        return collective_ops.all_gather(
            in_value,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    @def_function.function
    def run_all_gather_2devices():
      in_value = constant_op.constant([1.])
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.all_gather(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev0],
                communication_hint=communication))
      with ops.device(dev1):
        collectives.append(
            collective_ops.all_gather(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev1],
                communication_hint=communication))
      return collectives

    cpu_tokens = {}
    for i in range(16):
      with ops.device('/device:CPU:%d' % i):
        cpu_tokens[i] = create_ordering_token()

    @def_function.function
    def run_all_gather_16devices():
      group_size = 16
      group_key = 3
      instance_key = 1
      collectives = []
      for i in range(16):
        with ops.device('/device:CPU:%d' % i):
          collectives.append(
              collective_ops.all_gather(
                  constant_op.constant([i]),
                  group_size,
                  group_key,
                  instance_key,
                  ordering_token=cpu_tokens[i],
                  communication_hint=communication))
      return collectives

    self.assertAllClose(run_all_gather_1device(), [1.], rtol=1e-5, atol=1e-5)
    for result in run_all_gather_2devices():
      self.assertAllClose(result, [1., 1.], rtol=1e-5, atol=1e-5)

    for result in run_all_gather_16devices():
      self.assertAllClose(
          result, list(range(16)), rtol=1e-5, atol=1e-5)

  def testBroadcast(self, collective_ops, device, communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_broadcast_2devices():
      shape = [3]
      in_value = constant_op.constant([1., 2., 3.], shape=shape)
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.broadcast_send(
                in_value,
                shape,
                in_value.dtype,
                group_size,
                group_key,
                instance_key,
                communication_hint=communication))
      with ops.device(dev1):
        collectives.append(
            collective_ops.broadcast_recv(
                shape,
                in_value.dtype,
                group_size,
                group_key,
                instance_key,
                communication_hint=communication))
      return collectives

    for result in run_broadcast_2devices():
      self.assertAllClose(result, [1., 2., 3.], rtol=1e-5, atol=1e-5)

  def testAllToAll(self, collective_ops, device, communication):
    if str(collective_ops) == 'v1':
      self.skipTest('CollectiveAllToAllV1 is not implemented.')
    devices = ['/device:%s:0' % device, '/device:%s:1' % device]

    tokens = {}
    for dev in devices:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_to_all_1device():
      with ops.device(devices[0]):
        in_value = constant_op.constant([1.0])
        group_size = 1
        group_key = 1
        instance_key = 1
        return collective_ops.all_to_all(
            in_value,
            group_size,
            group_key,
            instance_key,
            communication_hint=communication,
            ordering_token=tokens[devices[0]],
        )

    @def_function.function
    def run_all_to_all_2devices():
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      for i in range(2):
        with ops.device(devices[i]):
          collectives.append(
              collective_ops.all_to_all(
                  constant_op.constant([i, i]),
                  group_size,
                  group_key,
                  instance_key,
                  ordering_token=tokens[devices[i]],
                  communication_hint=communication,
              )
          )
      return collectives

    self.assertAllClose(run_all_to_all_1device(), [1.0])
    for result in run_all_to_all_2devices():
      self.assertAllClose(result, [0.0, 1.0])

  def testInstanceKeyScopedUnderGroupKey(self, collective_ops, device,
                                         communication):
    if device == 'GPU' and context.num_gpus() < 4:
      self.skipTest('not enough GPU')

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    dev2 = '/device:%s:2' % device
    dev3 = '/device:%s:3' % device

    tokens = {}
    for dev in [dev0, dev1, dev2, dev3]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_reduce_4devices_same_instance_key():
      # Use a common instance key for both groups.
      instance_key = 0
      # We will create 2 groups each with 2 devices.
      group_size = 2
      # Group 0 comprises dev0 and dev1.
      group0_key = 0
      # Group 1 comprises dev2 and dev3.
      group1_key = 1
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(1.),
                group_size,
                group0_key,
                instance_key,
                ordering_token=tokens[dev0],
            ))
      with ops.device(dev1):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(2.),
                group_size,
                group0_key,
                instance_key,
                ordering_token=tokens[dev1],
            ))
      with ops.device(dev2):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(3.),
                group_size,
                group1_key,
                instance_key,
                ordering_token=tokens[dev2],
            ))
      with ops.device(dev3):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(4.),
                group_size,
                group1_key,
                instance_key,
                ordering_token=tokens[dev3],
            ))
      return collectives

    results = run_all_reduce_4devices_same_instance_key()
    self.assertAllClose(results[0], 3., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[1], 3., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[2], 7., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[3], 7., rtol=1e-5, atol=1e-5)

  def testCollectiveGroupSizeOne(self, collective_ops, device, communication):
    dev0 = '/device:%s:0' % device

    group_size = 1
    group_key = 100
    in_value = [1., 2., 3., 4.]
    in_tensor = constant_op.constant(in_value)

    tokens = {}
    for dev in [dev0]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    with ops.device(dev0):
      reduced_tensor = collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key=100,
          ordering_token=tokens[dev0],
          communication_hint=communication)
    self.assertAllEqual(in_value, reduced_tensor.numpy())

    with ops.device(dev0):
      gathered_tensor = collective_ops.all_gather(
          in_tensor,
          group_size,
          group_key,
          instance_key=200,
          ordering_token=tokens[dev0],
          communication_hint=communication)
    self.assertAllEqual(in_value, gathered_tensor.numpy())

  def testCollectiveInvalidKey(self, collective_ops, device, communication):
    dev0 = '/device:%s:0' % device

    group_size = 1
    group_key = 100
    instance_key = 100
    in_value = [1., 2., 3., 4.]
    in_tensor = constant_op.constant(in_value)

    tokens = {}
    for dev in [dev0]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    with ops.device(dev0):
      reduced_tensor = collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          ordering_token=tokens[dev0],
          communication_hint=communication)
    self.assertAllEqual(in_value, reduced_tensor.numpy())

    with self.assertRaisesRegex(
        errors.InternalError, 'instance 100 expected type 0 and data_type 1 but'
        ' got type 2 and data_type 1'):
      with ops.device(dev0):
        collective_ops.all_gather(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

  def testMultipleGroups(self, collective_ops, device, communication):
    if device == 'GPU' and context.num_gpus() < 4:
      self.skipTest('not enough GPU')

    num_elements = 4
    tokens = {}
    for device_idx in range(num_elements):
      dev = '/{}:{}'.format(device, device_idx)
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_reduce(group_size, group_key):
      instance_key = group_key
      input_value = [float(group_key) for i in range(num_elements)]
      collectives = []
      for device_idx in range(group_size):
        dev = '/{}:{}'.format(device, device_idx)
        with ops.device(dev):
          input_tensor = constant_op.constant(input_value)
          collectives.append(
              collective_ops.all_reduce(
                  input_tensor,
                  group_size,
                  group_key,
                  instance_key,
                  ordering_token=tokens[dev],
                  communication_hint=communication))
      return collectives

    def run_and_assert(group_size, group_key):
      for reduced_tensor in run_all_reduce(group_size, group_key):
        self.assertAllEqual(
            [float(group_key) * group_size for i in range(num_elements)],
            reduced_tensor.numpy())

    run_and_assert(group_size=2, group_key=1)
    run_and_assert(group_size=3, group_key=2)


@combinations.generate(
    combinations.times(
        combinations.combine(
            collective_ops=[combinations.NamedObject('v2', CollectiveOpsV2)],
            mode='eager',
            max_subdivs_per_device=[-1, 0, 16]), device_combination))
class AllReduceWithSubdivisionsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  def testReduce(self, collective_ops, device, communication,
                 max_subdivs_per_device):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    tokens = {}
    for dev in [dev0, dev1]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_reduce_1device():
      with ops.device(dev0):
        in_value = constant_op.constant([1.])
        group_size = 1
        group_key = 1
        instance_key = 1
        if max_subdivs_per_device == -1:
          return collective_ops.all_reduce(
              in_value,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[dev0],
              communication_hint=communication)
        else:
          return collective_ops.all_reduce(
              in_value,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[dev0],
              communication_hint=communication,
              max_subdivs_per_device=max_subdivs_per_device)

    @def_function.function
    def run_all_reduce_2devices():
      in_value = constant_op.constant([1.])
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.all_reduce(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev0],
                communication_hint=communication))
      with ops.device(dev1):
        collectives.append(
            collective_ops.all_reduce(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev1],
                communication_hint=communication))
      return collectives

    self.assertAllClose(run_all_reduce_1device(), [1.], rtol=1e-5, atol=1e-5)
    for result in run_all_reduce_2devices():
      self.assertAllClose(result, [2.], rtol=1e-5, atol=1e-5)


@combinations.generate(
    combinations.combine(required_physical_gpus=2, mode='eager'))
class XlaTest(test.TestCase, parameterized.TestCase):

  def testReduce(self):
    device0 = '/device:GPU:0'
    device1 = '/device:GPU:1'
    group_size = 2
    group_key = 100
    instance_key = 100
    results = []

    def all_reduce(device):

      with ops.device(device):
        token = create_ordering_token()

      @def_function.function(jit_compile=True)
      def f():
        return _collective_ops.all_reduce_v2([1.],
                                             group_size,
                                             group_key,
                                             instance_key,
                                             ordering_token=token)

      with ops.device(device):
        results.append(f())

    t0 = threading.Thread(target=all_reduce, args=(device0,))
    t1 = threading.Thread(target=all_reduce, args=(device1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    self.assertAllEqual(results, [[2.], [2.]])

  def testReduceSameGraph(self):
    device0 = '/device:GPU:0'
    device1 = '/device:GPU:1'
    group_size = 2
    group_key = 100
    instance_key = 100
    results = []

    @def_function.function(jit_compile=True)
    def func():

      def all_reduce(device):

        with ops.device(device):
          token = create_ordering_token()

          return _collective_ops.all_reduce_v2([1.],
                                               group_size,
                                               group_key,
                                               instance_key,
                                               ordering_token=token)

      results.append(all_reduce(device0))
      results.append(all_reduce(device1))
      return results

    # FIXME(b/204228837): the error shall no longer be about resources
    # after multi-device support in jit_compile lands. This will likely
    # becomes a deadlock near ResolveDeviceAssignment, or an error in the MLIR
    # bridge on resetting CollectiveInfo.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Trying to access resource'):
      func()


@combinations.generate(
    combinations.combine(
        required_physical_gpus=2, mode='eager', jit_compile=[True, False]))
class GroupAssignmentTest(test.TestCase, parameterized.TestCase):

  def testGroupAssignmentBeforeAllReduce(self, jit_compile):
    device0 = '/device:GPU:0'
    device1 = '/device:GPU:1'
    instance_key = 100
    results = []

    group_assignment = [[0], [1]]

    def all_reduce(device, device_index):

      with ops.device(device):
        token = create_ordering_token()

      @def_function.function(jit_compile=jit_compile)
      def f(device_index):
        group_size, group_key = _collective_ops.assign_group_v2(
            group_assignment=group_assignment,
            device_index=device_index,
            base_key=1)
        return _collective_ops.all_reduce_v2([1.],
                                             group_size,
                                             group_key,
                                             instance_key,
                                             ordering_token=token)

      with ops.device(device):
        results.append(f(device_index))

    t0 = threading.Thread(target=all_reduce, args=(device0, 0))
    t1 = threading.Thread(target=all_reduce, args=(device1, 1))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    self.assertAllEqual(results, [[1.], [1.]])

  def testTwoGroupAssignmentBeforeAllReduce(self, jit_compile):
    device0 = '/device:GPU:0'
    device1 = '/device:GPU:1'
    instance_key = 100
    results = []

    group_assignment1 = [[0], [1]]
    group_assignment2 = [[0, 1]]

    def all_reduce(device, device_index):

      with ops.device(device):
        token = create_ordering_token()

      @def_function.function(jit_compile=jit_compile)
      def f(device_index):
        group_size, group_key = _collective_ops.assign_group_v2(
            group_assignment=group_assignment1,
            device_index=device_index,
            base_key=1)
        r1 = _collective_ops.all_reduce_v2([1.],
                                           group_size,
                                           group_key,
                                           instance_key,
                                           ordering_token=token)

        group_size, group_key = _collective_ops.assign_group_v2(
            group_assignment=group_assignment2,
            device_index=device_index,
            base_key=10000)
        r2 = _collective_ops.all_reduce_v2([1.],
                                           group_size,
                                           group_key,
                                           instance_key,
                                           ordering_token=token)
        return r1, r2

      with ops.device(device):
        results.append(f(device_index))

    t0 = threading.Thread(target=all_reduce, args=(device0, 0))
    t1 = threading.Thread(target=all_reduce, args=(device1, 1))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    self.assertAllEqual(results, [[[1.], [2.]], [[1.], [2.]]])


@combinations.generate(
    combinations.times(collective_op_combinations, device_combination))
class AbortCollectiveOpsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  def testAbortGroupParamsResolution(self, collective_op, device,
                                     communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    def abort_fn():
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      # This hangs on params resolution since we're only launching one
      # collective for a group size of 2.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    t.join()
    # Reset the context in order to reset the collective executor.
    _setup_context()

    # After reset non-NCCL collectives should work.
    def collective_fn():
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    def_function.function(collective_fn)()

  def testAbortInstanceParamsResolution(self, collective_op, device,
                                        communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    def collective_fn():
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    # First perform a normal all-reduce to complete the group resolution.
    def_function.function(collective_fn)()

    def abort_fn():
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    # Use a different instance key to trigger another instance resolution.
    instance_key = 101
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      # This hangs on params resolution since we're only launching one
      # collective for a group size of 2.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    context._reset_context()  # pylint: disable=protected-access
    t.join()
    # Reset the context in order to reset the collective executor.
    _setup_context()

    # After reset non-NCCL collectives should work.
    def_function.function(collective_fn)()

  def testAbortCommunication(self, collective_op, device, communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    # First perform a normal collective to finish resolution.
    def collective_fn():
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    def_function.function(collective_fn)()

    # Launch a collective that hangs, and abort the collective executor after
    # the launch.
    def abort_fn():
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[device],
            communication_hint=communication)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[device],
            communication_hint=communication)

    # Reset the context in order to reset the collective executor.
    t.join()
    _setup_context()
    def_function.function(collective_fn)()


class OpCancellationTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce',
                                           CollectiveOpsV1.all_reduce),
                  combinations.NamedObject('all_gather',
                                           CollectiveOpsV1.all_gather),
              ],
              mode='eager'), device_combination))
  def testOpErrorNotAbortIfNoCollective(self, collective_op, device,
                                        communication):
    # Do not abort if there's no active collective ops. There could be
    # exceptions like EOF which we expect users to catch, aborting collective
    # ops on all op errors intervenes with this workflow.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    dataset = dataset_ops.Dataset.from_tensors([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def collective_fn(in_tensor):
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    @def_function.function
    def f():
      iterator = iter(dataset)
      collective_fn(next(iterator))
      # This next(iterator) should raise EOF.
      collective_fn(next(iterator))

    collective_fn(constant_op.constant([1.]))
    with self.assertRaises(errors.OutOfRangeError):
      f()
    collective_fn(constant_op.constant([1.]))

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce_v2',
                                           CollectiveOpsV2.all_reduce),
                  combinations.NamedObject('all_gather_v2',
                                           CollectiveOpsV2.all_gather),
              ],
              mode='eager'), device_combination))
  def testOpErrorNotAbortIfNoCollectiveV2(self, collective_op, device,
                                          communication):
    # Do not abort if there's no active collective ops. There could be
    # exceptions like EOF which we expect users to catch, aborting collective
    # ops on all op errors intervenes with this workflow.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    dataset = dataset_ops.Dataset.from_tensors([1.])
    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def collective_fn(in_tensor):
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              communication_hint=communication,
              ordering_token=tokens[device])

    @def_function.function
    def f():
      iterator = iter(dataset)
      collective_fn(next(iterator))
      # This next(iterator) should raise EOF.
      collective_fn(next(iterator))

    collective_fn(constant_op.constant([1.]))
    with self.assertRaises(errors.OutOfRangeError):
      f()
    collective_fn(constant_op.constant([1.]))

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce',
                                           CollectiveOpsV1.all_reduce),
                  combinations.NamedObject('all_gather',
                                           CollectiveOpsV1.all_gather),
              ],
              mode='eager'), device_combination))
  def testOpErrorAbortWithCollective(self, collective_op, device,
                                     communication):
    # Abort v1 collective ops if there're active collective ops at the time of
    # an op error. This is due to the inability to cancel collective ops, and op
    # errors may cause running collective ops to hang.
    dev0 = '/device:%s:0' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])
    # Make the dataset sleep a while so that the collective is being executed
    # when the EOF happens.
    dataset = dataset_ops.Dataset.from_tensors([1.]).apply(
        dataset_testing.sleep(sleep_microseconds=200))

    tokens = {}
    for device in [dev0]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def f():
      # Launch a collective op that won't be able to finish to test abortion
      # when other ops error.
      with ops.device(dev0):
        ret = collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)
      iterator = iter(dataset)
      next(iterator)
      # This should raise EOF.
      next(iterator)
      return ret

    with self.assertRaises(errors.OutOfRangeError):
      f()
    # Now collective ops is aborted, subsequent collective ops should fail with
    # the previous error.
    with self.assertRaises(errors.CancelledError):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce_v2',
                                           CollectiveOpsV2.all_reduce),
                  combinations.NamedObject('all_gather_v2',
                                           CollectiveOpsV2.all_gather),
              ],
              mode='eager'), device_combination))
  def testOpErrorNotAbortWithCollectiveV2(self, collective_op, device,
                                          communication):
    # Do not abort v2 collective ops even if there're active collective ops at
    # the time of an op error. We rely cancellation to terminate active
    # collective ops.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def collective_fn():
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    # Local params resolution cannot be cancelled yet, so we perform a normal
    # collective so that the group is resolved.
    collective_fn()

    # Make the dataset sleep a while so that the collective is being executed
    # when the EOF happens.
    dataset = dataset_ops.Dataset.from_tensors([1.]).apply(
        dataset_testing.sleep(sleep_microseconds=200))

    @def_function.function
    def f():
      # Launch a collective op that won't be able to finish to test cancellation
      # when other ops error.
      with ops.device(dev0):
        ret = collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)
      iterator = iter(dataset)
      next(iterator)
      # This should raise EOF.
      next(iterator)
      return ret

    with self.assertRaises(errors.OutOfRangeError):
      f()
    # Collective ops shouldn't be aborted and new collectives should be able to
    # proceed.
    collective_fn()

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce_v2',
                                           CollectiveOpsV2.all_reduce),
                  combinations.NamedObject('all_gather_v2',
                                           CollectiveOpsV2.all_gather),
              ],
              mode='eager'), device_combination))
  def testCancelDuringParamResolutionV2(self, collective_op, device,
                                        communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])
    t1_cancellation_manager = cancellation.CancellationManager()
    t2_cancellation_manager = cancellation.CancellationManager()

    @def_function.function
    def _collective_fn(x):
      # Run an assertion to crash one of the two function executions running
      # collectives. We explicitly cancel the other in response.
      assert_op = check_ops.assert_equal(x, in_tensor)
      with ops.control_dependencies([assert_op]):
        return collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            # This test cannot use ordering_token because the placement
            # occurs outside of tf.function and we cannot relocate the token
            # after concrete function is created.
            # since there is only 1 collective Op in the graph there is no
            # need to use a token for ordering.
            communication_hint=communication)

    collective_concrete = _collective_fn.get_concrete_function(in_tensor)

    finish_mu = threading.Lock()
    finishes = 0

    def _placement_wrapper(device, x, my_cancellation, other_cancellation):
      try:
        with ops.device(device):
          cancelable_collective = my_cancellation.get_cancelable_function(
              collective_concrete)
          return cancelable_collective(x)
      except errors.InvalidArgumentError:
        # `assert_equal` failed for this execution of the function. The other
        # function would deadlock without cancellation.
        other_cancellation.start_cancel()
      except errors.CancelledError:
        pass
      nonlocal finishes
      with finish_mu:
        finishes += 1

    t1 = threading.Thread(
        target=_placement_wrapper,
        args=(dev0, constant_op.constant([1.]), t1_cancellation_manager,
              t2_cancellation_manager))
    t2 = threading.Thread(
        target=_placement_wrapper,
        # Will cause the assertion to fail
        args=(dev1, constant_op.constant([2.]), t2_cancellation_manager,
              t1_cancellation_manager))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    self.assertEqual(finishes, 2)


@combinations.generate(
    combinations.times(collective_op_combinations, device_combination))
class TimeoutTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  def testTimeout(self, collective_op, device, communication):
    timeout = 1.5

    tokens = {}
    for i in range(2):
      dev = '/{}:{}'.format(device, i)
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run(group_size, reported_group_size=None):
      group_key = 20
      instance_key = 30
      tensor = [1., 2., 3., 4.]
      results = []
      if reported_group_size is None:
        reported_group_size = group_size
      for i in range(group_size):
        dev = '/{}:{}'.format(device, i)
        with ops.device(dev):
          input_data = constant_op.constant(tensor)
          result = collective_op(
              input_data,
              group_size=reported_group_size,
              group_key=group_key,
              instance_key=instance_key,
              ordering_token=tokens[dev],
              communication_hint=communication,
              timeout=timeout)
          results.append(result)
      return results

    run(2, 2)

    start_time = time.time()
    with self.assertRaisesRegex(errors.DeadlineExceededError,
                                'Collective has timed out during execution'):
      run(1, 2)
    elapsed = time.time() - start_time
    self.assertAllGreaterEqual(elapsed, timeout)

  def testParamResolutionAfterTimeout(self, collective_op, device,
                                      communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    timeout = 1.5
    group_key = 20
    instance_key = 30
    input_data = constant_op.constant([1., 2., 3., 4.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    # This timeout comes from param solution.
    with self.assertRaisesRegex(
        errors.DeadlineExceededError,
        'Collective has timed out waiting for other workers'):
      with ops.device(dev0):
        collective_op(
            input_data,
            group_size=2,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication,
            timeout=timeout)

    # We launch the second device after the first device times out. This is to
    # simulate the situation when other workers are slow and the timeout is
    # short. It should error immediately.
    with self.assertRaisesRegex(
        errors.DeadlineExceededError,
        'Collective has timed out waiting for other workers'):
      with ops.device(dev1):
        collective_op(
            input_data,
            group_size=2,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=tokens[dev1],
            communication_hint=communication)

  def testExecutionAfterTimeout(self, collective_op, device, communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    timeout = 1.5
    group_key = 20
    instance_key = 30
    input_data = constant_op.constant([1., 2., 3., 4.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def run():
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              input_data,
              group_size=2,
              group_key=group_key,
              instance_key=instance_key,
              ordering_token=tokens[device],
              communication_hint=communication,
              timeout=timeout)

    # Run a normal all-reduce to complete param resolution.
    run()

    with self.assertRaisesRegex(errors.DeadlineExceededError,
                                'Collective has timed out during execution'):
      with ops.device(dev0):
        collective_op(
            input_data,
            group_size=2,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication,
            timeout=timeout)

    # We launch the second device after the first device times out. This is to
    # simulate the situation when other workers are slow and the timeout is
    # short. It should error immediately.
    with self.assertRaisesRegex(errors.DeadlineExceededError,
                                'Collective has timed out during execution'):
      with ops.device(dev1):
        # No timeout.
        collective_op(
            input_data,
            group_size=2,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=tokens[dev1],
            communication_hint=communication)


class CommunicationHintTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  @combinations.generate(
      combinations.times(collective_op_combinations,
                         combinations.combine(required_gpus=[0, 1])))
  def testNCCLFallbackOnCPU(self, collective_op):
    # communication_hint=NCCL should work for CPU by falling back to RING. The
    # test doesn't actually require GPU, only GPU builds. We specify
    # required_gpus=1 so that it's tested with GPU builds.
    dev0 = '/device:CPU:0'
    dev1 = '/device:CPU:1'
    group_key = 20
    instance_key = 30
    input_data = constant_op.constant([1., 2., 3., 4.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def run():
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              input_data,
              group_size=2,
              group_key=group_key,
              instance_key=instance_key,
              ordering_token=tokens[device],
              communication_hint='NCCL')

    run()


@combinations.generate(
    combinations.times(
        combinations.combine(
            collective_op=[
                combinations.NamedObject('all_reduce_v2',
                                         CollectiveOpsV2.all_reduce),
                combinations.NamedObject('all_gather_v2',
                                         CollectiveOpsV2.all_gather),
            ],
            mode='eager'), device_combination))
class OrderingTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  def testOrdering(self, collective_op, device, communication):
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    with ops.device(dev0):
      token0 = create_ordering_token()
    with ops.device(dev1):
      token1 = create_ordering_token()

    @def_function.function
    def f():
      # Launch the first collective with token.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=token0,
            name='FirstChainedDev0')
      with ops.device(dev1):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=token1,
            name='FirstChainedDev1')
      # Launch the second collective without token.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            name='UnchainedDev0')
      with ops.device(dev1):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            name='UnchainedDev1')
      # Launch the third collective with token.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key + 1,
            ordering_token=token0,
            name='SecondChainedDev0')
      with ops.device(dev1):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key + 1,
            ordering_token=token1,
            name='SecondChainedDev1')

    graph = f.get_concrete_function().graph
    for device, suffix in [(dev0, 'Dev0'), (dev1, 'Dev1')]:

      first = graph.get_operation_by_name('FirstChained' + suffix)
      second = graph.get_operation_by_name('Unchained' + suffix)
      third = graph.get_operation_by_name('SecondChained' + suffix)
      self.assertIsNotNone(first)
      self.assertTrue(first.device.endswith(device))
      self.assertIsNotNone(second)
      self.assertTrue(second.device.endswith(device))
      self.assertIsNotNone(third)
      self.assertTrue(third.device.endswith(device))

      # Try to find the third collective, which should have the first collective
      # as a control input.
      self.assertLen(third.control_inputs, 1)
      self.assertEqual(third.control_inputs[0].name, 'FirstChained' + suffix)

      self.assertEmpty(second.control_inputs)
      self.assertEmpty(first.control_inputs)


class InputPipelineTest(test.TestCase):

  def setUp(self):
    super().setUp()
    _setup_context()

  def testMap(self):
    group_size = 2
    group_key = 100
    instance_key = 100

    def create_dataset_and_fetch_one(t):
      dataset = dataset_ops.Dataset.from_tensor_slices([t])

      def reduce_fn(t):
        # A token is created for each device.
        token = create_ordering_token()
        return CollectiveOpsV2.all_reduce(
            t,
            group_size=group_size,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=token)

      dataset = dataset.map(reduce_fn)
      return next(iter(dataset))

    @def_function.function
    def f():
      with ops.device('CPU:0'):
        value0 = create_dataset_and_fetch_one([1.])
      with ops.device('CPU:1'):
        value1 = create_dataset_and_fetch_one([2.])
      return value0, value1

    self.assertAllEqual(self.evaluate(f()), [[3.], [3.]])


@combinations.generate(
    combinations.times(
        combinations.combine(collective_op=[
            combinations.NamedObject('all_reduce_v2',
                                     CollectiveOpsV2.all_reduce),
            combinations.NamedObject('all_gather_v2',
                                     CollectiveOpsV2.all_gather)
        ]), device_combination))
class InvalidInputTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  def testInvalidGroupKey(self, collective_op, device, communication):
    dev0 = '/device:%s:0' % device
    group_size = 2
    group_key = [100]
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    with self.assertRaises(errors.InvalidArgumentError):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            communication_hint=communication)

  def testInvalidGroupSize(self, collective_op, device, communication):
    dev0 = '/device:%s:0' % device
    group_size = -2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    with self.assertRaises(errors.InvalidArgumentError):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            communication_hint=communication)

  def testInvalidInstanceKey(self, collective_op, device, communication):
    dev0 = '/device:%s:0' % device
    group_size = 2
    group_key = 100
    instance_key = [100]
    in_tensor = constant_op.constant([1.])

    with self.assertRaises(errors.InvalidArgumentError):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            communication_hint=communication)


class CollectiveOpsV3Test(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _setup_context()

  def testGroupInitialization(self):
    group_size = 2
    group_key = 100

    @def_function.function
    def f():
      with ops.device('CPU:0'):
        _collective_ops.initialize_communicator(
            group_key=group_key, rank=0, group_size=group_size)
      with ops.device('CPU:1'):
        _collective_ops.initialize_communicator(
            group_key=group_key, rank=1, group_size=group_size)

      # TODO(b/193864859): Add validation with reduction op.

    self.evaluate(f())

  @combinations.generate(device_combination)
  def testAllReduceV3(self, device, communication):
    group_size = 2
    group_key = 101

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_all_reduce_2devices():
      collectives = []
      with ops.device(dev0):
        group_handle0 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=0,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_reduce_v3(
                group_handle0, [1.0], reduction='Add'))
      with ops.device(dev1):
        group_handle1 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=1,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_reduce_v3(
                group_handle1, [2.0], reduction='Add'))
      return collectives

    for result in run_all_reduce_2devices():
      self.assertAllClose(result, [3.], rtol=1e-5, atol=1e-5)

  @combinations.generate(device_combination)
  def testAllToAllV3(self, device, communication):
    group_size = 2
    group_key = 104

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_all_to_all_2devices():
      collectives = []
      with ops.device(dev0):
        group_handle0 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=0,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle0, [1.0, 3.0]))
      with ops.device(dev1):
        group_handle1 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=1,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle1, [2.0, 4.0]))
      return collectives

    result = run_all_to_all_2devices()
    self.assertAllClose(result[0], [1.0, 2.0], rtol=1e-5, atol=1e-5)
    self.assertAllClose(result[1], [3.0, 4.0], rtol=1e-5, atol=1e-5)

  @combinations.generate(device_combination)
  def testAllToAllV3DifferentUserRank(self, device, communication):
    group_size = 2
    group_key = 105

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_all_to_all_2devices():
      collectives = []
      with ops.device(dev0):
        group_handle0 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=1,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle0, [1.0, 3.0]))
      with ops.device(dev1):
        group_handle1 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=0,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle1, [2.0, 4.0]))
      return collectives

    result = run_all_to_all_2devices()
    self.assertAllClose(result[0], [2.0, 1.0], rtol=1e-5, atol=1e-5)
    self.assertAllClose(result[1], [4.0, 3.0], rtol=1e-5, atol=1e-5)

  @combinations.generate(device_combination)
  def testAllToAllV3DifferentUserRankWithTensorInput(self, device,
                                                     communication):

    group_size = 2
    group_key = 106

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_all_to_all_2devices():
      collectives = []
      with ops.device(dev0):
        group_handle0 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=1,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle0,
                                          constant_op.constant([1.0, 2.0])))
      with ops.device(dev1):
        group_handle1 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=0,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle1,
                                          constant_op.constant([3.0, 4.0])))

      return collectives

    result = run_all_to_all_2devices()
    # FIXME(b/214407359): This is correct.
    # result[0] is rank 1 and shall have 4, 2.
    self.assertAllClose(result[1], [4.0, 2.0], rtol=1e-5, atol=1e-5)
    self.assertAllClose(result[0], [3.0, 1.0], rtol=1e-5, atol=1e-5)


def _setup_context(num_devices=4):
  context._reset_context()
  test_util.set_logical_devices_to_at_least('CPU', num_devices)
  context.ensure_initialized()
  context.set_log_device_placement(True)


if __name__ == '__main__':
  os.environ['NCCL_DEBUG'] = 'INFO'
  v2_compat.enable_v2_behavior()
  test.main()
