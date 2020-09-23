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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import collective_ops as _collective_ops
from tensorflow.python.platform import test


class CollectiveOpsV1(object):
  all_reduce = _collective_ops.all_reduce
  all_gather = _collective_ops.all_gather


class CollectiveOpsV2(object):
  all_reduce = _collective_ops.all_reduce_v2
  all_gather = _collective_ops.all_gather_v2


@combinations.generate(
    combinations.combine(
        collective_ops=[
            combinations.NamedObject('v1', CollectiveOpsV1),
            combinations.NamedObject('v2', CollectiveOpsV2)
        ],
        mode='eager'))
class CollectiveOpsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  def testReduce(self, collective_ops):

    @def_function.function
    def run_all_reduce_1cpu():
      with ops.device('/device:CPU:0'):
        in_value = constant_op.constant([1.])
        group_size = 1
        group_key = 1
        instance_key = 1
        return collective_ops.all_reduce(in_value, group_size, group_key,
                                         instance_key)

    @def_function.function
    def run_all_reduce_2cpus():
      in_value = constant_op.constant([1.])
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device('/device:CPU:0'):
        collectives.append(
            collective_ops.all_reduce(in_value, group_size, group_key,
                                      instance_key))
      with ops.device('/device:CPU:1'):
        collectives.append(
            collective_ops.all_reduce(in_value, group_size, group_key,
                                      instance_key))
      return collectives

    self.assertAllClose(run_all_reduce_1cpu(), [1.], rtol=1e-5, atol=1e-5)
    for result in run_all_reduce_2cpus():
      self.assertAllClose(result, [2.], rtol=1e-5, atol=1e-5)

  def testGather(self, collective_ops):

    @def_function.function
    def run_all_gather_1cpu():
      with ops.device('/device:CPU:0'):
        in_value = constant_op.constant([1.])
        group_size = 1
        group_key = 1
        instance_key = 1
        return collective_ops.all_gather(in_value, group_size, group_key,
                                         instance_key)

    @def_function.function
    def run_all_gather_2cpus():
      in_value = constant_op.constant([1.])
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device('/device:CPU:0'):
        collectives.append(
            collective_ops.all_gather(in_value, group_size, group_key,
                                      instance_key))
      with ops.device('/device:CPU:1'):
        collectives.append(
            collective_ops.all_gather(in_value, group_size, group_key,
                                      instance_key))
      return collectives

    self.assertAllClose(run_all_gather_1cpu(), [1.], rtol=1e-5, atol=1e-5)
    for result in run_all_gather_2cpus():
      self.assertAllClose(result, [1., 1.], rtol=1e-5, atol=1e-5)

  def testInstanceKeyScopedUnderGroupKey(self, collective_ops):

    @def_function.function
    def run_all_reduce_4cpus_same_instance_key():
      # Use a common instance key for both groups.
      instance_key = 0
      # We will create 2 groups each with 2 devices.
      group_size = 2
      # Group 0 comprises cpu:0 and cpu:1.
      group0_key = 0
      # Group 1 comprises cpu:2 and cpu:3.
      group1_key = 1
      collectives = []
      with ops.device('/device:CPU:0'):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(1.), group_size, group0_key, instance_key))
      with ops.device('/device:CPU:1'):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(2.), group_size, group0_key, instance_key))
      with ops.device('/device:CPU:2'):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(3.), group_size, group1_key, instance_key))
      with ops.device('/device:CPU:3'):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(4.), group_size, group1_key, instance_key))
      return collectives

    results = run_all_reduce_4cpus_same_instance_key()
    self.assertAllClose(results[0], 3., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[1], 3., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[2], 7., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[3], 7., rtol=1e-5, atol=1e-5)

  def testCollectiveGroupSizeOne(self, collective_ops):
    group_size = 1
    group_key = 100
    instance_key = 100
    in_value = [1, 2, 3, 4]
    in_tensor = constant_op.constant(in_value)

    reduced_tensor = collective_ops.all_reduce(in_tensor, group_size, group_key,
                                               instance_key)
    self.assertAllEqual(in_value, reduced_tensor.numpy())

    gathered_tensor = collective_ops.all_gather(
        in_tensor, group_size, group_key, instance_key)
    self.assertAllEqual(in_value, gathered_tensor.numpy())

  def testMultipleGroups(self, collective_ops):
    num_elements = 4

    @def_function.function
    def run_all_reduce(group_size, group_key):
      instance_key = group_key
      input_value = [group_key for i in range(num_elements)]
      collectives = []
      for device_idx in range(group_size):
        with ops.device('/CPU:{}'.format(device_idx)):
          input_tensor = constant_op.constant(input_value)
          collectives.append(
              collective_ops.all_reduce(input_tensor, group_size, group_key,
                                        instance_key))
      return collectives

    def run_and_assert(group_size, group_key):
      for reduced_tensor in run_all_reduce(group_size, group_key):
        self.assertAllEqual(
            [group_key * group_size for i in range(num_elements)],
            reduced_tensor.numpy())

    run_and_assert(group_size=2, group_key=1)
    run_and_assert(group_size=3, group_key=2)


@combinations.generate(
    combinations.combine(
        collective_op=[
            combinations.NamedObject('all_reduce', _collective_ops.all_reduce),
            combinations.NamedObject('all_reduce_v2',
                                     _collective_ops.all_reduce_v2),
            combinations.NamedObject('all_gather', _collective_ops.all_gather),
            combinations.NamedObject('all_gather_v2',
                                     _collective_ops.all_gather_v2),
        ],
        mode='eager',
        communication=['ring']))
class AbortCollectiveOpsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    _setup_context()
    super().setUp()

  def testAbortGroupParamsResolution(self, collective_op, communication):
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    def abort_fn():
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      # This hangs on params resolution since we're only launching one
      # collective for a group size of 2.
      collective_op(in_tensor, group_size, group_key, instance_key)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      collective_op(in_tensor, group_size, group_key, instance_key)

    t.join()
    # Reset the context in order to reset the collective executor.
    _setup_context()

    # After reset non-NCCL collectives should work.
    def collective_fn():
      for device in ['CPU:0', 'CPU:1']:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              communication_hint=communication)

    def_function.function(collective_fn)()

  def testAbortInstanceParamsResolution(self, collective_op, communication):
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    def collective_fn():
      for device in ['CPU:0', 'CPU:1']:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
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
      collective_op(in_tensor, group_size, group_key, instance_key)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      collective_op(in_tensor, group_size, group_key, instance_key)

    context._reset_context()  # pylint: disable=protected-access
    t.join()
    # Reset the context in order to reset the collective executor.
    _setup_context()

    # After reset non-NCCL collectives should work.
    def_function.function(collective_fn)()

  def testAbortCommunication(self, collective_op, communication):
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    # First perform a normal collective to finish resolution.
    def collective_fn():
      for device in ['CPU:0', 'CPU:1']:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
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
      collective_op(in_tensor, group_size, group_key, instance_key)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      collective_op(in_tensor, group_size, group_key, instance_key)

    # Reset the context in order to reset the collective executor.
    t.join()
    _setup_context()
    def_function.function(collective_fn)()


def _setup_context():
  context._reset_context()
  cpus = config.list_physical_devices('CPU')
  config.set_logical_device_configuration(cpus[0], [
      context.LogicalDeviceConfiguration(),
      context.LogicalDeviceConfiguration(),
      context.LogicalDeviceConfiguration(),
      context.LogicalDeviceConfiguration()
  ])
  context.ensure_initialized()


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
