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
# ==============================================================================
"""Tests for CrossDeviceOps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _get_devices(devices):
  if isinstance(devices, (tuple, list)):
    return tuple(device_util.resolve(d) for d in devices)
  elif isinstance(devices, value_lib.DistributedValues):
    return devices._devices
  elif isinstance(devices, ops.Tensor):
    return (device_util.resolve(devices.device),)
  return (device_util.resolve(devices),)


def _make_per_replica(values, devices, regroup=False):
  devices = _get_devices(devices)
  assert len(values) == len(devices)

  # We simulate the result of regroup called on PerReplica which strips the
  # PerReplica wrapper if it has only one value.
  if len(values) == 1 and regroup:
    with ops.device(devices[0]):
      placed_v = array_ops.identity(values[0])
    return placed_v

  index = []
  for d, v in zip(devices, values):
    with ops.device(d):
      placed_v = array_ops.identity(v)
    index.append(placed_v)
  return value_lib.regroup(index)


# pylint: disable=g-doc-args,g-doc-return-or-yield
def _fake_mirrored(value, devices):
  """Create a faked Mirrored object for testing.

  All components of the returned Mirrored have the same objects, which is not
  true in reality.
  """
  devices = _get_devices(devices)
  values = []
  for d in devices:
    with ops.device(d):
      values.append(array_ops.identity(value))
  return value_lib.regroup(
      values,
      wrap_class=value_lib.Mirrored)


def _make_indexed_slices(values, indices, dense_shape, device):
  with ops.device(device):
    tensor = ops.IndexedSlices(
        values=constant_op.constant(values),
        indices=constant_op.constant(indices),
        dense_shape=constant_op.constant(dense_shape))
  return tensor


def _make_mirrored_indexed_slices(devices, values, indices, dense_shape):
  values = [_make_indexed_slices(values, indices, dense_shape, d)
            for d in devices]
  return value_lib.regroup(
      values,
      wrap_class=value_lib.Mirrored)


_cpu_device = "/device:CPU:0"


class CrossDeviceOpsTestBase(test.TestCase, parameterized.TestCase):

  def _assert_indexed_slices_equal(self, left, right):
    self.assertIsInstance(left, ops.IndexedSlices)
    self.assertIsInstance(right, ops.IndexedSlices)
    self.assertEqual(
        device_util.resolve(left.device), device_util.resolve(right.device))
    self.assertAllEqual(
        self.evaluate(ops.convert_to_tensor(left)),
        self.evaluate(ops.convert_to_tensor(right)))

  def _assert_mirrored_equal(self, left_list, right_list, sess):
    if not isinstance(left_list, list):
      left_list, right_list = [left_list], [right_list]

    for left, right in zip(left_list, right_list):
      self.assertEqual(type(left), type(right))

      # Convert Mirrored to a list since sess.run(Mirrored) only returns one
      # value.
      if isinstance(left, value_lib.Mirrored):
        left, right = left.values, right.values
      else:
        # When there's only one replica Mirrored is automatically unwrapped.
        left, right = [left], [right]

      for left_value, right_value in zip(left, right):
        self.assertEqual(left_value.device, right_value.device)

      # Densify IndexedSlices.
      left = [ops.convert_to_tensor(v) for v in left]
      right = [ops.convert_to_tensor(v) for v in right]
      left, right = sess.run((left, right))
      for left_value, right_value in zip(left, right):
        self.assertAllEqual(left_value, right_value)

  def _testReductionAndBroadcast(self, cross_device_ops, devices):
    if context.num_gpus() < sum(1 for d in devices if "GPU" in d.upper()):
      self.skipTest("Not enough GPUs")

    with self.cached_session() as sess:
      values = [constant_op.constant(float(d)) for d in range(len(devices))]
      per_replica = _make_per_replica(values, devices)
      mean = (len(devices) - 1.) / 2.

      values_2 = [constant_op.constant(d + 1.0) for d in range(len(devices))]
      per_replica_2 = _make_per_replica(values_2, devices)
      mean_2 = mean + 1.

      destination_mirrored = _fake_mirrored(1., devices)
      destination_different = _fake_mirrored(1.,
                                             device_util.resolve(_cpu_device))
      destination_str = device_util.resolve(_cpu_device)

      all_destinations = [
          destination_mirrored,
          destination_different,
          destination_str,
      ]

      # test reduce()
      for destinations in all_destinations:
        self._assert_mirrored_equal(
            cross_device_ops.reduce(
                reduce_util.ReduceOp.MEAN,
                per_replica,
                destinations=destinations), _fake_mirrored(mean, destinations),
            sess)
        self._assert_mirrored_equal(
            cross_device_ops.reduce(
                reduce_util.ReduceOp.MEAN,
                per_replica_2,
                destinations=destinations),
            _fake_mirrored(mean_2, destinations), sess)
        self._assert_mirrored_equal(
            cross_device_ops.reduce(
                reduce_util.ReduceOp.SUM,
                per_replica,
                destinations=destinations),
            _fake_mirrored(mean * len(devices), destinations), sess)
        self._assert_mirrored_equal(
            cross_device_ops.reduce(
                reduce_util.ReduceOp.SUM,
                per_replica_2,
                destinations=destinations),
            _fake_mirrored(mean_2 * len(devices), destinations), sess)

      # test batch_reduce()
      for d1, d2 in itertools.product(all_destinations, all_destinations):
        self._assert_mirrored_equal(
            cross_device_ops.batch_reduce(reduce_util.ReduceOp.MEAN,
                                          [(per_replica, d1),
                                           (per_replica_2, d2)]),
            [_fake_mirrored(mean, d1),
             _fake_mirrored(mean_2, d2)], sess)
        self._assert_mirrored_equal(
            cross_device_ops.batch_reduce(reduce_util.ReduceOp.SUM,
                                          [(per_replica, d1),
                                           (per_replica_2, d2)]),
            [
                _fake_mirrored(mean * len(devices), d1),
                _fake_mirrored(mean_2 * len(devices), d2)
            ], sess)

      # test broadcast()
      for destinations in all_destinations:
        self._assert_mirrored_equal(
            cross_device_ops.broadcast(constant_op.constant(1.), destinations),
            _fake_mirrored(1., destinations), sess)

  def _testIndexedSlicesAllReduce(self, devices, cross_device_ops_instance,
                                  reduce_op, batch_reduce):
    with self.cached_session() as sess:
      dense_shape = [5, 2]
      t0 = _make_indexed_slices([[1., 2.]], [1], dense_shape, devices[0])
      t1 = _make_indexed_slices([[3., 4.], [5., 6.]], [1, 3], dense_shape,
                                devices[1])
      per_replica = value_lib.PerReplica((t0, t1))

      if batch_reduce:
        result = cross_device_ops_instance.batch_reduce(
            reduce_op, [(per_replica, per_replica)])
      else:
        result = cross_device_ops_instance.reduce(reduce_op, per_replica,
                                                  per_replica)

      total_indices_with_dups = [1, 1, 3]
      total_indices_without_dups = [1, 3]

      if reduce_op == reduce_util.ReduceOp.SUM:
        total_values_with_dups = [[1., 2.], [3., 4.], [5., 6.]]
        total_values_without_dups = [[4., 6.], [5., 6.]]
      else:
        assert reduce_op == reduce_util.ReduceOp.MEAN
        total_values_with_dups = [[0.5, 1.], [1.5, 2.], [2.5, 3.]]
        total_values_without_dups = [[2., 3.], [2.5, 3.]]

      total_mirrored_with_dups = _make_mirrored_indexed_slices(
          devices, total_values_with_dups, total_indices_with_dups, dense_shape)
      total_mirrored_without_dups = _make_mirrored_indexed_slices(
          devices, total_values_without_dups, total_indices_without_dups,
          dense_shape)

      # Test that the result is semantically equal to both the concatenated
      # IndexedSlices, as well as when the duplicate indices are summed up.
      if batch_reduce:
        total_mirrored_with_dups = [total_mirrored_with_dups]
        total_mirrored_without_dups = [total_mirrored_without_dups]

      self._assert_mirrored_equal(total_mirrored_with_dups, result, sess)
      self._assert_mirrored_equal(total_mirrored_without_dups, result, sess)


class SingleWorkerCrossDeviceOpsTest(CrossDeviceOpsTestBase):

  reduction_to_one_combinations = combinations.combine(
      cross_device_ops=[
          combinations.NamedObject("DefaultReductionToOneDevice",
                                   cross_device_ops_lib.ReductionToOneDevice()),
          combinations.NamedObject(
              "ReductionToCPUDeviceCrossDeviceOps",
              cross_device_ops_lib.ReductionToOneDevice(
                  reduce_to_device=_cpu_device)),
          combinations.NamedObject(
              "AccumulateNCrossDeviceOp",
              cross_device_ops_lib.ReductionToOneDevice(
                  accumulation_fn=math_ops.add_n)),
      ],
      devices=[
          ["/cpu:0"],
          ["/cpu:0", "/gpu:0"],
          ["/gpu:0", "/gpu:1"],
      ],
      mode=["graph", "eager"])
  allreduce_combinations = combinations.combine(
      cross_device_ops=[
          combinations.NamedObject(
              "AllReduce",
              cross_device_ops_lib.AllReduceCrossDeviceOps("nccl", 1)),
          combinations.NamedObject(
              "AllReduceNoGradientRepacking",
              cross_device_ops_lib.AllReduceCrossDeviceOps("nccl", 0)),
          combinations.NamedObject("NcclAllReduce",
                                   cross_device_ops_lib.NcclAllReduce()),
          combinations.NamedObject(
              "HierarchicalCopy",
              cross_device_ops_lib.HierarchicalCopyAllReduce(8)),
      ],
      devices=[
          ["/gpu:0", "/gpu:1"],
      ],
      mode=["graph", "eager"])

  @combinations.generate(reduction_to_one_combinations + allreduce_combinations)
  def testReductionAndBroadcast(self, cross_device_ops, devices):
    if isinstance(
        cross_device_ops._obj,  # pylint: disable=protected-access
        cross_device_ops_lib.AllReduceCrossDeviceOps
    ) and context.executing_eagerly():
      self.skipTest("b/149881884")
    self._testReductionAndBroadcast(cross_device_ops, devices)

  def testChooseAlgorithm(self):
    # Not use nccl if there is any cpu device.
    self.assertIsInstance(
        cross_device_ops_lib.choose_the_best(["/cpu:0"]),
        cross_device_ops_lib.ReductionToOneDevice)

    # Not use nccl if requested device is not visible to TensorFlow.
    # TODO(yuefengz): make `choose_the_best` work with device strings
    # self.assertIsInstance(
    #     cross_device_ops_lib.choose_the_best(["/gpu:100"]),
    #     cross_device_ops_lib.ReductionToOneDevice)

    if context.num_gpus() < 1:
      return

    devices = ["/gpu:0"]

    def mock_get_registered_kernels_for_op(op):
      if op == "NcclAllReduce":
        return [object]
      else:
        return []

    # Use nccl if nccl kernel is found.
    with test.mock.patch.object(kernels, "get_registered_kernels_for_op",
                                mock_get_registered_kernels_for_op):
      self.assertIsInstance(
          cross_device_ops_lib.choose_the_best(devices),
          cross_device_ops_lib.NcclAllReduce)

    # Not use nccl if nccl kernel is not found.
    with test.mock.patch.object(kernels,
                                "get_registered_kernels_for_op", lambda _: []):
      self.assertIsInstance(
          cross_device_ops_lib.choose_the_best(devices),
          cross_device_ops_lib.ReductionToOneDevice)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      required_gpus=1))
  def testSimpleReduceWithIndexedSlices(self):
    devices = ["/cpu:0", "/gpu:0"]
    t0 = _make_indexed_slices([[1., 2.]], [1], [5, 2], devices[0])
    t1 = _make_indexed_slices([[3., 4.], [5., 6.]], [1, 3], [5, 2], devices[1])
    per_replica = value_lib.PerReplica((t0, t1))
    result = cross_device_ops_lib._simple_reduce(
        per_replica, devices[0], math_ops.add_n, reduce_util.ReduceOp.SUM)

    # Test that the result is semantically equal to both the concatenated
    # IndexedSlices with and without duplicate indices.
    total_with_dups = _make_indexed_slices(
        [[1., 2.], [3., 4.], [5., 6.]], [1, 1, 3], [5, 2], devices[0])
    total_without_dups = _make_indexed_slices(
        [[4., 6.], [5., 6.]], [1, 3], [5, 2], devices[0])
    self._assert_indexed_slices_equal(total_with_dups, result)
    self._assert_indexed_slices_equal(total_without_dups, result)

  @combinations.generate(
      combinations.combine(
          cross_device_ops_instance=[
              combinations.NamedObject(
                  "ReductionToOneDevice",
                  cross_device_ops_lib.ReductionToOneDevice()),
              combinations.NamedObject(
                  "AllReduceCrossDeviceOps",
                  cross_device_ops_lib.AllReduceCrossDeviceOps())
          ],
          reduce_op=[reduce_util.ReduceOp.SUM, reduce_util.ReduceOp.MEAN],
          batch_reduce=[True, False],
          mode=["graph", "eager"],
          required_gpus=1))
  def testIndexedSlicesAllReduce(self, cross_device_ops_instance, reduce_op,
                                 batch_reduce):
    devices = ["/cpu:0", "/gpu:0"]
    self._testIndexedSlicesAllReduce(devices, cross_device_ops_instance,
                                     reduce_op, batch_reduce)


class MultiWorkerCrossDeviceOpsTest(multi_worker_test_base.MultiWorkerTestBase,
                                    CrossDeviceOpsTestBase):

  worker_devices = [
      "/job:worker/replica:0/task:0", "/job:worker/replica:0/task:1"
  ]
  multi_worker_allreduce_combinations = combinations.combine(
      cross_device_ops=[
          combinations.NamedObject(
              "MultiWorkerAllReduce",
              cross_device_ops_lib.MultiWorkerAllReduce(worker_devices, 2,
                                                        ("pscpu/pscpu", 2, -1),
                                                        0)),
          combinations.NamedObject(
              "MultiWorkerAllReducePack",
              cross_device_ops_lib.MultiWorkerAllReduce(worker_devices, 2,
                                                        ("pscpu/pscpu", 2, -1),
                                                        1)),
          combinations.NamedObject(
              "MultiWorkerAllReduceMultipleSpecs",
              cross_device_ops_lib.MultiWorkerAllReduce(
                  worker_devices, 2, [("pscpu/pscpu", 2, 100),
                                      ("xring", 2, -1)], 0)),
      ],
      devices=[
          [
              "/job:worker/replica:0/task:0/device:CPU:0",
              "/job:worker/replica:0/task:1/device:CPU:0"
          ],
          [
              "/job:worker/replica:0/task:0/device:GPU:0",
              "/job:worker/replica:0/task:1/device:GPU:0"
          ],
          [
              "/job:worker/replica:0/task:0/device:GPU:0",
              "/job:worker/replica:0/task:0/device:GPU:1",
              "/job:worker/replica:0/task:1/device:GPU:0",
              "/job:worker/replica:0/task:1/device:GPU:1"
          ],
      ],
      mode=["graph"])

  @combinations.generate(multi_worker_allreduce_combinations)
  def testReductionAndBroadcast(self, cross_device_ops, devices):
    # Mimic the default device of multi-worker strategies.
    with ops.device("/job:worker/replica:0/task:0"):
      self._testReductionAndBroadcast(cross_device_ops, devices)


NUM_WORKERS = 3

CollectiveCommunication = cross_device_ops_lib.CollectiveCommunication


class CollectiveAllReduceTest(multi_worker_test_base.MultiWorkerTestBase,
                              CrossDeviceOpsTestBase):

  collective_key_base = 100000

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 3 workers."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=NUM_WORKERS, num_ps=0)

  def setUp(self):
    super(CollectiveAllReduceTest, self).setUp()
    # Reusing keys is not supported well. So we have to give a different
    # collective key base for different tests.
    CollectiveAllReduceTest.collective_key_base += 100000

  def _get_test_objects(self,
                        task_type,
                        task_id,
                        num_gpus=0,
                        communication=CollectiveCommunication.AUTO,
                        use_strategy_object=False,
                        local_mode=False):
    collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=10 + CollectiveAllReduceTest.collective_key_base,
        op_instance_key_start=100 + CollectiveAllReduceTest.collective_key_base,
        variable_instance_key_start=10000 +
        CollectiveAllReduceTest.collective_key_base)
    if local_mode:
      if num_gpus:
        devices = ["/device:GPU:%d" % i for i in range(num_gpus)]
      else:
        devices = ["/device:CPU:0"]

      if use_strategy_object:
        strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
            communication=communication)
        strategy.extended._collective_keys = collective_keys
        strategy.extended._cross_device_ops._collective_keys = collective_keys
        return strategy, devices, ""
      else:
        collective_all_reduce_ops = cross_device_ops_lib.CollectiveAllReduce(
            1,
            num_gpus,
            collective_keys=collective_keys,
            communication=communication)
        return collective_all_reduce_ops, devices, ""
    else:
      # NCCL requires physical GPUs for every replica, which we can't do with
      # simulated multi host set up now.
      assert communication != CollectiveCommunication.NCCL
      if num_gpus:
        devices = [
            "/job:%s/task:%d/replica:0/device:GPU:%d" % (task_type, task_id, i)
            for i in range(num_gpus)
        ]
      else:
        devices = [
            "/job:%s/task:%d/replica:0/device:CPU:0" % (task_type, task_id)
        ]

      if use_strategy_object:
        strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
            communication=communication)
        strategy.configure(
            cluster_spec=self._cluster_spec,
            task_type=task_type,
            task_id=task_id)
        strategy.extended._collective_keys = collective_keys
        strategy.extended._cross_device_ops._collective_keys = collective_keys
        return (strategy, devices,
                "grpc://" + self._cluster_spec[task_type][task_id])
      else:
        collective_all_reduce_ops = cross_device_ops_lib.CollectiveAllReduce(
            NUM_WORKERS,
            num_gpus,
            collective_keys=collective_keys,
            communication=communication)
        return (collective_all_reduce_ops, devices,
                "grpc://" + self._cluster_spec[task_type][task_id])

  def _test_reduction(self,
                      task_type,
                      task_id,
                      num_gpus,
                      communication,
                      use_strategy_object=False,
                      local_mode=False,
                      hints=None):
    collective_all_reduce, devices, master_target = self._get_test_objects(
        task_type,
        task_id,
        num_gpus,
        communication=communication,
        use_strategy_object=use_strategy_object,
        local_mode=local_mode)
    if local_mode:
      num_workers = 1
      worker_device = None
    else:
      num_workers = len(self._cluster_spec.get("chief", [])) + len(
          self._cluster_spec.get("worker", []))
      worker_device = "/job:%s/task:%d" % (task_type, task_id)

    def _reduce(test_object, reduce_op, per_replica, destinations):
      if use_strategy_object:
        with test_object.scope():
          return test_object.extended.reduce_to(reduce_op, per_replica,
                                                destinations, hints)
      else:
        return test_object.reduce(reduce_op, per_replica, destinations, hints)

    def _batch_reduce(test_object, reduce_op, value_destination_pairs):
      if use_strategy_object:
        with test_object.scope():
          return test_object.extended.batch_reduce_to(reduce_op,
                                                      value_destination_pairs,
                                                      hints)
      else:
        return test_object.batch_reduce(reduce_op, value_destination_pairs,
                                        hints)

    with ops.Graph().as_default(), \
         ops.device(worker_device), \
         self.cached_session(target=master_target) as sess:
      # Collective ops doesn't support scalar tensors, so we have to construct
      # 1-d tensors.
      values = [constant_op.constant([float(d)]) for d in range(len(devices))]
      per_replica = _make_per_replica(values, devices)
      mean = np.array([(len(devices) - 1.) / 2.])

      values_2 = [constant_op.constant([d + 1.0]) for d in range(len(devices))]
      per_replica_2 = _make_per_replica(values_2, devices)
      mean_2 = np.array([mean[0] + 1.])

      destination_mirrored = _fake_mirrored(1., devices)
      destination_different = _fake_mirrored(1., _cpu_device)
      destination_str = _cpu_device

      all_destinations = [
          destination_different, destination_mirrored, destination_str
      ]

      # test reduce()
      for destinations in all_destinations:
        self._assert_mirrored_equal(
            _reduce(
                collective_all_reduce,
                reduce_util.ReduceOp.MEAN,
                per_replica,
                destinations=destinations), _fake_mirrored(mean, destinations),
            sess)
        self._assert_mirrored_equal(
            _reduce(
                collective_all_reduce,
                reduce_util.ReduceOp.MEAN,
                per_replica_2,
                destinations=destinations),
            _fake_mirrored(mean_2, destinations), sess)
        self._assert_mirrored_equal(
            _reduce(
                collective_all_reduce,
                reduce_util.ReduceOp.SUM,
                per_replica,
                destinations=destinations),
            _fake_mirrored(mean * len(devices) * num_workers, destinations),
            sess)
        self._assert_mirrored_equal(
            _reduce(
                collective_all_reduce,
                reduce_util.ReduceOp.SUM,
                per_replica_2,
                destinations=destinations),
            _fake_mirrored(mean_2 * len(devices) * num_workers, destinations),
            sess)

      # test batch_reduce()
      for d1, d2 in itertools.product(all_destinations, all_destinations):
        self._assert_mirrored_equal(
            _batch_reduce(collective_all_reduce, reduce_util.ReduceOp.MEAN,
                          [(per_replica, d1), (per_replica_2, d2)]),
            [_fake_mirrored(mean, d1),
             _fake_mirrored(mean_2, d2)], sess)
        self._assert_mirrored_equal(
            _batch_reduce(collective_all_reduce, reduce_util.ReduceOp.SUM,
                          [(per_replica, d1), (per_replica_2, d2)]),
            [
                _fake_mirrored(mean * len(devices) * num_workers, d1),
                _fake_mirrored(mean_2 * len(devices) * num_workers, d2)
            ], sess)

  def _get_indexed_slices(self,
                          devices,
                          start_i,
                          variable_length,
                          as_per_replica=True):
    dense_shape = [10, 2]
    values = ([[1., 2.]], [[3., 4.]], [[2., 1.]], [[0., 0.]], [[3., 1.]],
              [[2., 1.]])
    indices = ([1], [2], [3], [4], [5], [6])

    # values and indices that have variable lengths.
    vl_values = ([[1., 2.], [3., 4.]], [[3., 4.]], [[2., 1.]], [[0., 0.]],
                 [[3., 1.], [2., 1.]], [[2., 1.]])
    vl_indices = ([1, 2], [2], [3], [4], [5, 6], [6])

    indexed_slices = []
    for i, d in enumerate(devices):
      idx = i + start_i
      indexed_slices.append(
          _make_indexed_slices(
              vl_values[idx] if variable_length else values[idx],
              vl_indices[idx] if variable_length else indices[idx], dense_shape,
              d))
    if as_per_replica:
      per_replica = value_lib.PerReplica(indexed_slices)
      return per_replica
    else:
      return indexed_slices

  def _test_reduce_indexed_slices(self,
                                  task_type,
                                  task_id,
                                  num_gpus,
                                  communication,
                                  batch_reduce,
                                  variable_length,
                                  local_mode=False):
    collective_all_reduce, devices, master_target = self._get_test_objects(
        task_type,
        task_id,
        num_gpus,
        communication=communication,
        local_mode=local_mode)
    if local_mode:
      num_workers = 1
      worker_device = None
    else:
      num_workers = len(self._cluster_spec.get("chief", [])) + len(
          self._cluster_spec.get("worker", []))
      worker_device = "/job:%s/task:%d" % (task_type, task_id)
    with ops.Graph().as_default(), \
         ops.device(worker_device), \
         self.cached_session(target=master_target) as sess:
      per_replica = self._get_indexed_slices(devices,
                                             (task_id or 0) * max(num_gpus, 1),
                                             variable_length)

      if batch_reduce:
        result = collective_all_reduce.batch_reduce(
            reduce_util.ReduceOp.SUM, [(per_replica, per_replica)])[0]
      else:
        result = collective_all_reduce.reduce(reduce_util.ReduceOp.SUM,
                                              per_replica, per_replica)
      if num_gpus > 1:
        self.assertIsInstance(result, value_lib.Mirrored)

      run_options = config_pb2.RunOptions()
      run_options.experimental.collective_graph_key = 7
      if num_gpus > 1:
        result = sess.run([ops.convert_to_tensor(v) for v in result.values],
                          options=run_options)[0]
      else:
        result = sess.run(ops.convert_to_tensor(result), options=run_options)

      # Reduce the same indexed slices on CPU locally as our expected results.
      devices_cpu = [(worker_device or "") + "/device:CPU:0"] * (
          max(num_gpus, 1) * num_workers)
      per_replica_on_cpu = self._get_indexed_slices(
          devices_cpu, 0, variable_length, as_per_replica=False)
      expected_result = cross_device_utils.aggregate_tensors_or_indexed_slices(
          per_replica_on_cpu)
      expected_result = sess.run(ops.convert_to_tensor(expected_result))

      self.assertAllEqual(expected_result, result)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          required_gpus=[0, 1, 2],
          use_strategy_object=[True, False],
          bytes_per_pack=[0, 1, 4]))
  def testReductionDistributed(self, required_gpus, use_strategy_object,
                               bytes_per_pack):
    hints = collective_util.Hints(bytes_per_pack=bytes_per_pack)
    self._run_between_graph_clients(
        self._test_reduction,
        self._cluster_spec,
        required_gpus,
        communication=CollectiveCommunication.RING,
        use_strategy_object=use_strategy_object,
        hints=hints)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          required_gpus=[0, 1, 2],
          variable_length=[True, False]))
  def testReduceIndexedSlicesDistributed(self, required_gpus, variable_length):
    self._run_between_graph_clients(
        self._test_reduce_indexed_slices,
        self._cluster_spec,
        required_gpus,
        communication=CollectiveCommunication.RING,
        batch_reduce=True,
        variable_length=variable_length)

  # Collective ops doesn't support strategy with one device.
  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          required_gpus=2,
          communication=[
              CollectiveCommunication.NCCL, CollectiveCommunication.RING
          ],
          use_strategy_object=[True, False]))
  def testReductionLocal(self, required_gpus, communication,
                         use_strategy_object):
    self._test_reduction(
        None,
        None,
        required_gpus,
        communication=communication,
        use_strategy_object=use_strategy_object,
        local_mode=True)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          required_gpus=2,
          batch_reduce=[True, False],
          variable_length=[True, False],
          communication=[
              CollectiveCommunication.NCCL, CollectiveCommunication.RING
          ]))
  def testReduceIndexedSlicesLocal(self, required_gpus, batch_reduce,
                                   variable_length, communication):
    self._test_reduce_indexed_slices(
        None,
        None,
        required_gpus,
        communication=communication,
        batch_reduce=batch_reduce,
        variable_length=variable_length,
        local_mode=True)


if __name__ == "__main__":
  test.main()
