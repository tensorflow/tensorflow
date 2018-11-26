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

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _make_per_replica(values, devices, regroup=False):
  devices = cross_device_ops_lib.get_devices_from(devices)
  assert len(values) == len(devices)

  # We simulate the result of regroup called on PerReplica which strips the
  # PerReplica wrapper if it has only one value.
  if len(values) == 1 and regroup:
    with ops.device(devices[0]):
      placed_v = array_ops.identity(values[0])
    return placed_v

  index = {}
  for d, v in zip(devices, values):
    with ops.device(d):
      placed_v = array_ops.identity(v)
    index[d] = placed_v
  return value_lib.PerReplica(index)


# pylint: disable=g-doc-args,g-doc-return-or-yield
def _fake_mirrored(value, devices):
  """Create a faked Mirrored object for testing.

  All components of the returned Mirrored have the same objects, which is not
  true in reality.
  """
  devices = cross_device_ops_lib.get_devices_from(devices)
  return value_lib.Mirrored(
      {d: v for d, v in zip(devices, [value] * len(devices))})


def _make_indexed_slices(values, indices, dense_shape, device):
  with ops.device(device):
    tensor = ops.IndexedSlices(
        values=constant_op.constant(values),
        indices=constant_op.constant(indices),
        dense_shape=constant_op.constant(dense_shape))
  return tensor


def _make_mirrored_indexed_slices(devices, values, indices, dense_shape):
  return value_lib.Mirrored({
      d: _make_indexed_slices(values, indices, dense_shape, d) for d in devices
  })


_cpu_device = "/device:CPU:0"


class CrossDeviceOpsTestBase(test.TestCase, parameterized.TestCase):

  def _assert_indexed_slices_equal(self, left, right):
    self.assertIsInstance(left, ops.IndexedSlices)
    self.assertIsInstance(right, ops.IndexedSlices)
    self.assertEqual(device_util.resolve(left.device),
                     device_util.resolve(right.device))
    self.assertAllEqual(
        self.evaluate(ops.convert_to_tensor(left)),
        self.evaluate(ops.convert_to_tensor(right)))

  def _assert_values_equal(self, left, right):
    if isinstance(left, list):
      for l, r in zip(left, right):
        self._assert_values_equal(l, r)
    else:
      self.assertEqual(type(left), type(right))
      self.assertEqual(set(left.devices), set(right.devices))
      if isinstance(list(left._index.values())[0], ops.IndexedSlices):
        for (d, v) in left._index.items():
          self._assert_indexed_slices_equal(v, right._index[d])
      elif context.executing_eagerly():
        self.assertEqual([v.numpy() for v in left._index.values()],
                         list(right._index.values()))
      else:
        with self.cached_session() as sess:
          self.assertEqual(
              sess.run(list(left._index.values())), list(right._index.values()))

  def _testReductionAndBroadcast(self, cross_device_ops, distribution):
    devices = distribution.extended.worker_devices

    values = [constant_op.constant(float(d)) for d in range(len(devices))]
    per_replica = _make_per_replica(values, devices)
    mean = (len(devices) - 1.) / 2.

    values_2 = [constant_op.constant(d + 1.0) for d in range(len(devices))]
    per_replica_2 = _make_per_replica(values_2, devices)
    mean_2 = mean + 1.

    destination_mirrored = _fake_mirrored(1., devices)
    destination_different = _fake_mirrored(1., _cpu_device)
    destination_str = _cpu_device
    destination_list = devices

    all_destinations = [
        destination_mirrored, destination_different, destination_str,
        destination_list
    ]

    # test reduce()
    for destinations in all_destinations:
      self._assert_values_equal(
          cross_device_ops.reduce(
              reduce_util.ReduceOp.MEAN,
              per_replica,
              destinations=destinations),
          _fake_mirrored(mean, destinations))
      self._assert_values_equal(
          cross_device_ops.reduce(
              reduce_util.ReduceOp.MEAN,
              per_replica_2,
              destinations=destinations),
          _fake_mirrored(mean_2, destinations))
      self._assert_values_equal(
          cross_device_ops.reduce(
              reduce_util.ReduceOp.SUM, per_replica,
              destinations=destinations),
          _fake_mirrored(mean * len(devices), destinations))
      self._assert_values_equal(
          cross_device_ops.reduce(
              reduce_util.ReduceOp.SUM,
              per_replica_2,
              destinations=destinations),
          _fake_mirrored(mean_2 * len(devices), destinations))

    # test batch_reduce()
    for d1, d2 in itertools.product(all_destinations, all_destinations):
      self._assert_values_equal(
          cross_device_ops.batch_reduce(
              reduce_util.ReduceOp.MEAN,
              [(per_replica, d1), (per_replica_2, d2)]),
          [
              _fake_mirrored(mean, d1),
              _fake_mirrored(mean_2, d2)
          ])
      self._assert_values_equal(
          cross_device_ops.batch_reduce(
              reduce_util.ReduceOp.SUM,
              [(per_replica, d1), (per_replica_2, d2)]),
          [
              _fake_mirrored(mean * len(devices), d1),
              _fake_mirrored(mean_2 * len(devices), d2)
          ])

    # test broadcast()
    for destinations in all_destinations:
      self._assert_values_equal(
          cross_device_ops.broadcast(constant_op.constant(1.), destinations),
          _fake_mirrored(1., destinations))


class SingleWorkerCrossDeviceOpsTest(CrossDeviceOpsTestBase):
  # TODO(yuefengz): decouple the num_gpus check from distribution in
  # combinations module so that we can pass in devices instead of a distribution
  # strategy.
  reduction_to_one_combinations = combinations.combine(
      cross_device_ops=[
          combinations.NamedObject(
              "DefaultReductionToOneDeviceCrossDeviceOps",
              cross_device_ops_lib.ReductionToOneDeviceCrossDeviceOps()),
          combinations.NamedObject(
              "ReductionToCPUDeviceCrossDeviceOps",
              cross_device_ops_lib.ReductionToOneDeviceCrossDeviceOps(
                  reduce_to_device=_cpu_device)),
          combinations.NamedObject(
              "AccumulateNCrossDeviceOp",
              cross_device_ops_lib.ReductionToOneDeviceCrossDeviceOps(
                  accumulation_fn=math_ops.accumulate_n)),
      ],
      distribution=[
          combinations.one_device_strategy,
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.mirrored_strategy_with_two_gpus,
          combinations.core_mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_two_gpus
      ],
      mode=["graph", "eager"])
  allreduce_combinations = combinations.combine(
      cross_device_ops=[
          combinations.NamedObject(
              "AllReduce",
              cross_device_ops_lib.AllReduceCrossDeviceOps("nccl", 1, 0, 0)),
          combinations.NamedObject(
              "HierarchicalCopy",
              cross_device_ops_lib.AllReduceCrossDeviceOps(
                  "hierarchical_copy", 8, 0, 0)),
          combinations.NamedObject(
              "AllReduceNoGradientRepacking",
              cross_device_ops_lib.AllReduceCrossDeviceOps("nccl", 0, 0, 0)),
          combinations.NamedObject(
              "HierarchicalCopyAggregateSmallTensors",
              cross_device_ops_lib.AllReduceCrossDeviceOps(
                  "hierarchical_copy", 0, 100, 10))
      ],
      distribution=[combinations.mirrored_strategy_with_two_gpus,
                    combinations.core_mirrored_strategy_with_two_gpus],
      mode=["graph", "eager"])

  @combinations.generate(reduction_to_one_combinations + allreduce_combinations)
  def testReductionAndBroadcast(self, cross_device_ops, distribution):
    with distribution.scope():
      self._testReductionAndBroadcast(cross_device_ops, distribution)

  def testChooseAlgorithm(self):
    device_links = [[1, 2, 3, 4], [0, 2, 3, 5], [0, 1, 3, 6], [0, 1, 2, 7],
                    [0, 5, 6, 7], [1, 4, 6, 7], [2, 4, 5, 7], [3, 4, 5, 6]]
    result = cross_device_ops_lib._choose_all_reduce_algorithm(device_links)
    self.assertIsInstance(result, cross_device_ops_lib.AllReduceCrossDeviceOps)
    self.assertEqual(result._all_reduce_alg, "hierarchical_copy")
    self.assertEqual(result._num_packs, 8)

    # if there are only 4 devices
    device_links = [[1, 2, 3, 4], [0, 2, 3, 5], [0, 1, 3, 6], [0, 1, 2, 7]]
    result = cross_device_ops_lib._choose_all_reduce_algorithm(device_links)
    self.assertIsInstance(result, cross_device_ops_lib.AllReduceCrossDeviceOps)
    self.assertEqual(result._all_reduce_alg, "nccl")
    self.assertEqual(result._num_packs, 1)

    # if devices links contain each device itself
    device_links = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6],
                    [0, 1, 2, 3, 7], [0, 4, 5, 6, 7], [1, 4, 5, 6, 7],
                    [2, 4, 5, 6, 7], [3, 4, 5, 6, 7]]
    result = cross_device_ops_lib._choose_all_reduce_algorithm(device_links)
    self.assertIsInstance(result, cross_device_ops_lib.AllReduceCrossDeviceOps)
    self.assertEqual(result._all_reduce_alg, "hierarchical_copy")
    self.assertEqual(result._num_packs, 8)

    # if not dgx1-like links
    device_links = [[0, 2, 3, 5], [0, 1, 3, 6], [0, 1, 2, 7], [0, 5, 6, 7],
                    [1, 4, 6, 7], [2, 4, 5, 7], [3, 4, 5, 6], [1, 2, 3, 4]]
    result = cross_device_ops_lib._choose_all_reduce_algorithm(device_links)
    self.assertIsInstance(result, cross_device_ops_lib.AllReduceCrossDeviceOps)
    self.assertEqual(result._all_reduce_alg, "nccl")
    self.assertEqual(result._num_packs, 1)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      required_gpus=1))
  def testSimpleReduceWithIndexedSlices(self):
    devices = ["/cpu:0", "/gpu:0"]
    t0 = _make_indexed_slices([[1., 2.]], [1], [5, 2], devices[0])
    t1 = _make_indexed_slices([[3., 4.], [5., 6.]], [1, 3], [5, 2], devices[1])
    per_replica = value_lib.PerReplica({devices[0]: t0, devices[1]: t1})
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
                  "ReductionToOneDeviceCrossDeviceOps",
                  cross_device_ops_lib.ReductionToOneDeviceCrossDeviceOps()),
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
    dense_shape = [5, 2]
    t0 = _make_indexed_slices([[1., 2.]], [1], dense_shape, devices[0])
    t1 = _make_indexed_slices(
        [[3., 4.], [5., 6.]], [1, 3], dense_shape, devices[1])
    per_replica = value_lib.PerReplica({devices[0]: t0, devices[1]: t1})

    if batch_reduce:
      result = cross_device_ops_instance.batch_reduce(
          reduce_op, [(per_replica, devices)])
    else:
      result = cross_device_ops_instance.reduce(
          reduce_op, per_replica, devices)

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

    self._assert_values_equal(total_mirrored_with_dups, result)
    self._assert_values_equal(total_mirrored_without_dups, result)


class MultiWorkerCrossDeviceOpsTest(multi_worker_test_base.MultiWorkerTestBase,
                                    CrossDeviceOpsTestBase):

  worker_devices = [
      "/job:worker/replica:0/task:0", "/job:worker/replica:0/task:1"
  ]
  multi_worker_allreduce_combinations = combinations.combine(
      cross_device_ops=[
          combinations.NamedObject(
              "MultiWorkerAllReduce",
              cross_device_ops_lib.MultiWorkerAllReduce(
                  worker_devices, 2, ("pscpu/pscpu", 2, -1), 0, 0, 0)),
          combinations.NamedObject(
              "MultiWorkerAllReducePack",
              cross_device_ops_lib.MultiWorkerAllReduce(
                  worker_devices, 2, ("pscpu/pscpu", 2, -1), 1, 0, 0)),
          combinations.NamedObject(
              "MultiWorkerAllReduceAggregation",
              cross_device_ops_lib.MultiWorkerAllReduce(
                  worker_devices, 2, ("pscpu/pscpu", 2, -1), 0, 100, 10)),
          combinations.NamedObject(
              "MultiWorkerAllReduceMultipleSpecs",
              cross_device_ops_lib.MultiWorkerAllReduce(
                  worker_devices, 2, [("pscpu/pscpu", 2, 100),
                                      ("xring", 2, -1)], 0, 0, 0)),
      ],
      distribution=[
          combinations.NamedDistribution(
              "MirroredCPU",
              lambda: mirrored_strategy.MirroredStrategy(num_gpus_per_worker=0),
              required_gpus=0),
          combinations.NamedDistribution(
              "Mirrored1GPU",
              lambda: mirrored_strategy.MirroredStrategy(num_gpus_per_worker=1),
              required_gpus=1),
          combinations.NamedDistribution(
              "Mirrored2GPUs",
              lambda: mirrored_strategy.MirroredStrategy(num_gpus_per_worker=2),
              required_gpus=2),
          # pylint: disable=g-long-lambda
          combinations.NamedDistribution(
              "CoreMirroredCPU",
              lambda: mirrored_strategy.CoreMirroredStrategy(
                  num_gpus_per_worker=0),
              required_gpus=0),
          combinations.NamedDistribution(
              "CoreMirrored1GPU",
              lambda: mirrored_strategy.CoreMirroredStrategy(
                  num_gpus_per_worker=1),
              required_gpus=1),
          combinations.NamedDistribution(
              "CoreMirrored2GPUs",
              lambda: mirrored_strategy.CoreMirroredStrategy(
                  num_gpus_per_worker=2),
              required_gpus=2),
      ],
      mode=["graph"])

  @combinations.generate(multi_worker_allreduce_combinations)
  def testReductionAndBroadcast(self, cross_device_ops, distribution):
    distribution.configure(cluster_spec={
        "worker":
            ["/job:worker/replica:0/task:0", "/job:worker/replica:0/task:1"]
    })
    with distribution.scope():
      self._testReductionAndBroadcast(cross_device_ops, distribution)


class MultiWorkerCollectiveAllReduceTest(
    multi_worker_test_base.MultiWorkerTestBase, parameterized.TestCase):

  collective_key_base = 100000

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0)

  def setUp(self):
    super(MultiWorkerCollectiveAllReduceTest, self).setUp()
    # Reusing keys are not supported well. So we have to give a different
    # collective key base for different tests.
    MultiWorkerCollectiveAllReduceTest.collective_key_base += 100000

  def _get_test_objects(self, task_type, task_id, num_gpus=0, local_mode=False):
    collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=10 * num_gpus +
        MultiWorkerCollectiveAllReduceTest.collective_key_base,
        instance_key_start=num_gpus * 100 +
        MultiWorkerCollectiveAllReduceTest.collective_key_base,
        instance_key_with_id_start=num_gpus * 10000 +
        MultiWorkerCollectiveAllReduceTest.collective_key_base)
    if local_mode:
      collective_all_reduce_ops = cross_device_ops_lib.CollectiveAllReduce(
          1, num_gpus, collective_keys=collective_keys)
      if num_gpus:
        devices = ["/device:GPU:%d" % i for i in range(num_gpus)]
      else:
        devices = ["/device:CPU:0"]
      return collective_all_reduce_ops, devices, ""
    else:
      collective_all_reduce_ops = cross_device_ops_lib.CollectiveAllReduce(
          3, num_gpus, collective_keys=collective_keys)
      if num_gpus:
        devices = [
            "/job:%s/task:%d/device:GPU:%d" % (task_type, task_id, i)
            for i in range(num_gpus)
        ]
      else:
        devices = ["/job:%s/task:%d" % (task_type, task_id)]
      return (collective_all_reduce_ops, devices,
              "grpc://" + self._cluster_spec[task_type][task_id])

  def _assert_values_equal(self, left, right, sess):
    if isinstance(left, list):
      for l, r in zip(left, right):
        self._assert_values_equal(l, r, sess)
    else:
      self.assertEqual(type(left), type(right))
      self.assertEqual(set(left.devices), set(right.devices))

      run_options = config_pb2.RunOptions()
      run_options.experimental.collective_graph_key = 6

      left_values = np.array(
          sess.run(list(left._index.values()), options=run_options)).flatten()
      right_values = np.array(list(right._index.values())).flatten()
      self.assertEqual(len(left_values), len(right_values))
      for l, r in zip(left_values, right_values):
        self.assertEqual(l, r)

  def _test_reduction(self, task_type, task_id, num_gpus, local_mode=False):
    collective_all_reduce, devices, master_target = self._get_test_objects(
        task_type, task_id, num_gpus, local_mode=local_mode)
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
      # Collective ops doesn't support scalar tensors, so we have to construct
      # 1-d tensors.
      values = [constant_op.constant([float(d)]) for d in range(len(devices))]
      per_replica = _make_per_replica(values, devices, regroup=True)
      mean = np.array([(len(devices) - 1.) / 2.])

      values_2 = [constant_op.constant([d + 1.0]) for d in range(len(devices))]
      per_replica_2 = _make_per_replica(values_2, devices)
      mean_2 = np.array([mean[0] + 1.])

      destination_mirrored = _fake_mirrored(1., devices)
      destination_different = _fake_mirrored(1., _cpu_device)
      destination_str = _cpu_device
      destination_list = devices

      all_destinations = [
          destination_different, destination_mirrored, destination_str,
          destination_list
      ]

      # test reduce()
      for destinations in all_destinations:
        self._assert_values_equal(
            collective_all_reduce.reduce(
                reduce_util.ReduceOp.MEAN,
                per_replica,
                destinations=destinations),
            _fake_mirrored(mean, destinations), sess)
        self._assert_values_equal(
            collective_all_reduce.reduce(
                reduce_util.ReduceOp.MEAN,
                per_replica_2,
                destinations=destinations),
            _fake_mirrored(mean_2, destinations), sess)
        self._assert_values_equal(
            collective_all_reduce.reduce(
                reduce_util.ReduceOp.SUM,
                per_replica,
                destinations=destinations),
            _fake_mirrored(mean * len(devices) * num_workers, destinations),
            sess)
        self._assert_values_equal(
            collective_all_reduce.reduce(
                reduce_util.ReduceOp.SUM,
                per_replica_2,
                destinations=destinations),
            _fake_mirrored(mean_2 * len(devices) * num_workers, destinations),
            sess)

      # test batch_reduce()
      for d1, d2 in itertools.product(all_destinations, all_destinations):
        self._assert_values_equal(
            collective_all_reduce.batch_reduce(reduce_util.ReduceOp.MEAN,
                                               [(per_replica, d1),
                                                (per_replica_2, d2)]),
            [
                _fake_mirrored(mean, d1),
                _fake_mirrored(mean_2, d2)
            ], sess)
        self._assert_values_equal(
            collective_all_reduce.batch_reduce(reduce_util.ReduceOp.SUM,
                                               [(per_replica, d1),
                                                (per_replica_2, d2)]),
            [
                _fake_mirrored(mean * len(devices) * num_workers, d1),
                _fake_mirrored(mean_2 * len(devices) * num_workers, d2)
            ], sess)

    return True

  @combinations.generate(
      combinations.combine(mode=["graph"], num_gpus=[0, 1, 2], required_gpus=1))
  def testReductionDistributed(self, num_gpus):
    if context.num_gpus() < num_gpus:
      return
    self._run_between_graph_clients(self._test_reduction, self._cluster_spec,
                                    num_gpus)

  # Collective ops doesn't support strategy with one device.
  def testReductionLocal(self, num_gpus=2):
    if context.num_gpus() < num_gpus:
      return
    self._test_reduction(None, None, num_gpus, local_mode=True)


if __name__ == "__main__":
  test.main()
