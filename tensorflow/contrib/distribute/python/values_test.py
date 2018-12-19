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
"""Tests for the distributed values library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import nest


class DistributedValuesTest(test.TestCase):

  def testGetEager(self):
    with ops.device("/device:CPU:0"):
      one = constant_op.constant(1)
      two = constant_op.constant(2)
      device_map = values.ReplicaDeviceMap(("/device:CPU:0", "/device:GPU:0"))
      v = values.DistributedValues(device_map, (one, two))
      self.assertEqual(two, v.get("/device:GPU:0"))
      self.assertEqual(one, v.get())
      with self.assertRaises(ValueError):
        self.assertIsNone(v.get("/device:GPU:2"))

  def testGetGraph(self):
    with context.graph_mode(), \
        ops.Graph().as_default(), \
        ops.device("/device:CPU:0"):
      one = constant_op.constant(1)
      two = constant_op.constant(2)
      device_map = values.ReplicaDeviceMap(("/device:CPU:0", "/device:GPU:0"))
      v = values.DistributedValues(device_map, (one, two))
      self.assertEqual(two, v.get("/device:GPU:0"))
      self.assertEqual(one, v.get())
      with self.assertRaises(ValueError):
        self.assertIsNone(v.get("/device:GPU:2"))

  def testCanonicalization(self):
    canonical_cpu = ("/job:localhost/replica:0/task:0/device:CPU:0",)
    v = values.DistributedValues(values.SingleDeviceMap(""), (42,))
    self.assertEqual(canonical_cpu, v.devices)
    v = values.DistributedValues(values.SingleDeviceMap("/device:CPU:0"), (42,))
    self.assertEqual(canonical_cpu, v.devices)
    v = values.DistributedValues(values.SingleDeviceMap("/cpu:0"), (42,))
    self.assertEqual(canonical_cpu, v.devices)
    v = values.DistributedValues(values.SingleDeviceMap("/CPU:0"), (42,))
    self.assertEqual(canonical_cpu, v.devices)
    with self.assertRaises(AssertionError):
      v = values.DistributedValues(
          values.SingleDeviceMap("/device:cpu:0"), (42,))

  def testIsTensorLike(self):
    with context.graph_mode(), \
         ops.Graph().as_default(), \
         ops.device("/device:CPU:0"):
      one = constant_op.constant(1)
      two = constant_op.constant(2)
      device_map = values.ReplicaDeviceMap(("/device:CPU:0", "/device:GPU:0"))
      v = values.DistributedValues(device_map, (one, two))
      self.assertEqual(two, v.get("/device:GPU:0"))
      self.assertEqual(one, v.get())
      self.assertTrue(v.is_tensor_like)
      self.assertTrue(tensor_util.is_tensor(v))

  def testIsTensorLikeWithAConstant(self):
    with context.graph_mode(), \
         ops.Graph().as_default(), \
         ops.device("/device:CPU:0"):
      one = constant_op.constant(1)
      two = 2.0
      device_map = values.ReplicaDeviceMap(("/device:CPU:0", "/device:GPU:0"))
      v = values.DistributedValues(device_map, (one, two))
      self.assertEqual(two, v.get("/device:GPU:0"))
      self.assertEqual(one, v.get())
      self.assertFalse(v.is_tensor_like)
      self.assertFalse(tensor_util.is_tensor(v))


class DistributedDelegateTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testGetAttr(self):
    with ops.device("/device:CPU:0"):

      class Foo(object):

        def __init__(self, x):
          self.x = x

      device_map = values.ReplicaDeviceMap(("/device:CPU:0", "/device:GPU:0"))
      v = values.DistributedDelegate(device_map, (Foo(7), Foo(8)))
      self.assertEqual(7, v.x)
      with self.assertRaises(AttributeError):
        _ = v.y

  @test_util.run_in_graph_and_eager_modes
  def testOperatorOverride(self):
    with ops.device("/device:CPU:0"):
      device_map = values.ReplicaDeviceMap(("/device:CPU:0", "/device:GPU:0"))
      v = values.DistributedDelegate(device_map, (7, 8))
      # v should act like int(7).
      self.assertEqual(8, v + 1)
      self.assertEqual(10, 3 + v)
      self.assertEqual(14, v + v)
      self.assertEqual(5, v - 2)
      self.assertEqual(6, 13 - v)
      self.assertEqual(0, v - v)
      self.assertEqual(14, v * 2)
      self.assertEqual(21, 3 * v)
      self.assertEqual(49, v * v)
      self.assertEqual(3.5, v / 2)
      self.assertEqual(1.5, 10.5 / v)
      self.assertEqual(3, v // 2)
      self.assertEqual(2, 15 // v)
      self.assertEqual(1, v % 2)
      self.assertEqual(2, 16 % v)
      self.assertTrue(v < 12)
      self.assertTrue(v <= 12)
      self.assertFalse(v > 12)
      self.assertFalse(v >= 12)
      self.assertFalse(12 < v)
      self.assertFalse(12 <= v)
      self.assertTrue(12 > v)
      self.assertTrue(12 >= v)
      self.assertEqual(3, v & 3)
      self.assertEqual(3, 11 & v)
      self.assertEqual(15, v | 8)
      self.assertEqual(23, 16 | v)
      self.assertEqual(4, v ^ 3)
      self.assertEqual(12, 11 ^ v)
      self.assertEqual(343, pow(v, 3))
      self.assertEqual(3, pow(v, 3, 10))
      self.assertEqual(128, pow(2, v))
      self.assertEqual(-7, -v)
      self.assertEqual(~7, ~v)
      self.assertEqual(7, abs(v))
      with self.assertRaises(TypeError):
        _ = v[2]


def _device_str(d):
  return "/device:GPU:" + str(d)


def _nested_value(d):
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


def _make_mirrored():
  v = []
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))
  device_map = values.ReplicaDeviceMap(devices)
  mirrored = values.MirroredVariable(device_map, v,
                                     variable_scope.VariableAggregation.SUM)
  return v, device_map, mirrored


class RegroupAndSelectDeviceTest(test.TestCase):

  def _is_per_replica(self, result, expected, klass=values.PerReplica):
    self.assertIsInstance(result, klass)
    # We canonicalize the devices to match the device strings returned
    # by PerReplica, which also does device string canonicalization.
    devices = [device_util.canonicalize(_device_str(i))
               for i in range(len(expected))]
    self.assertEqual(set(devices), set(result.devices))
    for i, d in enumerate(devices):
      self.assertEqual(expected[i], result.get(d))
      self.assertEqual(expected[i], result.get(_device_str(i)))

  def testNested(self):
    device_map = values.ReplicaDeviceMap((_device_str(0), _device_str(1)))
    result = values.regroup(device_map,
                            (_nested_value("1"), _nested_value("2")))
    self.assertIsInstance(result, tuple)
    self.assertEqual(3, len(result))
    self._is_per_replica(result[0], ["a1", "a2"])
    self._is_per_replica(result[2], ["h1", "h2"])

    self.assertIsInstance(result[1], list)
    self.assertEqual(3, len(result[1]))
    self._is_per_replica(result[1][0], ["b1", "b2"])
    self._is_per_replica(result[1][2], ["g1", "g2"])

    self.assertIsInstance(result[1][1], dict)
    self.assertEqual(set(["c", "e"]), set(result[1][1].keys()))
    self._is_per_replica(result[1][1]["c"], ["d1", "d2"])
    self._is_per_replica(result[1][1]["e"], ["f1", "f2"])

    # Also test that we can undo the merge using select_replica()
    self.assertEqual(_nested_value("1"),
                     values.select_replica(0, result))
    self.assertEqual(_nested_value("2"),
                     values.select_replica(1, result))
    # select_device_mirrored() should fail due to non-mirrored values
    with self.assertRaises(TypeError):
      values.select_device_mirrored(_device_str(0), result)
    with self.assertRaises(TypeError):
      values.select_device_mirrored(_device_str(1), result)

  def testWrapClass(self):
    # Normally a mirrored value would be the same across devices, but
    # for a test it is convenient to be able to tell the values apart.
    device_map = values.ReplicaDeviceMap((_device_str(0), _device_str(1)))
    result = values.regroup(device_map,
                            (_nested_value("1"), _nested_value("2")),
                            values.Mirrored)
    self.assertIsInstance(result, tuple)
    self.assertEqual(3, len(result))
    self._is_per_replica(result[0], ["a1", "a2"], values.Mirrored)
    self._is_per_replica(result[2], ["h1", "h2"], values.Mirrored)

    self.assertIsInstance(result[1], list)
    self.assertEqual(3, len(result[1]))
    self._is_per_replica(result[1][0], ["b1", "b2"], values.Mirrored)
    self._is_per_replica(result[1][2], ["g1", "g2"], values.Mirrored)

    self.assertIsInstance(result[1][1], dict)
    self.assertEqual(set(["c", "e"]), set(result[1][1].keys()))
    self._is_per_replica(result[1][1]["c"], ["d1", "d2"], values.Mirrored)
    self._is_per_replica(result[1][1]["e"], ["f1", "f2"], values.Mirrored)

    # Also test that we can undo the merge using select_replica()
    self.assertEqual(_nested_value("1"),
                     values.select_replica(0, result))
    self.assertEqual(_nested_value("2"),
                     values.select_replica(1, result))
    # Values are marked as mirrored, so select_device_mirrored() is allowed.
    self.assertEqual(_nested_value("1"),
                     values.select_device_mirrored(_device_str(0), result))
    self.assertEqual(_nested_value("2"),
                     values.select_device_mirrored(_device_str(1), result))

  def testMirroredContainer(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")
    v, device_map, mirrored = _make_mirrored()
    result = values.regroup(device_map, v)
    self.assertIs(mirrored, result)

  def testSameId(self):
    foo = object()
    device_map = values.ReplicaDeviceMap((_device_str(0), _device_str(1)))
    result = values.regroup(device_map, (("a", foo), ("b", foo)))
    self.assertIsInstance(result, tuple)
    self.assertEqual(2, len(result))
    self._is_per_replica(result[0], ["a", "b"])
    self.assertIs(foo, result[1])

    # Test select_replica(), should undo the merge done by regroup().
    result_0 = values.select_replica(0, result)
    self.assertIsInstance(result_0, tuple)
    self.assertEqual(2, len(result_0))
    self.assertEqual("a", result_0[0])
    self.assertIs(foo, result_0[1])
    result_1 = values.select_replica(1, result)
    self.assertIsInstance(result_1, tuple)
    self.assertEqual(2, len(result_1))
    self.assertEqual("b", result_1[0])
    self.assertIs(foo, result_1[1])

  def testOneDevice(self):
    device_map = values.ReplicaDeviceMap((_device_str(0),))
    result = values.regroup(device_map, (_nested_value("1"),))
    # On one device regroup() and select_replica() are basically identity.
    self.assertEqual(_nested_value("1"), result)
    self.assertEqual(_nested_value("1"),
                     values.select_replica(0, result))

    # The one exception has to do with MirroredVariables.
    d = "/device:CPU:0"
    with ops.device(d):
      v = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)
      device_map = values.ReplicaDeviceMap((d,))
    mirrored = values.MirroredVariable(device_map, (v,),
                                       variable_scope.VariableAggregation.SUM)
    result = values.regroup(device_map, (v,))
    self.assertIs(mirrored, result)

  def testNamedTupleEstimatorSpec(self):
    with context.graph_mode(), ops.Graph().as_default():
      devices = []
      created_estimator_specs = []

      for device_id in range(3):
        spec = model_fn_lib.EstimatorSpec(
            mode=model_fn_lib.ModeKeys.TRAIN,
            loss=constant_op.constant(device_id / 2),
            train_op=array_ops.identity(constant_op.constant(device_id)))
        devices.append(_device_str(device_id))
        created_estimator_specs.append(spec)

      device_map = values.ReplicaDeviceMap(devices)
      merged_estimator_spec = values.regroup(
          device_map, created_estimator_specs)

      self.assertTrue(
          isinstance(merged_estimator_spec, model_fn_lib.EstimatorSpec))
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, merged_estimator_spec.mode)
      for device_id in range(3):
        d = _device_str(device_id)
        self.assertEqual(created_estimator_specs[device_id].loss,
                         merged_estimator_spec.loss.get(d))
        self.assertEqual(created_estimator_specs[device_id].train_op,
                         merged_estimator_spec.train_op.get(d))
        # Scaffold is populated by `EstimatorSpec.__new__`.
        self.assertEqual(created_estimator_specs[device_id].scaffold,
                         merged_estimator_spec.scaffold.get(d))
        # Also test that we can undo the merge using select_replica()
        self.assertEqual(created_estimator_specs[device_id],
                         values.select_replica(device_id,
                                               merged_estimator_spec))


class PerReplicaDatasetTest(test.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  def _test_iterator(self, devices, dataset, expected_values):
    device_map = values.ReplicaDeviceMap(devices)
    input_workers = values.InputWorkers(device_map)
    per_replica_dataset = values.PerReplicaDataset(dataset, input_workers, 0)
    if context.executing_eagerly():
      iterator = per_replica_dataset.make_one_shot_iterator()
    else:
      iterator = per_replica_dataset.make_initializable_iterator()
      self.evaluate([iterator.initializer])

    for expected_value in expected_values:
      next_element = iterator.get_next_as_list()
      computed_value = self.evaluate(next_element)
      self.assertEqual(expected_value, computed_value)

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next_as_list()
      self.evaluate(next_element)

  @test_util.run_in_graph_and_eager_modes
  def testOneDevice(self):
    devices = ["/device:CPU:0"]
    dataset = dataset_ops.Dataset.range(10)

    expected_values = [[i] for i in range(10)]

    self._test_iterator(devices, dataset, expected_values)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testMultipleDevices(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dataset = dataset_ops.Dataset.range(10)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    self._test_iterator(devices, dataset, expected_values)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testTupleDataset(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))

    expected_values = [[(i, i**2), (i+1, (i+1)**2)] for i in range(0, 10, 2)]

    self._test_iterator(devices, dataset, expected_values)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testUnevenDatasetBatches(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dataset = dataset_ops.Dataset.range(11)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]
    self._test_iterator(devices, dataset, expected_values)

  def testInitializableIterator(self):
    with context.graph_mode():
      devices = ["/device:CPU:0"]
      # Using random input since that is only allowed with initializable
      # iterator.
      dataset = dataset_ops.Dataset.from_tensor_slices(
          random_ops.random_uniform((10,)))

      device_map = values.ReplicaDeviceMap(devices)
      input_workers = values.InputWorkers(device_map)
      per_replica_dataset = values.PerReplicaDataset(dataset, input_workers, 0)
      iterator = per_replica_dataset.make_initializable_iterator()

      self.evaluate(iterator.initializer)
      next_element = iterator.get_next_as_list()
      for _ in range(10):
        self.evaluate(next_element)

      # Should fail after the input is finished.
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

      # After re-initializing the iterator, should be able to iterate again.
      self.evaluate(iterator.initializer)
      for _ in range(10):
        self.evaluate(next_element)


class MultiWorkerDatasetTest(multi_worker_test_base.MultiWorkerTestBase):

  def _test_iterator(self, sess, iterator, devices, expected_values):
    next_element = iterator.get_next()
    for r, device in enumerate(devices):
      v = values.select_replica(r, next_element)
      # The `v` here can be a tuple.
      for element in nest.flatten(v):
        self.assertTrue(element.device in device)

    for expected_value in expected_values:
      t = [values.select_replica(r, next_element) for r in range(len(devices))]
      actual = sess.run(t)
      self.assertEqual(expected_value, actual)

    with self.assertRaises(errors.OutOfRangeError):
      sess.run([values.select_replica(r, next_element)
                for r in range(len(devices))])

  def _test_dataset(self, dataset_fn, worker_devices, devices,
                    expected_values, auto_shard=True):
    device_map = values.ReplicaDeviceMap(devices)
    input_workers = values.InputWorkers(device_map, worker_devices)
    multi_worker_dataset = values.MultiWorkerDataset(
        dataset_fn, input_workers, auto_shard=auto_shard)
    multi_worker_iterator = multi_worker_dataset.make_initializable_iterator()
    with self.cached_session() as sess:
      sess.run(multi_worker_iterator.initializer)
      self._test_iterator(sess, multi_worker_iterator, devices, expected_values)

  def _cpu_devices(self):
    worker_devices = (
        ("/job:worker/replica:0/task:0",
         ["/job:worker/replica:0/task:0/device:CPU:0"]),
        ("/job:worker/replica:0/task:1",
         ["/job:worker/replica:0/task:1/device:CPU:0"])
    )
    devices = [
        "/job:worker/replica:0/task:0/device:CPU:0",
        "/job:worker/replica:0/task:1/device:CPU:0"
    ]
    return worker_devices, devices

  def _cpu_and_one_gpu_devices(self):
    worker_devices = (
        ("/job:worker/replica:0/task:0", (
            "/job:worker/replica:0/task:0/device:GPU:0",
            "/job:worker/replica:0/task:0/device:CPU:0"
        )),
        ("/job:worker/replica:0/task:1", (
            "/job:worker/replica:0/task:1/device:GPU:0",
            "/job:worker/replica:0/task:1/device:CPU:0"
        ))
    )
    devices = [
        "/job:worker/replica:0/task:0/device:GPU:0",
        "/job:worker/replica:0/task:0/device:CPU:0",
        "/job:worker/replica:0/task:1/device:GPU:0",
        "/job:worker/replica:0/task:1/device:CPU:0"
    ]
    return worker_devices, devices

  def testDataDistributionOneDevicePerWorker(self):
    worker_devices, devices = self._cpu_devices()
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      self._test_dataset(dataset_fn, worker_devices, devices,
                         [[0, 1], [2, 3], [4, 5], [6, 7]])

  def testDataDistributionNoAutoShard(self):
    worker_devices, devices = self._cpu_devices()
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(4)
      self._test_dataset(dataset_fn, worker_devices, devices,
                         [[0, 0], [1, 1], [2, 2], [3, 3]],
                         auto_shard=False)

  def testDataDistributionTwoDevicePerWorker(self):
    if context.num_gpus() < 1:
      self.skipTest("A GPU is not available for this test.")
    worker_devices, devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      self._test_dataset(dataset_fn, worker_devices, devices,
                         [[0, 2, 1, 3], [4, 6, 5, 7]])

  def testTupleDataset(self):
    worker_devices, devices = self._cpu_devices()

    with context.graph_mode():

      def dataset_fn():
        dataset1 = dataset_ops.Dataset.range(8)
        dataset2 = dataset_ops.Dataset.range(8).map(lambda x: x**2)
        return dataset_ops.Dataset.zip((dataset1, dataset2))

      expected_values = [
          [(i, i**2), (i + 1, (i + 1)**2)] for i in range(0, 8, 2)
      ]
      self._test_dataset(dataset_fn, worker_devices, devices,
                         expected_values)

  def testInitializableIterator(self):
    worker_devices, devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      device_map = values.ReplicaDeviceMap(devices)
      input_workers = values.InputWorkers(device_map, worker_devices)
      multi_worker_dataset = values.MultiWorkerDataset(
          dataset_fn, input_workers, auto_shard=True)
      multi_worker_iterator = multi_worker_dataset.make_initializable_iterator()

      sess.run(multi_worker_iterator.initializer)
      self._test_iterator(sess, multi_worker_iterator, devices,
                          [[0, 1], [2, 3], [4, 5], [6, 7]])

      # After re-initializing the iterator, should be able to iterate again.
      sess.run(multi_worker_iterator.initializer)
      self._test_iterator(sess, multi_worker_iterator, devices,
                          [[0, 1], [2, 3], [4, 5], [6, 7]])

  def testValueErrorForIterator(self):
    # Incompatiable arguments.
    d1 = "/device:GPU:0"
    d2 = "/device:GPU:1"
    device_map = values.ReplicaDeviceMap([d1, d2])
    input_workers = values.InputWorkers(
        device_map, (("w1", (d1,)), ("w2", (d2,))))
    with self.assertRaises(ValueError):
      values.MultiWorkerDataIterator([("w1", None)], input_workers)

  def testDuplicateDevices(self):
    _, devices = self._cpu_devices()
    devices.append("/job:worker/replica:0/task:0/device:CPU:0")
    with self.assertRaises(ValueError):
      _ = values.ReplicaDeviceMap(devices)


class InputIteratorTestBase(test.TestCase):

  def _test_iterator(self, input_type, dataset_fn, worker_device_pairs,
                     expected_values, sess=None, split_batch_by=None):
    devices = nest.flatten([ds for _, ds in worker_device_pairs])
    device_map = values.ReplicaDeviceMap(devices)
    input_workers = values.InputWorkers(device_map, worker_device_pairs)

    if input_type == "input_fn":
      input_contexts = [
          distribute_lib.InputContext() for _ in worker_device_pairs]
      input_fn = lambda _: dataset_fn()
      iterator = values.InputFunctionIterator(
          input_fn, input_workers, input_contexts)
    else:
      iterator = values.DatasetIterator(
          dataset_fn(), input_workers, split_batch_by)

    evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)

    evaluate(control_flow_ops.group(iterator.initialize()))

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
      self.assertAllEqual(expected_value, computed_value)

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next()
      evaluate([values.select_replica(r, next_element)
                for r in range(len(devices))])

    # After re-initializing the iterator, should be able to iterate again.
    evaluate(control_flow_ops.group(iterator.initialize()))

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
      self.assertAllEqual(expected_value, computed_value)


class InputIteratorSingleWorkerTest(InputIteratorTestBase,
                                    parameterized.TestCase):

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"]))
  def testOneDeviceCPU(self, input_type):
    worker_device_pairs = [("", ["/device:CPU:0"])]
    dataset_fn = lambda: dataset_ops.Dataset.range(10)

    expected_values = [[i] for i in range(10)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTwoDevicesOneGPUOneCPU(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    dataset_fn = lambda: dataset_ops.Dataset.range(10)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTupleDataset(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    def dataset_fn():
      dataset1 = dataset_ops.Dataset.range(10)
      dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
      return dataset_ops.Dataset.zip((dataset1, dataset2))

    expected_values = [[(i, i**2), (i+1, (i+1)**2)] for i in range(0, 10, 2)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testUnevenDatasetBatches(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    dataset_fn = lambda: dataset_ops.Dataset.range(11)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]
    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["dataset"],
      split_batch_by=[None, 2],
      required_gpus=1))
  def testBatchSplitting(self, input_type, split_batch_by):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    batch_size = 10
    dataset_fn = lambda: dataset_ops.Dataset.range(100).batch(batch_size)

    updated_batch_size = (
        batch_size // split_batch_by if split_batch_by else batch_size)
    expected_values = [[range(i, i+updated_batch_size),
                        range(i+updated_batch_size, i+2*updated_batch_size)]
                       for i in range(0, 100, updated_batch_size*2)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values, sess=None,
                        split_batch_by=split_batch_by)


class InputIteratorMultiWorkerTest(
    multi_worker_test_base.MultiWorkerTestBase, InputIteratorTestBase,
    parameterized.TestCase):

  def _cpu_devices(self):
    return [
        ("/job:worker/replica:0/task:0",
         ["/job:worker/replica:0/task:0/device:CPU:0"]),
        ("/job:worker/replica:0/task:1",
         ["/job:worker/replica:0/task:1/device:CPU:0"])]

  def _cpu_and_one_gpu_devices(self):
    return [
        ("/job:worker/replica:0/task:0", [
            "/job:worker/replica:0/task:0/device:GPU:0",
            "/job:worker/replica:0/task:0/device:CPU:0"
        ]),
        ("/job:worker/replica:0/task:1", [
            "/job:worker/replica:0/task:1/device:GPU:0",
            "/job:worker/replica:0/task:1/device:CPU:0"
        ])
    ]

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"]))
  def testOneDevicePerWorker(self, input_type):
    worker_devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      dataset_fn = lambda: dataset_ops.Dataset.range(4)
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          [[0, 0], [1, 1], [2, 2], [3, 3]], sess)

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTwoDevicesPerWorker(self, input_type):
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      dataset_fn = lambda: dataset_ops.Dataset.range(4)
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          [[0, 1, 0, 1], [2, 3, 2, 3]], sess)

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"]))
  def testTupleDataset(self, input_type):
    worker_devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      def dataset_fn():
        dataset1 = dataset_ops.Dataset.range(4)
        dataset2 = dataset_ops.Dataset.range(4).map(lambda x: x**2)
        return dataset_ops.Dataset.zip((dataset1, dataset2))

      expected_values = [[(i, i**2), (i, i**2)] for i in range(0, 4)]
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          expected_values, sess)


class SplitDatasetBatchTest(test.TestCase):

  def testBatchDataset(self):
    dataset = dataset_ops.Dataset.range(100).batch(20)
    split_batch_by = 2
    result_dataset = values._split_dataset_batch(dataset, split_batch_by)
    expected_values = [range(i, i+10) for i in range(0, 100, 10)]
    result = [self.evaluate(el) for el in result_dataset]
    self.assertAllEqual(expected_values, result)

  def testMapAndBatchDataset(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.apply(batching.map_and_batch(lambda x: x, 20))
    split_batch_by = 2
    result_dataset = values._split_dataset_batch(dataset, split_batch_by)
    expected_values = [range(i, i+10) for i in range(0, 100, 10)]
    result = [self.evaluate(el) for el in result_dataset]
    self.assertAllEqual(expected_values, result)

  def testPrefetchDataset(self):
    dataset = dataset_ops.Dataset.range(100).batch(20).prefetch(1)
    split_batch_by = 2
    result_dataset = values._split_dataset_batch(dataset, split_batch_by)
    expected_values = [range(i, i+10) for i in range(0, 100, 10)]
    result = [self.evaluate(el) for el in result_dataset]
    self.assertAllEqual(expected_values, result)


class MirroredVariableTest(test.TestCase, parameterized.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    v, _, mirrored = _make_mirrored()

    self.assertEqual(v[0].name, mirrored.name)
    self.assertEqual(v[0].dtype, mirrored.dtype)
    self.assertEqual(v[0].shape, mirrored.shape)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableOnAnotherDevice(self):
    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    device_map = values.ReplicaDeviceMap(("/job:foo/device:CPU:0",))
    mirrored = values.MirroredVariable(device_map, (v,),
                                       variable_scope.VariableAggregation.MEAN)

    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)

  def _assign_mirrored(self, devices, v, new):
    for d, var, n in zip(devices, v, new):
      with ops.device(d):
        self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveAndRestoreMirroredOneGraph(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.cached_session(config=self.config) as sess:
      v, device_map, mirrored = _make_mirrored()
      devices = device_map.all_devices

      # Overwrite the initial values.
      self._assign_mirrored(devices, v, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path, saver = self._save_return_saver(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(devices, v, [5., 6.])

      # Restores the saved value of 3. to both variables.
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  def _save_mirrored(self):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, device_map, mirrored = _make_mirrored()
      devices = device_map.all_devices

      # Overwrite the initial values.
      self._assign_mirrored(devices, v, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path = self._save(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(devices, v, [5., 6.])
    return save_path

  def _save_normal(self):
    """Save variables without mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(3.))

      # Saves the current value of var, 3.
      save_path = self._save(sess, var)

      # Change the values between save and restore.
      self.evaluate(var.assign(5.))
    return save_path

  def _restore_normal(self, save_path):
    """Restore to variables without mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3. to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3., self.evaluate(var))

  def _restore_mirrored(self, save_path):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, device_map, mirrored = _make_mirrored()
      devices = device_map.all_devices

      # Overwrite the initial values.
      self._assign_mirrored(devices, v, [7., 8.])

      # Restores the saved value of 3. to both variables.
      saver = saver_lib.Saver(var_list=[mirrored])
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveMirroredRestoreMirrored(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored()
    self._restore_mirrored(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveMirroredRestoreNormal(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored()
    self._restore_normal(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveNormalRestoreMirrored(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_normal()
    self._restore_mirrored(save_path)

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_one_gpu,
          combinations.core_mirrored_strategy_with_one_gpu],
      mode=["graph"]))
  def testFetchAMirroredVariable(self, distribution):
    with self.session(graph=ops.Graph()) as sess, distribution.scope():
      with ops.device("/device:GPU:0"):
        v = variable_scope.get_variable(
            name="v", initializer=1., use_resource=True)
      mirrored = values.MirroredVariable(
          values.ReplicaDeviceMap(("/device:GPU:0",)), (v,),
          variable_scope.VariableAggregation.MEAN)
      sess.run(variables_lib.global_variables_initializer())
      sess.run({"complicated": mirrored})


_devices = ("/device:GPU:0", "/device:CPU:0")


def _make_replica_local(method):
  device_map = values.ReplicaDeviceMap(_devices)
  v = []
  for d, n, init in zip(_devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))
  replica_local = values.ReplicaLocalVariable(device_map, v, method)
  return v, replica_local


class ReplicaLocalVariablePropertiesTest(test.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")
    v, replica_local = _make_replica_local(
        variable_scope.VariableAggregation.SUM)

    self.assertEqual(v[0].name, replica_local.name)
    self.assertEqual(v[0].dtype, replica_local.dtype)
    self.assertEqual(v[0].shape, replica_local.shape)
    self.assertEqual(variable_scope.VariableAggregation.SUM,
                     replica_local.aggregation)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableOnAnotherDevice(self):
    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    device_map = values.ReplicaDeviceMap(("/job:foo/device:CPU:0",))
    replica_local = values.ReplicaLocalVariable(
        device_map, (v,), variable_scope.VariableAggregation.MEAN)

    self.assertEqual(v.name, replica_local.name)
    self.assertEqual(v.dtype, replica_local.dtype)
    self.assertEqual(v.shape, replica_local.shape)
    self.assertEqual(variable_scope.VariableAggregation.MEAN,
                     replica_local.aggregation)

  def testTensorConversion(self):
    with context.graph_mode():
      _, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM)
      converted = ops.internal_convert_to_tensor(replica_local, as_ref=False)
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

      converted = ops.internal_convert_to_tensor(replica_local, as_ref=True)
      # Resources variable are converted to tensors as well when as_ref is True.
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)


@combinations.generate(combinations.combine(
    distribution=[
        combinations.mirrored_strategy_with_gpu_and_cpu,
        combinations.core_mirrored_strategy_with_gpu_and_cpu],
    mode=["graph", "eager"]))
class ReplicaLocalVariableTest(test.TestCase, parameterized.TestCase):

  def _assign_replica_local(self, devices, v, new):
    for d, var, n in zip(devices, v, new):
      with ops.device(d):
        self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  def testSaveAndRestoreReplicaLocalSumOneGraph(self, distribution):
    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM)

      # Overwrite the initial values.
      self._assign_replica_local(_devices, v, [3., 4.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 7.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(_devices, v, [5., 6.])

        # Restores the saved value of 7. which gets divided equally
        # between the variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def testSaveAndRestoreReplicaLocalMeanOneGraph(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN)

      # Overwrite the initial values.
      self._assign_replica_local(_devices, v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(_devices, v, [5., 6.])

        # Restores the saved value of 3.5 to both variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _save_replica_local_mean(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN)

      # Overwrite the initial values.
      self._assign_replica_local(_devices, v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(_devices, v, [5., 6.])
    return save_path

  def _save_replica_local_sum(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local("sum")

      # Overwrite the initial values.
      self._assign_replica_local(_devices, v, [1.5, 2.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(_devices, v, [5., 6.])
    return save_path

  def _save_normal(self):
    """Save variables without mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(3.5))

      # Saves the current value of var, 3.5.
      save_path = self._save(sess, var)

      # Change the values between save and restore.
      self.evaluate(var.assign(5.))
    return save_path

  def _restore_normal(self, save_path):
    """Restore to variables without mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3.5 to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3.5, self.evaluate(var))

  def _restore_replica_local_mean(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN)

      # Overwrite the initial values.
      self._assign_replica_local(_devices, v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _restore_replica_local_sum(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM)

      # Overwrite the initial values.
      self._assign_replica_local(_devices, v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([1.75, 1.75], self.evaluate([v[0], v[1]]))

  def testSaveReplicaLocalRestoreReplicaLocalMean(self, distribution):
    save_path = self._save_replica_local_mean(distribution)
    self._restore_replica_local_mean(save_path, distribution)

  def testSaveReplicaLocalRestoreReplicaLocalSum(self, distribution):
    save_path = self._save_replica_local_sum(distribution)
    self._restore_replica_local_sum(save_path, distribution)

  def testSaveReplicaLocalMeanRestoreNormal(self, distribution):
    save_path = self._save_replica_local_mean(distribution)
    self._restore_normal(save_path)

  def testSaveReplicaLocalSumRestoreNormal(self, distribution):
    save_path = self._save_replica_local_sum(distribution)
    self._restore_normal(save_path)

  def testSaveNormalRestoreReplicaLocalMean(self, distribution):
    save_path = self._save_normal()
    self._restore_replica_local_mean(save_path, distribution)

  def testSaveNormalRestoreReplicaLocalSum(self, distribution):
    save_path = self._save_normal()
    self._restore_replica_local_sum(save_path, distribution)


if __name__ == "__main__":
  test.main()
