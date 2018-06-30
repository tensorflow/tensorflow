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

import collections
import os

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.contrib.distribute.python import values
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import device_util
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import nest


class DistributedValuesTest(test.TestCase):

  def testGetEager(self):
    with ops.device("/device:CPU:0"):
      one = constant_op.constant(1)
      two = constant_op.constant(2)
      v = values.DistributedValues({"/device:CPU:0": one, "/device:GPU:0": two})
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
      v = values.DistributedValues({"/device:CPU:0": one, "/device:GPU:0": two})
      self.assertEqual(two, v.get("/device:GPU:0"))
      self.assertEqual(one, v.get())
      with self.assertRaises(ValueError):
        self.assertIsNone(v.get("/device:GPU:2"))

  def testCanonicalization(self):
    canonical_cpu = ["/job:localhost/replica:0/task:0/device:CPU:0"]
    v = values.DistributedValues({"": 42})
    self.assertEqual(canonical_cpu, list(v._index.keys()))
    v = values.DistributedValues({"/device:CPU:0": 42})
    self.assertEqual(canonical_cpu, list(v._index.keys()))
    v = values.DistributedValues({"/cpu:0": 42})
    self.assertEqual(canonical_cpu, list(v._index.keys()))
    v = values.DistributedValues({"/CPU:0": 42})
    self.assertEqual(canonical_cpu, list(v._index.keys()))
    with self.assertRaises(AssertionError):
      v = values.DistributedValues({"/device:cpu:0": 42})


class DistributedDelegateTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testGetAttr(self):
    with ops.device("/device:CPU:0"):

      class Foo(object):

        def __init__(self, x):
          self.x = x

      v = values.DistributedDelegate(
          {"/device:CPU:0": Foo(7), "/device:GPU:0": Foo(8)})
      self.assertEqual(7, v.x)
      with self.assertRaises(AttributeError):
        _ = v.y

  @test_util.run_in_graph_and_eager_modes
  def testOperatorOverride(self):
    with ops.device("/device:CPU:0"):
      v = values.DistributedDelegate({"/device:CPU:0": 7, "/device:GPU:0": 8})
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
  index = {}
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))
      index[d] = v[-1]
  mirrored = values.MirroredVariable(index, v[0],
                                     variable_scope.VariableAggregation.SUM)
  return v, devices, mirrored


class RegroupAndSelectDeviceTest(test.TestCase):

  def _is_per_device(self, result, expected, klass=values.PerDevice):
    self.assertIsInstance(result, klass)
    # We canonicalize the devices to match the device strings returned
    # by PerDevice, which also does device string canonicalization.
    devices = [device_util.canonicalize(_device_str(i))
               for i in range(len(expected))]
    self.assertEqual(set(devices), set(result.devices))
    for i, d in enumerate(devices):
      self.assertEqual(expected[i], result.get(d))
      self.assertEqual(expected[i], result.get(_device_str(i)))

  def testNested(self):
    result = values.regroup({_device_str(0): _nested_value("1"),
                             _device_str(1): _nested_value("2")})
    self.assertIsInstance(result, tuple)
    self.assertEqual(3, len(result))
    self._is_per_device(result[0], ["a1", "a2"])
    self._is_per_device(result[2], ["h1", "h2"])

    self.assertIsInstance(result[1], list)
    self.assertEqual(3, len(result[1]))
    self._is_per_device(result[1][0], ["b1", "b2"])
    self._is_per_device(result[1][2], ["g1", "g2"])

    self.assertIsInstance(result[1][1], dict)
    self.assertEqual(set(["c", "e"]), set(result[1][1].keys()))
    self._is_per_device(result[1][1]["c"], ["d1", "d2"])
    self._is_per_device(result[1][1]["e"], ["f1", "f2"])

    # Also test that we can undo the merge using select_device()
    self.assertEqual(_nested_value("1"),
                     values.select_device(_device_str(0), result))
    self.assertEqual(_nested_value("2"),
                     values.select_device(_device_str(1), result))
    # select_device_mirrored() should fail due to non-mirrored values
    with self.assertRaises(TypeError):
      values.select_device_mirrored(_device_str(0), result)
    with self.assertRaises(TypeError):
      values.select_device_mirrored(_device_str(1), result)

  def testWrapClass(self):
    # Normally a mirrored value would be the same across devices, but
    # for a test it is convenient to be able to tell the values apart.
    result = values.regroup({_device_str(0): _nested_value("1"),
                             _device_str(1): _nested_value("2")},
                            values.Mirrored)
    self.assertIsInstance(result, tuple)
    self.assertEqual(3, len(result))
    self._is_per_device(result[0], ["a1", "a2"], values.Mirrored)
    self._is_per_device(result[2], ["h1", "h2"], values.Mirrored)

    self.assertIsInstance(result[1], list)
    self.assertEqual(3, len(result[1]))
    self._is_per_device(result[1][0], ["b1", "b2"], values.Mirrored)
    self._is_per_device(result[1][2], ["g1", "g2"], values.Mirrored)

    self.assertIsInstance(result[1][1], dict)
    self.assertEqual(set(["c", "e"]), set(result[1][1].keys()))
    self._is_per_device(result[1][1]["c"], ["d1", "d2"], values.Mirrored)
    self._is_per_device(result[1][1]["e"], ["f1", "f2"], values.Mirrored)

    # Also test that we can undo the merge using select_device()
    self.assertEqual(_nested_value("1"),
                     values.select_device(_device_str(0), result))
    self.assertEqual(_nested_value("2"),
                     values.select_device(_device_str(1), result))
    # Values are marked as mirrored, so select_device_mirrored() is allowed.
    self.assertEqual(_nested_value("1"),
                     values.select_device_mirrored(_device_str(0), result))
    self.assertEqual(_nested_value("2"),
                     values.select_device_mirrored(_device_str(1), result))

  def testMirroredContainer(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")
    v, devices, mirrored = _make_mirrored()
    result = values.regroup(dict(zip(devices, v)))
    self.assertIs(mirrored, result)

  def testSameId(self):
    foo = object()
    result = values.regroup({_device_str(0): ("a", foo),
                             _device_str(1): ("b", foo)})
    self.assertIsInstance(result, tuple)
    self.assertEqual(2, len(result))
    self._is_per_device(result[0], ["a", "b"])
    self.assertIs(foo, result[1])

    # Test select_device(), should undo the merge done by regroup().
    result_0 = values.select_device(_device_str(0), result)
    self.assertIsInstance(result_0, tuple)
    self.assertEqual(2, len(result_0))
    self.assertEqual("a", result_0[0])
    self.assertIs(foo, result_0[1])
    result_1 = values.select_device(_device_str(1), result)
    self.assertIsInstance(result_1, tuple)
    self.assertEqual(2, len(result_1))
    self.assertEqual("b", result_1[0])
    self.assertIs(foo, result_1[1])

  def testOneDevice(self):
    result = values.regroup({_device_str(0): _nested_value("1")})
    # On one device regroup() and select_device() are basically identity.
    self.assertEqual(_nested_value("1"), result)
    self.assertEqual(_nested_value("1"),
                     values.select_device(_device_str(0), result))

    # The one exception has to do with MirroredVariables.
    d = "/device:CPU:0"
    with ops.device(d):
      v = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)
      index = {d: v}
    mirrored = values.MirroredVariable(index, v,
                                       variable_scope.VariableAggregation.SUM)
    result = values.regroup(index)
    self.assertIs(mirrored, result)

  def testNamedTupleEstimatorSpec(self):
    with context.graph_mode(), ops.Graph().as_default():
      created_estimator_specs = {}
      to_regroup = {}

      for device_id in range(3):
        spec = model_fn_lib.EstimatorSpec(
            mode=model_fn_lib.ModeKeys.TRAIN,
            loss=constant_op.constant(device_id / 2),
            train_op=array_ops.identity(constant_op.constant(device_id)))
        created_estimator_specs[device_id] = spec
        to_regroup[_device_str(device_id)] = spec

      merged_estimator_spec = values.regroup(to_regroup)

      self.assertTrue(
          isinstance(merged_estimator_spec, model_fn_lib.EstimatorSpec))
      self.assertEquals(model_fn_lib.ModeKeys.TRAIN, merged_estimator_spec.mode)
      for device_id in range(3):
        d = _device_str(device_id)
        self.assertEquals(created_estimator_specs[device_id].loss,
                          merged_estimator_spec.loss.get(d))
        self.assertEquals(created_estimator_specs[device_id].train_op,
                          merged_estimator_spec.train_op.get(d))
        # Scaffold is populated by `EstimatorSpec.__new__`.
        self.assertEquals(created_estimator_specs[device_id].scaffold,
                          merged_estimator_spec.scaffold.get(d))
        # Also test that we can undo the merge using select_device()
        self.assertEquals(created_estimator_specs[device_id],
                          values.select_device(_device_str(device_id),
                                               merged_estimator_spec))


class PerDeviceDatasetTest(test.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  def _test_iterator_no_prefetch(self, devices, dataset, expected_values):
    per_device_dataset = values.PerDeviceDataset(
        dataset, devices, prefetch_on_device=False)
    iterator = per_device_dataset.make_one_shot_iterator()

    for expected_value in expected_values:
      next_element = iterator.get_next()
      actual = self.evaluate([
          values.select_device(d, next_element) for d in devices])
      self.assertEqual(expected_value, actual)

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next()
      self.evaluate([
          values.select_device(d, next_element) for d in devices])

  def _test_iterator_with_prefetch(self, devices, dataset, expected_values):
    if not context.executing_eagerly():
      per_device_dataset = values.PerDeviceDataset(
          dataset, devices, prefetch_on_device=True)
      iterator = per_device_dataset.make_one_shot_iterator()

      # With prefetching, we cannot guarantee which input ends up on which
      # device, so we verify that the complete set seen on all devices is
      # correct, and equal numbers are distributed to each device.
      combined_actual = []
      combined_expected = []
      for expected_value in expected_values:
        next_element = iterator.get_next()
        combined_actual.extend(self.evaluate([
            values.select_device(d, next_element) for d in devices]))
        combined_expected.extend(expected_value)

      self.assertEqual(set(combined_expected), set(combined_actual))

      with self.assertRaises(errors.OutOfRangeError):
        next_element = iterator.get_next()
        self.evaluate([
            values.select_device(d, next_element) for d in devices])

  def _test_iterator(self, devices, dataset, expected_values):
    self._test_iterator_no_prefetch(devices, dataset, expected_values)
    self._test_iterator_with_prefetch(devices, dataset, expected_values)

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

      per_device_dataset = values.PerDeviceDataset(
          dataset, devices, prefetch_on_device=False)
      iterator = per_device_dataset.make_initializable_iterator()

      self.evaluate(iterator.initializer)
      next_element = iterator.get_next()
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

  def _test_iterator(self, iterator, devices, expected_values):
    next_element = iterator.get_next()
    for device in devices:
      v = values.select_device(device, next_element)
      # The `v` here can be a tuple.
      for element in nest.flatten(v):
        self.assertTrue(element.device in device)

    for expected_value in expected_values:
      actual = self.evaluate(
          [values.select_device(d, next_element) for d in devices])
      self.assertEqual(expected_value, actual)

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate([values.select_device(d, next_element) for d in devices])

  def _test_dataset(self, dataset_fn, worker_device_map, devices,
                    expected_values):
    multi_worker_dataset = values.MultiWorkerDataset(
        dataset_fn, worker_device_map, prefetch_on_device=False)
    multi_worker_iterator = multi_worker_dataset.make_one_shot_iterator()
    self._test_iterator(multi_worker_iterator, devices, expected_values)

  def _cpu_devices(self):
    worker_device_map = collections.OrderedDict(
        [("/job:worker/replica:0/task:0",
          ["/job:worker/replica:0/task:0/device:CPU:0"]),
         ("/job:worker/replica:0/task:1",
          ["/job:worker/replica:0/task:1/device:CPU:0"])])
    devices = [
        "/job:worker/replica:0/task:0/device:CPU:0",
        "/job:worker/replica:0/task:1/device:CPU:0"
    ]
    return worker_device_map, devices

  def _cpu_and_one_gpu_devices(self):
    # The worker_device_map doesn't have to be a OrderDict object, this is just
    # to simplify the testing so that we can pass expected values as a list
    # instead of a dict.
    worker_device_map = collections.OrderedDict(
        [("/job:worker/replica:0/task:0", [
            "/job:worker/replica:0/task:0/device:GPU:0",
            "/job:worker/replica:0/task:0/device:CPU:0"
        ]), ("/job:worker/replica:0/task:1", [
            "/job:worker/replica:0/task:1/device:GPU:0",
            "/job:worker/replica:0/task:1/device:CPU:0"
        ])])
    devices = [
        "/job:worker/replica:0/task:0/device:GPU:0",
        "/job:worker/replica:0/task:0/device:CPU:0",
        "/job:worker/replica:0/task:1/device:GPU:0",
        "/job:worker/replica:0/task:1/device:CPU:0"
    ]
    return worker_device_map, devices

  def testDataDistributionOneDevicePerWorker(self):
    worker_device_map, devices = self._cpu_devices()
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      self._test_dataset(dataset_fn, worker_device_map, devices,
                         [[0, 1], [2, 3], [4, 5], [6, 7]])

  def testDataDistributionTwoDevicePerWorker(self):
    if context.num_gpus() < 1:
      self.skipTest("A GPU is not available for this test.")
    worker_device_map, devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      self._test_dataset(dataset_fn, worker_device_map, devices,
                         [[0, 2, 1, 3], [4, 6, 5, 7]])

  def testTupleDataset(self):
    worker_device_map, devices = self._cpu_devices()

    with context.graph_mode():

      def dataset_fn():
        dataset1 = dataset_ops.Dataset.range(8)
        dataset2 = dataset_ops.Dataset.range(8).map(lambda x: x**2)
        return dataset_ops.Dataset.zip((dataset1, dataset2))

      expected_values = [
          [(i, i**2), (i + 1, (i + 1)**2)] for i in range(0, 8, 2)
      ]
      self._test_dataset(dataset_fn, worker_device_map, devices,
                         expected_values)

  def testInitializableIterator(self):
    worker_device_map, devices = self._cpu_devices()
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      multi_worker_dataset = values.MultiWorkerDataset(
          dataset_fn, worker_device_map, prefetch_on_device=False)
      multi_worker_iterator = multi_worker_dataset.make_initializable_iterator()

      self.evaluate(multi_worker_iterator.initializer)
      self._test_iterator(multi_worker_iterator, devices,
                          [[0, 1], [2, 3], [4, 5], [6, 7]])

      # After re-initializing the iterator, should be able to iterate again.
      self.evaluate(multi_worker_iterator.initializer)
      self._test_iterator(multi_worker_iterator, devices,
                          [[0, 1], [2, 3], [4, 5], [6, 7]])

  def testValueErrorForIterator(self):
    # Incompatiable arguments.
    with self.assertRaises(ValueError):
      values.MultiWorkerDataIterator({"w1": None}, {"w1": "d1", "w2": "d2"})

    # Test duplicated devices under same worker.
    worker_device_map, _ = self._cpu_devices()
    worker_device_map["/job:worker/replica:0/task:0"].append(
        "/job:worker/replica:0/task:0/device:CPU:0")
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      multi_worker_dataset = values.MultiWorkerDataset(
          dataset_fn, worker_device_map, prefetch_on_device=False)
      multi_worker_iterator = multi_worker_dataset.make_initializable_iterator()
      with self.assertRaises(ValueError):
        multi_worker_iterator.get_next()


class MirroredVariableTest(test.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    v, _, mirrored = _make_mirrored()

    self.assertEquals(v[0].name, mirrored.name)
    self.assertEquals(v[0].dtype, mirrored.dtype)
    self.assertEquals(v[0].shape, mirrored.shape)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableOnAnotherDevice(self):
    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    index = {"/job:foo/device:CPU:0": v}
    mirrored = values.MirroredVariable(index, v,
                                       variable_scope.VariableAggregation.MEAN)

    self.assertEquals(v.name, mirrored.name)
    self.assertEquals(v.dtype, mirrored.dtype)
    self.assertEquals(v.shape, mirrored.shape)

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

    with self.test_session() as sess:
      v, devices, mirrored = _make_mirrored()

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
    with self.test_session(graph=ops.Graph()) as sess:
      v, devices, mirrored = _make_mirrored()

      # Overwrite the initial values.
      self._assign_mirrored(devices, v, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path = self._save(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(devices, v, [5., 6.])
    return save_path

  def _save_normal(self):
    """Save variables without mirroring, returns save_path."""
    with self.test_session(graph=ops.Graph()) as sess:
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
    with self.test_session(graph=ops.Graph()) as sess:
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
    with self.test_session(graph=ops.Graph()) as sess:
      v, devices, mirrored = _make_mirrored()

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

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testFetchAMirroredVariable(self):
    if context.num_gpus() < 1 or context.executing_eagerly():
      self.skipTest("A GPU is not available for this test or it's eager mode.")

    with self.test_session(
        graph=ops.Graph()) as sess, mirrored_strategy.MirroredStrategy(
            ["/device:GPU:0"]).scope():
      with ops.device("/device:GPU:0"):
        v = variable_scope.get_variable(
            name="v", initializer=1., use_resource=True)
      mirrored = values.MirroredVariable({
          "/device:GPU:0": v
      }, v, variable_scope.VariableAggregation.MEAN)
      sess.run(variables_lib.global_variables_initializer())
      sess.run({"complicated": mirrored})


_devices = ["/device:GPU:0", "/device:CPU:0"]


def _make_tower_local(method):
  v = []
  index = {}
  for d, n, init in zip(_devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))
      index[d] = v[-1]
  tower_local = values.TowerLocalVariable(index, v[0], method)
  return v, tower_local


class TowerLocalVariableTest(test.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    v, tower_local = _make_tower_local(variable_scope.VariableAggregation.SUM)

    self.assertEquals(v[0].name, tower_local.name)
    self.assertEquals(v[0].dtype, tower_local.dtype)
    self.assertEquals(v[0].shape, tower_local.shape)
    self.assertEquals(variable_scope.VariableAggregation.SUM,
                      tower_local.aggregation)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableOnAnotherDevice(self):
    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    index = {"/job:foo/device:CPU:0": v}
    tower_local = values.TowerLocalVariable(
        index, v, variable_scope.VariableAggregation.MEAN)

    self.assertEquals(v.name, tower_local.name)
    self.assertEquals(v.dtype, tower_local.dtype)
    self.assertEquals(v.shape, tower_local.shape)
    self.assertEquals(variable_scope.VariableAggregation.MEAN,
                      tower_local.aggregation)

  def _assign_tower_local(self, devices, v, new):
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

  def _dist_scope(self):
    return mirrored_strategy.MirroredStrategy(_devices).scope()

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveAndRestoreTowerLocalSumOneGraph(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.test_session() as sess:
      v, tower_local = _make_tower_local(variable_scope.VariableAggregation.SUM)

      # Overwrite the initial values.
      self._assign_tower_local(_devices, v, [3., 4.])

      with self._dist_scope():
        # Saves the current value of v[0] + v[1], 7.
        save_path, saver = self._save_return_saver(sess, tower_local)

        # Change the values between save and restore.
        self._assign_tower_local(_devices, v, [5., 6.])

        # Restores the saved value of 7. which gets divided equally
        # between the variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveAndRestoreTowerLocalMeanOneGraph(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.test_session() as sess:
      v, tower_local = _make_tower_local(
          variable_scope.VariableAggregation.MEAN)

      # Overwrite the initial values.
      self._assign_tower_local(_devices, v, [3., 4.])

      with self._dist_scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5.
        save_path, saver = self._save_return_saver(sess, tower_local)

        # Change the values between save and restore.
        self._assign_tower_local(_devices, v, [5., 6.])

        # Restores the saved value of 3.5 to both variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _save_tower_local_mean(self):
    """Save variables with mirroring, returns save_path."""
    with self.test_session(graph=ops.Graph()) as sess:
      v, tower_local = _make_tower_local(
          variable_scope.VariableAggregation.MEAN)

      # Overwrite the initial values.
      self._assign_tower_local(_devices, v, [3., 4.])

      with self._dist_scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5
        save_path = self._save(sess, tower_local)

        # Change the values between save and restore.
        self._assign_tower_local(_devices, v, [5., 6.])
    return save_path

  def _save_tower_local_sum(self):
    """Save variables with mirroring, returns save_path."""
    with self.test_session(graph=ops.Graph()) as sess:
      v, tower_local = _make_tower_local("sum")

      # Overwrite the initial values.
      self._assign_tower_local(_devices, v, [1.5, 2.])

      with self._dist_scope():
        # Saves the current value of v[0] + v[1], 3.5
        save_path = self._save(sess, tower_local)

        # Change the values between save and restore.
        self._assign_tower_local(_devices, v, [5., 6.])
    return save_path

  def _save_normal(self):
    """Save variables without mirroring, returns save_path."""
    with self.test_session(graph=ops.Graph()) as sess:
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
    with self.test_session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3.5 to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3.5, self.evaluate(var))

  def _restore_tower_local_mean(self, save_path):
    """Restore to variables with mirroring in a fresh graph."""
    with self.test_session(graph=ops.Graph()) as sess:
      v, tower_local = _make_tower_local(
          variable_scope.VariableAggregation.MEAN)

      # Overwrite the initial values.
      self._assign_tower_local(_devices, v, [7., 8.])

      with self._dist_scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[tower_local])
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _restore_tower_local_sum(self, save_path):
    """Restore to variables with mirroring in a fresh graph."""
    with self.test_session(graph=ops.Graph()) as sess:
      v, tower_local = _make_tower_local(variable_scope.VariableAggregation.SUM)

      # Overwrite the initial values.
      self._assign_tower_local(_devices, v, [7., 8.])

      with self._dist_scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[tower_local])
        saver.restore(sess, save_path)
        self.assertEqual([1.75, 1.75], self.evaluate([v[0], v[1]]))

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveTowerLocalRestoreTowerLocalMean(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_tower_local_mean()
    self._restore_tower_local_mean(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveTowerLocalRestoreTowerLocalSum(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_tower_local_sum()
    self._restore_tower_local_sum(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveTowerLocalMeanRestoreNormal(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_tower_local_mean()
    self._restore_normal(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveTowerLocalSumRestoreNormal(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_tower_local_sum()
    self._restore_normal(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveNormalRestoreTowerLocalMean(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_normal()
    self._restore_tower_local_mean(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveNormalRestoreTowerLocalSum(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_normal()
    self._restore_tower_local_sum(save_path)

  def testTensorConversion(self):
    with context.graph_mode():
      _, tower_local = _make_tower_local(variable_scope.VariableAggregation.SUM)
      converted = ops.internal_convert_to_tensor(tower_local, as_ref=False)
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, tower_local.dtype)

      converted = ops.internal_convert_to_tensor(tower_local, as_ref=True)
      # Resources variable are converted to tensors as well when as_ref is True.
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, tower_local.dtype)


if __name__ == "__main__":
  test.main()
