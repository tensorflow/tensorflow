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
import itertools
import os
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model.model_utils import mode_keys
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.tracking import util as trackable_utils
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

def _make_mirrored_val(init_val=5.0):
  v = []
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, _ in zip(devices, ["v", "v/replica"]):
    with ops.device(d):
      v.append(constant_op.constant(init_val))
  device_map = values.ReplicaDeviceMap(devices)
  mirrored = values.Mirrored(device_map, v)
  return mirrored

def _make_mirrored():
  v = []
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))
  device_map = values.ReplicaDeviceMap(devices)
  mirrored = values.MirroredVariable(None, device_map, v,
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

  def testWrapAListOfTwoTuples(self):
    device_map = values.ReplicaDeviceMap((_device_str(0), _device_str(1)))
    result = values.regroup(device_map, [("1", "2"), ("3", "4")])
    self.assertIsInstance(result, tuple)
    self.assertEqual(2, len(result))
    self._is_per_replica(result[0], ("1", "3"), values.PerReplica)
    self._is_per_replica(result[1], ("2", "4"), values.PerReplica)

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
    mirrored = values.MirroredVariable(None, device_map, (v,),
                                       variable_scope.VariableAggregation.SUM)
    result = values.regroup(device_map, (v,))
    self.assertIs(mirrored, result)

  def testNamedTuple(self):

    # We include toy implementations of Scaffold and EstimatorSpec to
    # avoid a dependency on Estimator here.

    class Scaffold(object):
      pass

    class EstimatorSpec(collections.namedtuple(
        "EstimatorSpec", ["mode", "loss", "train_op", "scaffold"])):

      def __new__(cls, mode, loss, train_op, scaffold=None):
        return super(EstimatorSpec, cls).__new__(
            cls, mode=mode, loss=loss, train_op=train_op,
            scaffold=scaffold or Scaffold())

    with context.graph_mode(), ops.Graph().as_default():
      devices = []
      created_estimator_specs = []

      for device_id in range(3):
        spec = EstimatorSpec(
            mode=mode_keys.EstimatorModeKeys.TRAIN,
            loss=constant_op.constant(device_id / 2),
            train_op=array_ops.identity(constant_op.constant(device_id)))
        devices.append(_device_str(device_id))
        created_estimator_specs.append(spec)

      device_map = values.ReplicaDeviceMap(devices)
      merged_estimator_spec = values.regroup(
          device_map, created_estimator_specs)

      self.assertIsInstance(merged_estimator_spec, EstimatorSpec)
      self.assertEqual(mode_keys.EstimatorModeKeys.TRAIN,
                       merged_estimator_spec.mode)
      for device_id in range(3):
        d = _device_str(device_id)
        self.assertEqual(created_estimator_specs[device_id].loss,
                         merged_estimator_spec.loss.get(d))
        self.assertEqual(created_estimator_specs[device_id].train_op,
                         merged_estimator_spec.train_op.get(d))
        # Scaffold is populated by `EstimatorSpec.__new__`.
        self.assertEqual(created_estimator_specs[device_id].scaffold,
                         merged_estimator_spec.scaffold.get(d))
        self.assertIsInstance(created_estimator_specs[device_id].scaffold,
                              Scaffold)
        # Also test that we can undo the merge using select_replica()
        self.assertEqual(created_estimator_specs[device_id],
                         values.select_replica(device_id,
                                               merged_estimator_spec))


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
    mirrored = values.MirroredVariable(None, device_map, (v,),
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
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
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
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored()
    self._restore_mirrored(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveMirroredRestoreNormal(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored()
    self._restore_normal(save_path)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testSaveNormalRestoreMirrored(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_normal()
    self._restore_mirrored(save_path)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_gpu,
          ],
          mode=["graph"]))
  def testFetchAMirroredVariable(self, distribution):
    with self.session(graph=ops.Graph()) as sess, distribution.scope():
      with ops.device("/device:GPU:0"):
        v = variable_scope.get_variable(
            name="v", initializer=1., use_resource=True)
      mirrored = values.MirroredVariable(
          distribution, values.ReplicaDeviceMap(("/device:GPU:0",)), (v,),
          variable_scope.VariableAggregation.MEAN)
      sess.run(variables_lib.global_variables_initializer())
      sess.run({"complicated": mirrored})

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          mode=["graph", "eager"]))
  def testAssignOutOfScope_mirrored(self, distribution):
    with distribution.scope():
      mirrored = variables_lib.Variable(1.)
    if not isinstance(mirrored, values.MirroredVariable):
      self.assertIsInstance(mirrored, values.TPUMirroredVariable)
    self.evaluate(mirrored.assign(3.))
    self.assertEqual(self.evaluate(mirrored.read_value()), 3.)
    for component in mirrored.values:
      self.assertEqual(self.evaluate(component.read_value()), 3.)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.central_storage_strategy_with_two_gpus
          ],
          mode=["graph", "eager"]))
  def testAssignOutOfScope_aggregating(self, distribution):
    with distribution.scope():
      aggregating = variables_lib.Variable(1.)
    self.assertIsInstance(aggregating, values.AggregatingVariable)
    self.evaluate(aggregating.assign(3.))
    self.assertEqual(self.evaluate(aggregating.read_value()), 3.)
    self.assertEqual(self.evaluate(aggregating._v.read_value()), 3.)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["graph", "eager"]))
  def testExtendsVariable(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(1.)
    self.assertIsInstance(v, variables_lib.Variable)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["graph", "eager"]))
  def testCheckpointing(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(constant_op.constant([1., 2., 3., 4]))

    self.evaluate(v.initializer)
    before_save = self.evaluate(v.read_value())

    # Save random weights into checkpoint.
    checkpoint = trackable_utils.Checkpoint(v=v)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    with self.test_session():
      save_path = checkpoint.save(prefix)

    # Assign inverted value.
    self.evaluate(v.assign(constant_op.constant([4., 3., 2., 1.])))
    after_assign = self.evaluate(v.read_value())
    self.assertNotAllClose(before_save, after_assign)

    # Restore from the checkpoint.
    with self.test_session():
      checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    after_restore = self.evaluate(v)
    self.assertAllClose(before_save, after_restore)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["graph"]))
  def testTraceback(self, distribution):
    with distribution.scope():
      variable_scope.get_variable(
          name="testVar", initializer=1., use_resource=True)
      with self.assertRaisesRegex(
          ValueError, "Variable testVar already exists"):
        variable_scope.get_variable(
            name="testVar", initializer=1., use_resource=True)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["eager"]))
  def testInitializedToSameValueInsideEagerRun(self, distribution):
    v = [None]
    @def_function.function
    def step():
      def f():
        if v[0] is None:
          v[0] = variables_lib.Variable(random_ops.random_normal([]))
      distribution.experimental_run_v2(f)

    context.set_global_seed(None)
    step()
    vals = self.evaluate(v[0].values)
    self.assertAllEqual(vals[0], vals[1])

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["graph", "eager"]))
  def testSelectReplica(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(1.)
    self.assertIs(v, values.select_replica(0, v))

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["graph", "eager"]))
  def testModAfterAssign(self, distribution):
    with distribution.scope():
      v = variables_lib.Variable(0)
    def replica_fn():
      def merge_fn(_):
        return math_ops.mod(v.assign_add(1), 2)
      return distribution_strategy_context.get_replica_context().merge_call(
          merge_fn)

    @def_function.function
    def foo():
      distribution.experimental_run_v2(replica_fn)

    foo()


_TPU_STRATEGIES = (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)


def _make_replica_local(method, strategy=None):
  if strategy is None:
    devices = ("/device:GPU:0", "/device:CPU:0")
  else:
    devices = strategy.extended.worker_devices

  device_map = values.ReplicaDeviceMap(devices)
  v = []
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))

  if (strategy is not None) and isinstance(strategy, _TPU_STRATEGIES):
    var_cls = values.TPUSyncOnReadVariable
  else:
    var_cls = values.SyncOnReadVariable
  replica_local = var_cls(strategy, device_map, v, method)
  return v, replica_local


class SyncOnReadVariablePropertiesTest(test.TestCase):

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
    replica_local = values.SyncOnReadVariable(
        None, device_map, (v,), variable_scope.VariableAggregation.MEAN)

    self.assertEqual(v.name, replica_local.name)
    self.assertEqual(v.dtype, replica_local.dtype)
    self.assertEqual(v.shape, replica_local.shape)
    self.assertEqual(variable_scope.VariableAggregation.MEAN,
                     replica_local.aggregation)

  def testTensorConversion(self):
    with context.graph_mode():
      _, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM)
      converted = ops.convert_to_tensor(replica_local, as_ref=False)
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

      converted = ops.convert_to_tensor(replica_local, as_ref=True)
      # Resources variable are converted to tensors as well when as_ref is True.
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

  @test_util.run_v2_only
  def testCanPassToDefFun(self):
    @def_function.function
    def add1(x):
      return x + 1

    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    device_map = values.ReplicaDeviceMap(("/job:foo/device:CPU:0",))
    replica_local = values.SyncOnReadVariable(
        None, device_map, (v,), variable_scope.VariableAggregation.MEAN)
    self.assertEqual(2., self.evaluate(add1(replica_local)))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.tpu_strategy,
        ],
        mode=["graph", "eager"]))
class SyncOnReadVariableTest(test.TestCase, parameterized.TestCase):

  def _assign_replica_local(self, v, new):
    for var, n in zip(v, new):
      with ops.device(var.device):
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
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 7.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])

        # Restores the saved value of 7. which gets divided equally
        # between the variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def testSaveAndRestoreReplicaLocalMeanOneGraph(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])

        # Restores the saved value of 3.5 to both variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _save_replica_local_mean(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])
    return save_path

  def _save_replica_local_sum(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [1.5, 2.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])
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
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _restore_replica_local_sum(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [7., 8.])

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

  def testAssign(self, distribution):
    def assign(fn, v, update_value, cross_replica):
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        return distribution.experimental_local_results(
            distribution.experimental_run_v2(update_fn))
    updates = [("assign", 1.), ("assign_add", 1.), ("assign_sub", -1.)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = (  # VariableAggregation.SUM in cross-replica mode is tested below
        [x for x in itertools.product(updates, aggregations, [True, False])
         if not(x[1] == variables_lib.VariableAggregation.SUM and x[2])])
    for update, aggregation, cross_replica in options:
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(self.evaluate(component.read_value()),
                            self.evaluate(array_ops.ones_like(component)))

  def testAssignDtypeConversion(self, distribution):
    def assign(fn, v, update_value, cross_replica):
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        return distribution.experimental_local_results(
            distribution.experimental_run_v2(update_fn))
    updates = [("assign", 1), ("assign_add", 1), ("assign_sub", -1)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = (  # VariableAggregation.SUM in cross-replica mode is tested below
        [x for x in itertools.product(updates, aggregations, [True, False])
         if not(x[1] == variables_lib.VariableAggregation.SUM and x[2])])
    for update, aggregation, cross_replica in options:
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(self.evaluate(component.read_value()),
                            self.evaluate(array_ops.ones_like(component)))

  def testAssignWithAggregationSum(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(v.assign(1. * distribution.num_replicas_in_sync))
    for component in v._values:
      self.assertAllEqual(self.evaluate(component.read_value()),
                          self.evaluate(array_ops.ones_like(component)))

  def testAssignAddSubWithAggregationSum(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(variables_lib.global_variables_initializer())
    with self.assertRaisesRegex(
        ValueError, "SyncOnReadVariable does not support "):
      self.evaluate(v.assign_add(1.))
    with self.assertRaisesRegex(
        ValueError, "SyncOnReadVariable does not support "):
      self.evaluate(v.assign_sub(1.))

  def testReadValueInReplicaContext(self, distribution):
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      results = self.evaluate(distribution.experimental_local_results(
          distribution.experimental_run_v2(v.read_value)))
      for component, value in zip(v._values, results):
        self.assertAllEqual(self.evaluate(component.read_value()), value)

  def testReadValueInCrossReplicaContext(self, distribution):
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      if isinstance(distribution, _TPU_STRATEGIES):
        resolver = tpu_cluster_resolver.TPUClusterResolver('')
        tpu_strategy_util.initialize_tpu_system(resolver)
      with distribution.scope():
        v = variable_scope.variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      def assign(v=v):
        ctx = distribution_strategy_context.get_replica_context()
        replica_id = ctx.replica_id_in_sync_group
        return v.assign(math_ops.cast(replica_id, dtypes.float32))
      self.evaluate(distribution.experimental_local_results(
          distribution.experimental_run_v2(assign)))
      result = self.evaluate(v.read_value())
      num_replicas = distribution.num_replicas_in_sync
      sum_of_replica_values = num_replicas * (num_replicas - 1) / 2.
      if aggregation == variables_lib.VariableAggregation.SUM:
        expected = sum_of_replica_values
      elif aggregation == variables_lib.VariableAggregation.MEAN:
        expected = sum_of_replica_values / num_replicas
      else:
        expected = 0
      self.assertEqual(expected, result, aggregation)

  def testReadValueWithAggregationNoneInCrossReplicaContext(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.NONE)
    self.evaluate(variables_lib.global_variables_initializer())
    with self.assertRaisesRegex(
        ValueError, "Could not convert from .* VariableAggregation\\.NONE"):
      self.evaluate(v.read_value())

  def testInitializedToSameValueInsideEagerRun(self, distribution):
    if not context.executing_eagerly(): self.skipTest("eager only")

    v = [None]
    @def_function.function
    def step():
      def f():
        if v[0] is None:
          v[0] = variables_lib.Variable(
              random_ops.random_normal([]),
              synchronization=variables_lib.VariableSynchronization.ON_READ)
      distribution.experimental_run_v2(f)

    context.set_global_seed(None)
    step()
    vals = self.evaluate(v[0].values)
    self.assertAllEqual(vals[0], vals[1])

class MirroredTest(test.TestCase):

  def testAddOp(self):
    if context.num_gpus() < 1:
      self.skipTest("A GPU is not available for this test.")
    mirrored_val = _make_mirrored_val(init_val=3.)

    self.assertEqual(self.evaluate(constant_op.constant(6.)),
                     self.evaluate(mirrored_val + mirrored_val))
    self.assertEqual(self.evaluate(constant_op.constant(4.)),
                     self.evaluate(mirrored_val + 1))
    self.assertEqual(self.evaluate(mirrored_val + 1),
                     self.evaluate(math_ops.add(mirrored_val, 1)))
    self.assertEqual(type(mirrored_val + 1),
                     type(math_ops.add(mirrored_val, 1)))


class PerReplicaTest(test.TestCase, parameterized.TestCase):

  def testTypeSpec(self):
    device_map = values.SingleDeviceMap("CPU")
    vals = (constant_op.constant(1.),)
    per_replica = values.PerReplica(device_map, vals)

    spec = per_replica._type_spec
    self.assertEqual(spec._value_specs,
                     (tensor_spec.TensorSpec([], dtypes.float32),))
    self.assertEqual(spec._device_map, per_replica.device_map)
    self.assertEqual(spec._logical_device, per_replica.logical_device)

  def testTypeSpecRoundTrip(self):
    device_map = values.SingleDeviceMap("CPU")
    vals = (constant_op.constant(1.),)
    per_replica = values.PerReplica(device_map, vals)

    spec = per_replica._type_spec
    tensor_list = spec._to_components(per_replica)
    reconstructed = spec._from_components(tensor_list)

    self.assertEqual(per_replica.device_map, reconstructed.device_map)
    self.assertEqual(per_replica.logical_device, reconstructed.logical_device)
    self.assertAllEqual(per_replica.values, reconstructed.values)

  def testTypeSpecNest(self):
    device_map = values.ReplicaDeviceMap(["CPU:0", "CPU:1"])
    vals = (constant_op.constant(1.), constant_op.constant([5., 6.0]),)
    per_replica = values.PerReplica(device_map, vals)

    # Note: nest.map_structutre exercises nest.flatten and
    # nest.pack_sequence_as.
    result = nest.map_structure(lambda t: t + 10, per_replica,
                                expand_composites=True)

    self.assertEqual(per_replica.device_map, result.device_map)
    self.assertEqual(per_replica.logical_device, result.logical_device)
    self.assertLen(result.values, 2)
    self.assertAllEqual(result.values[0], 11.)
    self.assertAllEqual(result.values[1], [15., 16.0])

  @test_util.run_in_graph_and_eager_modes
  def testIsGraphTensor(self):
    per_replica = values.PerReplica(values.SingleDeviceMap("CPU"),
                                    (constant_op.constant(1.),))
    for t in nest.flatten(per_replica, expand_composites=True):
      self.assertEqual(hasattr(t, "graph"), not context.executing_eagerly())

  def testDoesNotTriggerFunctionTracing(self):
    traces = []

    @def_function.function
    def f(x):
      traces.append(None)  # Only happens on trace.
      return x

    per_replica = values.PerReplica(
        values.SingleDeviceMap("CPU"), (constant_op.constant(1.),))

    # Trace once.
    f(per_replica)
    self.assertNotEmpty(traces)
    del traces[:]

    per_replica_spec = per_replica._type_spec
    for _ in range(5):
      vals = per_replica_spec._to_components(per_replica)
      vals = [v * 2 for v in vals]
      per_replica = per_replica_spec._from_components(vals)

      output = f(per_replica)
      self.assertIsInstance(output, values.PerReplica)
      self.assertAllEqual(output._values, per_replica._values)
      self.assertAllEqual(output._device_map, per_replica._device_map)
      self.assertAllEqual(output._logical_device, per_replica._logical_device)
      self.assertEmpty(traces)  # Make sure we're not re-tracing `f`.

  def testFunctionCanReturnPerReplica(self):
    f = def_function.function(lambda x: x)
    x = values.PerReplica(
        values.SingleDeviceMap("CPU"), (constant_op.constant(1.),))
    y = f(x)
    self.assertIsNot(x, y)
    nest.map_structure(self.assertAllEqual, x, y, expand_composites=True)
    self.assertEqual(x._type_spec, y._type_spec)

  @test_util.run_in_graph_and_eager_modes
  def testCondWithTensorValues(self):
    device_map = values.SingleDeviceMap("CPU")
    per_replica_1 = values.PerReplica(device_map, (constant_op.constant("a"),))
    per_replica_2 = values.PerReplica(device_map,
                                      (constant_op.constant(["b", "c"]),))
    condition = array_ops.placeholder_with_default(True, [])

    result = control_flow_ops.cond(
        condition, lambda: per_replica_1, lambda: per_replica_2)

    self.assertEqual(per_replica_1.device_map, result.device_map)
    self.assertEqual(per_replica_1.logical_device, result.logical_device)
    self.assertLen(result.values, 1)
    self.assertAllEqual(result.values[0], "a")

  @test_util.run_in_graph_and_eager_modes
  def testCondWithValuesConvertibleToTensor(self):
    device_map = values.SingleDeviceMap("CPU")
    per_replica_1 = values.PerReplica(device_map, ("a",))
    per_replica_2 = values.PerReplica(device_map, ("b",))
    condition = array_ops.placeholder_with_default(True, [])

    result = control_flow_ops.cond(
        condition, lambda: per_replica_1, lambda: per_replica_2)

    self.assertEqual(per_replica_1.device_map, result.device_map)
    self.assertEqual(per_replica_1.logical_device, result.logical_device)
    self.assertLen(result.values, 1)
    self.assertAllEqual(result.values[0], "a")

  @test_util.build_as_function_and_v1_graph
  def testCondWithValuesNotConvertibleToTensor(self):
    device_map = values.SingleDeviceMap("CPU")
    per_replica_1 = values.PerReplica(device_map, (set(["a"]),))
    per_replica_2 = values.PerReplica(device_map, (set(["b", "c"]),))
    condition = array_ops.placeholder(dtypes.bool, [])

    with self.assertRaisesRegex(TypeError, "Could not build a TypeSpec for"):
      control_flow_ops.cond(
          condition, lambda: per_replica_1, lambda: per_replica_2)


class WorkerDeviceMapTest(test.TestCase, parameterized.TestCase):

  class ReplicaContext(object):

    def __init__(self, replica_id_in_sync_group):
      self.replica_id_in_sync_group = replica_id_in_sync_group

  def testBasic(self):
    devices = [
        "/job:worker/replica:0/task:0/device:CPU:0",
        "/job:worker/replica:0/task:2/device:CPU:0"
    ]
    device_map = values.WorkerDeviceMap(devices, 1)
    self.assertAllEqual(devices, device_map.all_devices)

    # pylint:disable=pointless-statement
    with self.assertRaisesWithPredicateMatch(
        ValueError, "`WorkerDeviceMap` is not indexed by replicas"):
      device_map.devices_by_replica

    self.assertEqual(1, device_map.num_logical_devices)

    self.assertEqual(2, device_map.num_replicas_in_graph)

    self.assertEqual(0, device_map.logical_device_from_values(["a", "b"]))

    self.assertAllEqual(devices, device_map.logical_to_actual_devices(0))

    replica_context = WorkerDeviceMapTest.ReplicaContext(1)
    self.assertEqual(
        "b", device_map.select_for_current_replica(["a", "b"], replica_context))

    with self.assertRaisesWithPredicateMatch(
        ValueError, "`WorkerDeviceMap` not indexed by replicas"):
      device_map.replica_for_device(devices[1])

    self.assertEqual("b", device_map.select_for_device(["a", "b"], devices[1]))

    with self.assertRaisesWithPredicateMatch(
        ValueError, "WorkerDeviceMap not indexed by replicas"):
      device_map.is_device_in_replica(devices[1], 1)

    self.assertEqual(
        "WorkerDeviceMap(('/job:worker/replica:0/task:0/device:CPU:0', "
        "'/job:worker/replica:0/task:2/device:CPU:0'), "
        "num_replicas_per_worker=1)", repr(device_map))

  def testMultipleReplicasPerWorker(self):
    devices = [
        "/job:worker/replica:0/task:0/device:CPU:0",
        "/job:worker/replica:0/task:2/device:CPU:0"
    ]
    device_map = values.WorkerDeviceMap(devices, 2)

    replica_context = WorkerDeviceMapTest.ReplicaContext(3)
    self.assertEqual(
        "b", device_map.select_for_current_replica(["a", "b"], replica_context))

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          mode=["graph", "eager"]))
  def testExperimentalLocalResultsOrder(self, distribution):
    # Create 2 devices in the device map, where the alphabetical order and the
    # actual order of devices are different.
    device_map = values.ReplicaDeviceMap(["CPU:2", "CPU:10"])
    vals = (
        constant_op.constant(1.),
        constant_op.constant([5., 6.0]),
    )
    per_replica = values.PerReplica(device_map, vals)
    results = self.evaluate(
        distribution.experimental_local_results(per_replica))

    # We expect the outputs order the same as the inputs order.
    self.assertLen(results, 2)
    self.assertAllEqual(1.0, results[0])
    self.assertAllEqual([5., 6.], results[1])


if __name__ == "__main__":
  test.main()
