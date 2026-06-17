# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the mirrored values library."""

import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util as ds_test_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import saver as saver_lib


def _make_mirrored(distribution=None):
  v = []
  if distribution:
    devices = distribution.extended.worker_devices
  else:
    devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(
          variable_scope.get_variable(
              name=n, initializer=init, use_resource=True))

  if (distribution
      is not None) and strategy_test_lib.is_tpu_strategy(distribution):
    var_cls = tpu_values.TPUMirroredVariable
  else:
    var_cls = values_lib.MirroredVariable
  mirrored = var_cls(distribution, v, variable_scope.VariableAggregation.SUM)
  return mirrored


def _make_mirrored_val(init_val=5.0):
  v = []
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, _ in zip(devices, ["v", "v/replica"]):
    with ops.device(d):
      v.append(constant_op.constant(init_val))
  return values_lib.Mirrored(v)


def mirrored_and_tpu_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ],
      mode=["graph", "eager"])


class MirroredVariableTest(test.TestCase, parameterized.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  def tearDown(self):
    super().tearDown()
    context._reset_context()

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    mirrored = _make_mirrored()
    v = mirrored.values[0]
    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableOnAnotherDevice(self):
    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    mirrored = values_lib.MirroredVariable(
        None, (v,), variable_scope.VariableAggregation.MEAN)

    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)


class MirroredVariableSaveRestoreTest(test.TestCase, parameterized.TestCase):

  def _assign_mirrored(self, v, new):
    for var, n in zip(v.values, new):
      self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  def _save_mirrored(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      mirrored = _make_mirrored(distribution)

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path = self._save(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(mirrored, [5., 6.])
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

  def _restore_mirrored(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      mirrored = _make_mirrored(distribution)
      v = mirrored.values

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [7., 8.])

      # Restores the saved value of 3. to both variables.
      saver = saver_lib.Saver(var_list=[mirrored])
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreMirroredOneGraph(self, distribution):
    with self.cached_session() as sess:
      mirrored = _make_mirrored(distribution)
      v = mirrored  .values

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path, saver = self._save_return_saver(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(mirrored, [5., 6.])

      # Restores the saved value of 3. to both variables.
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveMirroredRestoreMirrored(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored(distribution)
    self._restore_mirrored(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveMirroredRestoreNormal(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreMirrored(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_normal()
    self._restore_mirrored(save_path, distribution)


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


if __name__ == "__main__":
  ds_test_util.main()
