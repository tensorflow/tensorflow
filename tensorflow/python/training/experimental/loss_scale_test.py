# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for LossScale classes.."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module

# TODO(reedwm): Create test case using multiple graphs

# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribution_strategy_context.get_strategy


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


TESTCASES = ({
    'testcase_name': 'base',
    'strategy_fn': default_strategy_fn
}, {
    'testcase_name': 'distribute',
    'strategy_fn': create_mirrored_strategy
})


class FixedLossScaleTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_basic(self):
    loss_scale_value = 1000
    loss_scale = loss_scale_module.FixedLossScale(loss_scale_value)

    update_op, should_apply = loss_scale.update([constant_op.constant(0.)])
    self.evaluate(update_op)
    # should_apply should be a bool instead of a tensor, so that a tf.cond does
    # not have to be built in the graph by the caller.
    self.assertIsInstance(should_apply, bool)
    self.assertTrue(should_apply)
    self.assertEqual(loss_scale_value, self.evaluate(loss_scale()))

    update_op, should_apply = loss_scale.update(
        [constant_op.constant(float('NaN'))])
    self.evaluate(update_op)
    self.assertIsInstance(should_apply, bool)
    self.assertTrue(should_apply)
    self.assertEqual(loss_scale_value, self.evaluate(loss_scale()))

  @test_util.run_in_graph_and_eager_modes
  def test_serialization(self):
    loss_scale = loss_scale_module.get(123)
    config = loss_scale.get_config()
    loss_scale = loss_scale_module.FixedLossScale.from_config(config)
    self.assertEqual(self.evaluate(loss_scale()), 123.)

  @test_util.run_in_graph_and_eager_modes
  def test_call_type(self):
    scalar = loss_scale_module.FixedLossScale(123)
    self.assertIsInstance(scalar(), ops.Tensor)

  @test_util.run_in_graph_and_eager_modes
  def test_repr(self):
    loss_scale = loss_scale_module.FixedLossScale(123)
    self.assertEqual(repr(loss_scale), 'FixedLossScale(123.0)')


def _get_example_iter(inputs):
  dataset = dataset_ops.Dataset.from_tensor_slices(inputs)
  return dataset_ops.make_one_shot_iterator(dataset)


class DynamicLossScaleTest(test.TestCase, parameterized.TestCase):

  def _get_tensor(self, is_finite):
    tensor = cond.cond(is_finite, lambda: 1., lambda: float('NaN'))

    if not distribution_strategy_context.has_strategy():
      return tensor

    def get():
      rep_id = (
          distribution_strategy_context.get_replica_context()
          .replica_id_in_sync_group)
      return cond.cond(
          math_ops.equal(rep_id, 0), lambda: tensor, lambda: 1.)

    distribution = distribution_strategy_context.get_strategy()
    return distribution.extended.call_for_each_replica(get)

  def _test_helper(self,
                   inputs,
                   expected_outputs,
                   initial_loss_scale=1.,
                   increment_period=2,
                   multiplier=2):
    loss_scale = loss_scale_module.DynamicLossScale(
        initial_loss_scale=initial_loss_scale,
        increment_period=increment_period,
        multiplier=multiplier)
    itr = _get_example_iter(inputs)

    def update():
      is_finite = itr.get_next()
      grad = self._get_tensor(is_finite)
      update_op, should_apply_gradients = loss_scale.update([grad])
      assert_op = check_ops.assert_equal(should_apply_gradients, is_finite)
      if context.executing_eagerly():
        return
      with ops.control_dependencies([assert_op]):
        return array_ops.identity(update_op)

    actual_outputs = []

    if not context.executing_eagerly():
      update_op = update()
      self.evaluate(variables.global_variables_initializer())
    for _ in range(len(inputs)):
      if context.executing_eagerly():
        update()
      else:
        self.evaluate(update_op)
      actual_outputs.append(self.evaluate(loss_scale()))
    self.assertEqual(actual_outputs, expected_outputs)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_increase(self, strategy_fn):
    with strategy_fn().scope():
      inputs = [True] * 6
      expected_outputs = [1, 2, 2, 4, 4, 8]
      self._test_helper(inputs, expected_outputs)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_keep_increasing_until_capped(self, strategy_fn):
    with strategy_fn().scope():
      init_loss_scale = np.finfo(np.float32).max / 4
      max_float = np.finfo(np.float32).max

      inputs = [True] * 6
      # Output is capped the 2nd time it doubles.
      expected_outputs = [
          init_loss_scale, init_loss_scale * 2, init_loss_scale * 2, max_float,
          max_float, max_float
      ]

      self._test_helper(inputs, expected_outputs, init_loss_scale)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_decrease_every_step(self, strategy_fn):
    with strategy_fn().scope():
      inputs = [False] * 6
      init_loss_scale = 1024
      expected_outputs = [512, 256, 128, 64, 32, 16]

    self._test_helper(inputs, expected_outputs, init_loss_scale)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_keep_decreasing_until_one(self, strategy_fn):
    with strategy_fn().scope():
      inputs = [False] * 6
      init_loss_scale = 16
      expected_outputs = [8, 4, 2, 1, 1, 1]

      self._test_helper(inputs, expected_outputs, init_loss_scale)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_nan_clear_good_step(self, strategy_fn):
    with strategy_fn().scope():
      inputs = [True, True, True, False, True]
      expected_outputs = [1, 2, 2, 1, 1]
      self._test_helper(inputs, expected_outputs)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_trigger_loss_scale_update_each_step(self, strategy_fn):
    with strategy_fn().scope():
      init_loss_scale = 1
      increment_period = 1

      inputs = [True] * 3 + [False, True, True]
      expected_outputs = [2, 4, 8, 4, 8, 16]

      self._test_helper(inputs, expected_outputs, init_loss_scale,
                        increment_period)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_alternating_good_and_bad_gradients_trigger_each_step(
      self, strategy_fn):
    with strategy_fn().scope():
      init_loss_scale = 1
      increment_period = 1

      inputs = [True, False] * 4 + [True]
      expected_outputs = [2, 1, 2, 1, 2, 1, 2, 1, 2]
      self._test_helper(inputs, expected_outputs, init_loss_scale,
                        increment_period)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_alternating_good_and_bad_gradients_trigger_every_other_step(
      self, strategy_fn):
    with strategy_fn().scope():
      init_loss_scale = 32
      increment_period = 2

      inputs = [True, False] * 3 + [True]
      expected_outputs = [32, 16, 16, 8, 8, 4, 4]
      self._test_helper(inputs, expected_outputs, init_loss_scale,
                        increment_period)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_nondefault_multiplier(self, strategy_fn):
    with strategy_fn().scope():
      init_loss_scale = 4
      multiplier = 3
      inputs = [True, True, False, True, True]
      expected_outputs = [4, 12, 4, 4, 12]
      self._test_helper(
          inputs, expected_outputs, init_loss_scale, multiplier=multiplier)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_random_mix_good_and_bad_gradients(self, strategy_fn):
    with strategy_fn().scope():
      init_loss_scale = 4
      inputs = [
          False, True, True, True, False, True, False, True, True, True, False
      ]
      expected_outputs = [2, 2, 4, 4, 2, 2, 1, 1, 2, 2, 1]
      self._test_helper(inputs, expected_outputs, init_loss_scale)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_single_tensor_gradient(self, strategy_fn):
    with strategy_fn().scope():
      loss_scale = loss_scale_module.DynamicLossScale()
      grad = constant_op.constant(4.0)
      _, should_apply = loss_scale.update(grad)
      self.assertTrue(self.evaluate(should_apply))

  @test_util.run_in_graph_and_eager_modes
  def test_serialization(self):
    loss_scale = loss_scale_module.DynamicLossScale(
        initial_loss_scale=1, increment_period=2, multiplier=3)
    config = loss_scale.get_config()
    loss_scale = loss_scale_module.DynamicLossScale.from_config(config)
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(self.evaluate(loss_scale()), 1)
    self.assertEqual(loss_scale.increment_period, 2)
    self.assertEqual(loss_scale.multiplier, 3)

  @test_util.run_in_graph_and_eager_modes
  def test_update_with_none_gradients(self):
    loss_scale = loss_scale_module.DynamicLossScale()
    loss_scale.update([None])

  @test_util.run_in_graph_and_eager_modes
  def test_get(self):
    scalar = loss_scale_module.get('dynamic')
    scalar2 = loss_scale_module.DynamicLossScale()
    self.assertEqual(scalar.initial_loss_scale, scalar2.initial_loss_scale)
    self.assertEqual(scalar.increment_period, scalar2.increment_period)
    self.assertEqual(scalar.multiplier, scalar2.multiplier)

  @test_util.run_in_graph_and_eager_modes
  def test_call_type(self):
    scalar = loss_scale_module.DynamicLossScale()
    self.assertIsInstance(scalar(), ops.Tensor)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_repr(self, strategy_fn):
    with strategy_fn().scope():
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=1, increment_period=2, multiplier=3)
      if context.executing_eagerly():
        self.assertEqual(repr(loss_scale),
                         'DynamicLossScale(current_loss_scale=1.0, '
                         'num_good_steps=0, initial_loss_scale=1.0, '
                         'increment_period=2, multiplier=3.0)')
      else:
        self.assertEqual(repr(loss_scale),
                         'DynamicLossScale(initial_loss_scale=1.0, '
                         'increment_period=2, multiplier=3.0)')


if __name__ == '__main__':
  test.main()
