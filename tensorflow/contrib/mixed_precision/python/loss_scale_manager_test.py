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
"""Tests for LossScaleManager classes.."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.mixed_precision.python import loss_scale_manager as lsm_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _GetExampleIter(inputs):
  dataset = dataset_ops.Dataset.from_tensor_slices(inputs)
  return dataset.make_one_shot_iterator()


class FixedLossScaleManagerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_basic(self):
    itr = _GetExampleIter([True] * 10 + [False] * 10)

    loss_scale = 1000
    lsm = lsm_lib.FixedLossScaleManager(loss_scale)
    update_fn = lambda: lsm.update_loss_scale(itr.get_next())

    self.evaluate(variables.global_variables_initializer())
    if not context.executing_eagerly():
      update_op = update_fn()
    for _ in range(10):
      if context.executing_eagerly():
        update_fn()
      else:
        self.evaluate(update_op)
      self.assertEqual(loss_scale, self.evaluate(lsm.get_loss_scale()))


class ExponentialUpdateLossScaleManagerTest(test.TestCase):

  def _test_helper(self,
                   inputs,
                   expected_outputs,
                   init_loss_scale=1,
                   incr_every_n_step=2,
                   decr_every_n_nan_or_inf=2):
    ratio = 2
    lsm = lsm_lib.ExponentialUpdateLossScaleManager(
        init_loss_scale=init_loss_scale,
        incr_every_n_steps=incr_every_n_step,
        decr_every_n_nan_or_inf=decr_every_n_nan_or_inf,
        incr_ratio=ratio,
        decr_ratio=1. / ratio)
    itr = _GetExampleIter(inputs)
    update_fn = lambda: lsm.update_loss_scale(itr.get_next())

    self.evaluate(variables.global_variables_initializer())
    actual_outputs = []

    if not context.executing_eagerly():
      update_op = update_fn()
    for _ in range(len(inputs)):
      if context.executing_eagerly():
        update_fn()
      else:
        self.evaluate(update_op)
      actual_outputs.append(self.evaluate(lsm.get_loss_scale()))
    self.assertEqual(actual_outputs, expected_outputs)

  @test_util.run_in_graph_and_eager_modes()
  def test_increase_every_n_steps(self):
    inputs = [True] * 6
    expected_outputs = [1, 2, 2, 4, 4, 8]
    self._test_helper(inputs, expected_outputs)

  @test_util.run_in_graph_and_eager_modes()
  def test_keep_increasing_until_capped(self):
    init_loss_scale = np.finfo(np.float32).max / 4 + 10
    max_float = np.finfo(np.float32).max

    inputs = [True] * 6
    # Output is capped the 2nd time it doubles.
    expected_outputs = [
        init_loss_scale, init_loss_scale * 2, init_loss_scale * 2, max_float,
        max_float, max_float
    ]

    self._test_helper(inputs, expected_outputs, init_loss_scale)

  @test_util.run_in_graph_and_eager_modes()
  def test_decrease_every_n_steps(self):
    inputs = [False] * 6
    init_loss_scale = 1024
    expected_outputs = [1024, 512, 512, 256, 256, 128]

    self._test_helper(inputs, expected_outputs, init_loss_scale)

  @test_util.run_in_graph_and_eager_modes()
  def test_keep_decreasing_until_one(self):
    inputs = [False] * 10
    init_loss_scale = 16
    expected_outputs = [16, 8, 8, 4, 4, 2, 2, 1, 1, 1]

    self._test_helper(inputs, expected_outputs, init_loss_scale)

  @test_util.run_in_graph_and_eager_modes()
  def test_incr_bad_step_clear_good_step(self):
    inputs = [True, True, True, False, True]
    expected_outputs = [1, 2, 2, 2, 2]
    self._test_helper(inputs, expected_outputs)

  @test_util.run_in_graph_and_eager_modes()
  def test_incr_good_step_does_not_clear_bad_step(self):
    inputs = [True, True, True, False, True, False]
    expected_outputs = [1, 2, 2, 2, 2, 1]
    self._test_helper(inputs, expected_outputs)

  @test_util.run_in_graph_and_eager_modes()
  def test_trigger_loss_scale_update_each_step(self):
    """Test when incr_every_n_step and decr_every_n_nan_or_inf is 1."""
    init_loss_scale = 1
    incr_every_n_step = 1
    decr_every_n_nan_or_inf = 1

    inputs = [True] * 3 + [False, True, True]
    expected_outputs = [2, 4, 8, 4, 8, 16]

    self._test_helper(inputs, expected_outputs, init_loss_scale,
                      incr_every_n_step, decr_every_n_nan_or_inf)

  @test_util.run_in_graph_and_eager_modes()
  def test_alternating_good_and_bad_gradients_trigger_each_step(self):
    init_loss_scale = 1
    incr_every_n_step = 1
    decr_every_n_nan_or_inf = 1

    inputs = [True, False] * 4 + [True]
    expected_outputs = [2, 1, 2, 1, 2, 1, 2, 1, 2]
    self._test_helper(inputs, expected_outputs, init_loss_scale,
                      incr_every_n_step, decr_every_n_nan_or_inf)

  @test_util.run_in_graph_and_eager_modes()
  def test_alternating_good_and_bad_gradients_trigger_incr_every_2steps(self):
    init_loss_scale = 32
    incr_every_n_step = 2
    decr_every_n_nan_or_inf = 1

    inputs = [True, False] * 3 + [True]
    expected_outputs = [32, 16, 16, 8, 8, 4, 4]
    self._test_helper(inputs, expected_outputs, init_loss_scale,
                      incr_every_n_step, decr_every_n_nan_or_inf)

  @test_util.run_in_graph_and_eager_modes()
  def test_random_mix_good_and_bad_gradients(self):
    init_loss_scale = 4
    inputs = [
        False, False, True, True, True, False, True, False, True, True, True,
        False
    ]
    expected_outputs = [4, 2, 2, 4, 4, 4, 4, 2, 2, 4, 4, 4]
    self._test_helper(inputs, expected_outputs, init_loss_scale)


if __name__ == "__main__":
  test.main()
