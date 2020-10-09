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
"""Tests for trackable object SavedModel save."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save
from tensorflow.python.training.tracking import util


class _ModelWithOptimizerUsingDefun(util.Checkpoint):

  def __init__(self):
    self.dense = core.Dense(1)
    self.optimizer = adam.Adam(0.01)

  @def_function.function(
      input_signature=(tensor_spec.TensorSpec([None, 2], dtypes.float32),
                       tensor_spec.TensorSpec([None], dtypes.float32)),
  )
  def call(self, x, y):
    with backprop.GradientTape() as tape:
      loss = math_ops.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.dense.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {"loss": loss}


class MemoryTests(test.TestCase):

  def setUp(self):
    super(MemoryTests, self).setUp()
    self._model = _ModelWithOptimizerUsingDefun()

  @test_util.assert_no_garbage_created
  def test_no_reference_cycles(self):
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    self._model.call(x, y)
    if sys.version_info[0] < 3:
      # TODO(allenl): debug reference cycles in Python 2.x
      self.skipTest("This test only works in Python 3+. Reference cycles are "
                    "created in older Python versions.")
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(self._model, save_dir, self._model.call)


if __name__ == "__main__":
  test.main()
