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
"""Tests for eager execution_callbacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import execution_callbacks
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def log_zero():
  """Computes `log(0.0)`."""
  return math_ops.log(constant_op.constant(0.))


class ExecutionCallbacksTest(test.TestCase):

  def test_errstate_inf_raise(self):
    with execution_callbacks.errstate(inf_or_nan=execution_callbacks.RAISE):
      with self.assertRaises(execution_callbacks.InfOrNanError):
        log_zero()

  def test_errstate_inf_ignore(self):
    with execution_callbacks.errstate(inf_or_nan=execution_callbacks.IGNORE):
      self.assertEqual(-float("inf"), log_zero().numpy())

  def test_errstate_nesting(self):
    with execution_callbacks.errstate(inf_or_nan=execution_callbacks.RAISE):
      with execution_callbacks.errstate(inf_or_nan=execution_callbacks.IGNORE):
        self.assertEqual(-float("inf"), log_zero().numpy())

      with self.assertRaises(execution_callbacks.InfOrNanError):
        log_zero()


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
