# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.Assign*."""

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class AssignOpTest(test.TestCase):

  def _initAssignFetch(self, x, y, use_gpu):
    """Initialize a param to init and update it with y."""
    super(AssignOpTest, self).setUp()
    with test_util.device(use_gpu=use_gpu):
      p = variables.Variable(x)
      assign = state_ops.assign(p, y)
      self.evaluate(p.initializer)
      new_value = self.evaluate(assign)
      return self.evaluate(p), new_value

  def _initAssignAddFetch(self, x, y, use_gpu):
    """Initialize a param to init, and compute param += y."""
    with test_util.device(use_gpu=use_gpu):
      p = variables.Variable(x)
      add = state_ops.assign_add(p, y)
      self.evaluate(p.initializer)
      new_value = self.evaluate(add)
      return self.evaluate(p), new_value

  def _initAssignSubFetch(self, x, y, use_gpu):
    """Initialize a param to init, and compute param -= y."""
    with test_util.device(use_gpu=use_gpu):
      p = variables.Variable(x)
      sub = state_ops.assign_sub(p, y)
      self.evaluate(p.initializer)
      new_value = self.evaluate(sub)
      return self.evaluate(p), new_value

  def _testTypes(self, vals):
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
      x = np.zeros(vals.shape).astype(dtype)
      y = vals.astype(dtype)
      var_value, op_value = self._initAssignFetch(x, y, use_gpu=False)
      self.assertAllEqual(y, var_value)
      self.assertAllEqual(y, op_value)
      var_value, op_value = self._initAssignAddFetch(x, y, use_gpu=False)
      self.assertAllEqual(x + y, var_value)
      self.assertAllEqual(x + y, op_value)
      var_value, op_value = self._initAssignSubFetch(x, y, use_gpu=False)
      self.assertAllEqual(x - y, var_value)
      self.assertAllEqual(x - y, op_value)
      if test.is_built_with_gpu_support() and dtype in [np.float32, np.float64]:
        var_value, op_value = self._initAssignFetch(x, y, use_gpu=True)
        self.assertAllEqual(y, var_value)
        self.assertAllEqual(y, op_value)
        var_value, op_value = self._initAssignAddFetch(x, y, use_gpu=True)
        self.assertAllEqual(x + y, var_value)
        self.assertAllEqual(x + y, op_value)
        var_value, op_value = self._initAssignSubFetch(x, y, use_gpu=True)
        self.assertAllEqual(x - y, var_value)
        self.assertAllEqual(x - y, op_value)

  def testBasic(self):
    self._testTypes(np.arange(0, 20).reshape([4, 5]))

if __name__ == "__main__":
  test.main()
