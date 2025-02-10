# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TFE_TensorHandleToNumpy."""

import numpy as np

from tensorflow.python.eager import pywrap_tensor_test_util as util
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util


class MyPythonObject:
  pass


def my_layer(x):
  y = x**2
  y.my_dynamic_attribute = MyPythonObject()
  return y


class PywrapTensorTest(test.TestCase):

  def testGetScalarOne(self):
    result = util.get_scalar_one()
    self.assertIsInstance(result, np.ndarray)
    self.assertAllEqual(result, 1.0)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def test_no_leak(self):
    x = constant_op.constant([1, 2, 3])
    layer = my_layer(x)
    for _ in range(int(1e2)):
      layer = my_layer(x)
    self.assertIsNotNone(layer)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def test_no_leak_cycles(self):
    for i in range(int(1e2)):
      # use multiply to avoid cached tensors.
      x = 1.0 * constant_op.constant([1.0, 1, 1, i])
      y = 1.0 * constant_op.constant([1.0, 1, 2, i])
      x.self_ref = lambda x: x
      x.y = y
      y.x = x

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def test_no_leak_shape(self):
    for i in range(int(1e2)):
      # use multiply to avoid cached tensors.
      x = 1.0 * constant_op.constant([3.0, 1, 1, i])
      x.shape.x = x

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def test_no_leak_handle_data(self):
    for i in range(int(1e2)):
      # use multiply to avoid cached tensors.
      x = 1.0 * constant_op.constant([4.0, 1, 1, i])
      x._handle_data = x


if __name__ == "__main__":
  test.main()
