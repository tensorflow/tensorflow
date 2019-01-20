# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensors module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.utils import tensors
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class TensorsTest(test.TestCase):

  def _simple_tensor_array(self):
    return tensor_array_ops.TensorArray(dtypes.int32, size=3)

  def _simple_tensor_list(self):
    return list_ops.empty_tensor_list(
        element_shape=constant_op.constant([1]), element_dtype=dtypes.int32)

  def _simple_list_of_tensors(self):
    return [constant_op.constant(1), constant_op.constant(2)]

  def test_is_tensor_array(self):
    self.assertTrue(tensors.is_tensor_array(self._simple_tensor_array()))
    self.assertFalse(tensors.is_tensor_array(self._simple_tensor_list()))
    self.assertFalse(tensors.is_tensor_array(constant_op.constant(1)))
    self.assertFalse(tensors.is_tensor_array(self._simple_list_of_tensors()))
    self.assertFalse(tensors.is_tensor_array(None))

  def test_is_tensor_list(self):
    self.assertFalse(tensors.is_tensor_list(self._simple_tensor_array()))
    self.assertTrue(tensors.is_tensor_list(self._simple_tensor_list()))
    self.assertFalse(tensors.is_tensor_list(constant_op.constant(1)))
    self.assertFalse(tensors.is_tensor_list(self._simple_list_of_tensors()))
    self.assertFalse(tensors.is_tensor_list(None))


if __name__ == '__main__':
  test.main()
