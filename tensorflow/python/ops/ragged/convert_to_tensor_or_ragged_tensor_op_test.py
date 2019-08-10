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
"""Tests for ragged_tensor.convert_to_tensor_or_ragged."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedConvertToTensorOrRaggedTensorTest(test_util.TensorFlowTestCase,
                                              parameterized.TestCase):

  #=============================================================================
  # Tests where the 'value' param is a RaggedTensor
  #=============================================================================
  @parameterized.parameters([
      dict(pylist=[[1, 2], [3]]),
      dict(pylist=[[1, 2], [3]], preferred_dtype=dtypes.float32),
      dict(pylist=[[1, 2], [3]], preferred_dtype=dtypes.string),
      # Note: Conversion of a single np.array is tested below. These tests
      # check nestings consisting of multiple or irregularily-shaped np.arrays.
      dict(
          pylist=[np.array([1, 2]), np.array([3])],
          preferred_dtype=dtypes.string),
      dict(pylist=np.array([[1, 2], [3]]), preferred_dtype=dtypes.float32),
      dict(pylist=np.array([[1, 2], [3]]), preferred_dtype=dtypes.string),
      dict(
          pylist=[np.array([[1], np.array([2])]), [np.array([3])]],
          preferred_dtype=dtypes.float32),
      dict(pylist=[np.array(1)], preferred_dtype=dtypes.string),
  ])
  def testConvertRaggedTensor(self, pylist, dtype=None, preferred_dtype=None):
    rt = ragged_factory_ops.constant(pylist)
    converted = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        rt, dtype, preferred_dtype)
    self.assertIs(converted, rt)

  @parameterized.parameters([
      dict(
          pylist=[[1, 2], [3, 4]],
          dtype=dtypes.float32,
          message=('Tensor conversion requested dtype float32 for '
                   'RaggedTensor with dtype int32')),
      dict(
          pylist=np.array([[1, 2], [3, 4]]),
          dtype=dtypes.float32,
          message=('Tensor conversion requested dtype float32 for '
                   'RaggedTensor with dtype int32')),
      dict(
          pylist=[[1, 2], [3, 4]],
          dtype=dtypes.string,
          message=('Tensor conversion requested dtype string for '
                   'RaggedTensor with dtype .*')),
  ])
  def testConvertRaggedTensorError(self,
                                   pylist,
                                   message,
                                   dtype=None,
                                   preferred_dtype=None):
    rt = ragged_factory_ops.constant(pylist)

    with self.assertRaisesRegexp(ValueError, message):
      ragged_tensor.convert_to_tensor_or_ragged_tensor(rt, dtype,
                                                       preferred_dtype)

  #=============================================================================
  # Tests where the 'value' param is a RaggedTensorValue
  #=============================================================================
  @parameterized.parameters(
      [
          dict(
              value=ragged_factory_ops.constant_value([[1, 2], [3]],
                                                      dtype=np.int32),
              expected_dtype=dtypes.int32),
          dict(
              value=ragged_factory_ops.constant_value([[b'a', b'b'], [b'c']]),
              expected_dtype=dtypes.string),
          dict(
              value=ragged_factory_ops.constant_value([[1, 2], [3]],
                                                      dtype=np.int32),
              dtype=dtypes.float32,
              expected_dtype=dtypes.float32),
          dict(
              value=ragged_factory_ops.constant_value([[1, 2], [3]],
                                                      dtype=np.int32),
              preferred_dtype=dtypes.float32,
              expected_dtype=dtypes.float32),
          dict(
              value=ragged_factory_ops.constant_value([[1, 2], [3]],
                                                      dtype=np.int32),
              preferred_dtype=dtypes.string,
              expected_dtype=dtypes.int32),
      ])
  def testConvertRaggedTensorValue(self,
                                   value,
                                   dtype=None,
                                   preferred_dtype=None,
                                   expected_dtype=None):
    if expected_dtype is None:
      expected_dtype = value.dtype if dtype is None else dtype
    converted = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        value, dtype, preferred_dtype)
    self.assertEqual(value.ragged_rank, converted.ragged_rank)
    self.assertEqual(dtypes.as_dtype(expected_dtype), converted.dtype)
    self.assertAllEqual(value, converted)

  @parameterized.parameters([
      dict(
          value=ragged_factory_ops.constant_value([['a', 'b'], ['c']],
                                                  dtype=str),
          dtype=dtypes.int32,
          message=r"invalid literal for int\(\) with base 10: 'a'"),
  ])
  def testConvertRaggedTensorValueError(self,
                                        value,
                                        message,
                                        dtype=None,
                                        preferred_dtype=None):
    with self.assertRaisesRegexp(ValueError, message):
      ragged_tensor.convert_to_tensor_or_ragged_tensor(value, dtype,
                                                       preferred_dtype)

  #=============================================================================
  # Tests where the 'value' param is a Tensor
  #=============================================================================
  @parameterized.parameters([
      dict(pylist=[[1, 2], [3, 4]]),
      dict(pylist=[[1, 2], [3, 4]], preferred_dtype=dtypes.float32),
      dict(pylist=[[1, 2], [3, 4]], preferred_dtype=dtypes.string),
  ])
  def testConvertTensor(self, pylist, dtype=None, preferred_dtype=None):
    tensor = constant_op.constant(pylist)
    converted = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        tensor, dtype, preferred_dtype)
    self.assertIs(tensor, converted)

  @parameterized.parameters([
      dict(
          pylist=[[1, 2], [3, 4]],
          dtype=dtypes.float32,
          message=('Tensor conversion requested dtype float32 for '
                   'Tensor with dtype int32')),
      dict(
          pylist=[[1, 2], [3, 4]],
          dtype=dtypes.string,
          message=('Tensor conversion requested dtype string for '
                   'Tensor with dtype int32')),
  ])
  def testConvertTensorError(self,
                             pylist,
                             message,
                             dtype=None,
                             preferred_dtype=None):
    tensor = constant_op.constant(pylist)
    with self.assertRaisesRegexp(ValueError, message):
      ragged_tensor.convert_to_tensor_or_ragged_tensor(tensor, dtype,
                                                       preferred_dtype)

  #=============================================================================
  # Tests where the 'value' param is a np.array
  #=============================================================================
  @parameterized.parameters([
      dict(
          value=np.array([[1, 2], [3, 4]], dtype=np.int32),
          expected_dtype=dtypes.int32),
      dict(
          value=np.array([[b'a', b'b'], [b'c', b'd']]),
          expected_dtype=dtypes.string),
      dict(
          value=np.array([[1, 2], [3, 4]], dtype=np.int32),
          dtype=dtypes.float32,
          expected_dtype=dtypes.float32),
      dict(
          value=np.array([[1, 2], [3, 4]], dtype=np.int32),
          preferred_dtype=dtypes.float32,
          expected_dtype=dtypes.float32),
      dict(
          value=np.array([[1, 2], [3, 4]], dtype=np.int32),
          preferred_dtype=dtypes.string,
          expected_dtype=dtypes.int32),
  ])
  def testConvertNumpyArray(self,
                            value,
                            dtype=None,
                            preferred_dtype=None,
                            expected_dtype=None):
    if expected_dtype is None:
      expected_dtype = value.dtype if dtype is None else dtype
    converted = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        value, dtype, preferred_dtype)
    self.assertEqual(dtypes.as_dtype(expected_dtype), converted.dtype)
    self.assertAllEqual(value, converted)

  @parameterized.parameters([
      dict(
          value=np.array([['a', 'b'], ['c', 'd']], dtype=str),
          dtype=dtypes.int32,
          message=r"invalid literal for int\(\) with base 10: 'a'"),
  ])
  def testConvertNumpyArrayError(self,
                                 value,
                                 message,
                                 dtype=None,
                                 preferred_dtype=None):
    with self.assertRaisesRegexp(ValueError, message):
      ragged_tensor.convert_to_tensor_or_ragged_tensor(value, dtype,
                                                       preferred_dtype)


if __name__ == '__main__':
  googletest.main()
