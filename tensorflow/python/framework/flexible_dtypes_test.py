# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for tensorflow.python.framework.flexible_dtypes."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import variables
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.platform import test as tf_test


PromoMode = ops.PromoMode
DtypeConversionTestEnv = weak_tensor_test_util.DtypeConversionTestEnv

_ALL_INPUT_TYPES = [
    constant_op.constant(True, dtype=dtypes.bool),
    constant_op.constant(1, dtype=dtypes.uint8),
    constant_op.constant(1, dtype=dtypes.uint16),
    constant_op.constant(1, dtype=dtypes.uint32),
    constant_op.constant(1, dtype=dtypes.uint64),
    constant_op.constant(1, dtype=dtypes.int8),
    constant_op.constant(1, dtype=dtypes.int16),
    constant_op.constant(1, dtype=dtypes.int32),
    constant_op.constant(1, dtype=dtypes.int64),
    constant_op.constant(1, dtype=dtypes.bfloat16),
    constant_op.constant(1, dtype=dtypes.float16),
    constant_op.constant(1, dtype=dtypes.float32),
    constant_op.constant(1, dtype=dtypes.float64),
    constant_op.constant(1, dtype=dtypes.complex64),
    constant_op.constant(1, dtype=dtypes.complex128),
    constant_op.constant(1),
    np.array(True, dtype=np.bool_),
    np.array(1, dtype=np.uint8),
    np.array(1, dtype=np.uint16),
    np.array(1, dtype=np.uint32),
    np.array(1, dtype=np.uint64),
    np.array(1, dtype=np.int8),
    np.array(1, dtype=np.int16),
    np.array(1, dtype=np.int32),
    np.array(1, dtype=np.int64),
    np.array(1, dtype=np.float16),
    np.array(1, dtype=np.float32),
    np.array(1, dtype=np.float64),
    np.array(1, dtype=np.complex64),
    np.array(1, dtype=np.complex128),
    np.array(1),
    np.uint8(1),
    np.uint16(1),
    np.uint32(1),
    np.uint64(1),
    np.int8(1),
    np.int16(1),
    np.int32(1),
    np.int64(1),
    np.float16(1),
    np.float32(1),
    np.float64(1),
    np.complex64(1),
    np.complex128(1),
    np.int_(),
    1,
    1.0,
    1.0j,
]


class DtypesUtilTest(tf_test.TestCase, parameterized.TestCase):

  # Test all possible TF dtypes in ALL mode.
  @parameterized.parameters(
      (dtypes.bool, dtypes.bool, (dtypes.bool, False)),
      (dtypes.bool, dtypes.uint8, (dtypes.uint8, False)),
      (dtypes.bool, dtypes.uint16, (dtypes.uint16, False)),
      (dtypes.bool, dtypes.uint32, (dtypes.uint32, False)),
      (dtypes.bool, dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.bool, dtypes.int8, (dtypes.int8, False)),
      (dtypes.bool, dtypes.int16, (dtypes.int16, False)),
      (dtypes.bool, dtypes.int32, (dtypes.int32, False)),
      (dtypes.bool, dtypes.int64, (dtypes.int64, False)),
      (dtypes.bool, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.bool, dtypes.float16, (dtypes.float16, False)),
      (dtypes.bool, dtypes.float32, (dtypes.float32, False)),
      (dtypes.bool, dtypes.float64, (dtypes.float64, False)),
      (dtypes.bool, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.bool, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.uint8, dtypes.uint8, (dtypes.uint8, False)),
      (dtypes.uint8, dtypes.uint16, (dtypes.uint16, False)),
      (dtypes.uint8, dtypes.uint32, (dtypes.uint32, False)),
      (dtypes.uint8, dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.uint8, dtypes.int8, (dtypes.int16, False)),
      (dtypes.uint8, dtypes.int16, (dtypes.int16, False)),
      (dtypes.uint8, dtypes.int32, (dtypes.int32, False)),
      (dtypes.uint8, dtypes.int64, (dtypes.int64, False)),
      (dtypes.uint8, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.uint8, dtypes.float16, (dtypes.float16, False)),
      (dtypes.uint8, dtypes.float32, (dtypes.float32, False)),
      (dtypes.uint8, dtypes.float64, (dtypes.float64, False)),
      (dtypes.uint8, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.uint8, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.uint16, dtypes.uint16, (dtypes.uint16, False)),
      (dtypes.uint16, dtypes.uint32, (dtypes.uint32, False)),
      (dtypes.uint16, dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.uint16, dtypes.int8, (dtypes.int32, False)),
      (dtypes.uint16, dtypes.int16, (dtypes.int32, False)),
      (dtypes.uint16, dtypes.int32, (dtypes.int32, False)),
      (dtypes.uint16, dtypes.int64, (dtypes.int64, False)),
      (dtypes.uint16, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.uint16, dtypes.float16, (dtypes.float16, False)),
      (dtypes.uint16, dtypes.float32, (dtypes.float32, False)),
      (dtypes.uint16, dtypes.float64, (dtypes.float64, False)),
      (dtypes.uint16, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.uint16, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.uint32, dtypes.uint32, (dtypes.uint32, False)),
      (dtypes.uint32, dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.uint32, dtypes.int8, (dtypes.int64, False)),
      (dtypes.uint32, dtypes.int16, (dtypes.int64, False)),
      (dtypes.uint32, dtypes.int32, (dtypes.int64, False)),
      (dtypes.uint32, dtypes.int64, (dtypes.int64, False)),
      (dtypes.uint32, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.uint32, dtypes.float16, (dtypes.float16, False)),
      (dtypes.uint32, dtypes.float32, (dtypes.float32, False)),
      (dtypes.uint32, dtypes.float64, (dtypes.float64, False)),
      (dtypes.uint32, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.uint32, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.uint64, dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.uint64, dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.uint64, dtypes.int16, (dtypes.float64, True)),
      (dtypes.uint64, dtypes.int32, (dtypes.float64, True)),
      (dtypes.uint64, dtypes.int64, (dtypes.float64, True)),
      (dtypes.uint64, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.uint64, dtypes.float16, (dtypes.float16, False)),
      (dtypes.uint64, dtypes.float32, (dtypes.float32, False)),
      (dtypes.uint64, dtypes.float64, (dtypes.float64, False)),
      (dtypes.uint64, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.uint64, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.int8, dtypes.int8, (dtypes.int8, False)),
      (dtypes.int8, dtypes.int16, (dtypes.int16, False)),
      (dtypes.int8, dtypes.int32, (dtypes.int32, False)),
      (dtypes.int8, dtypes.int64, (dtypes.int64, False)),
      (dtypes.int8, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.int8, dtypes.float16, (dtypes.float16, False)),
      (dtypes.int8, dtypes.float32, (dtypes.float32, False)),
      (dtypes.int8, dtypes.float64, (dtypes.float64, False)),
      (dtypes.int8, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.int8, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.int16, dtypes.int16, (dtypes.int16, False)),
      (dtypes.int16, dtypes.int32, (dtypes.int32, False)),
      (dtypes.int16, dtypes.int64, (dtypes.int64, False)),
      (dtypes.int16, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.int16, dtypes.float16, (dtypes.float16, False)),
      (dtypes.int16, dtypes.float32, (dtypes.float32, False)),
      (dtypes.int16, dtypes.float64, (dtypes.float64, False)),
      (dtypes.int16, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.int16, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.int32, dtypes.int32, (dtypes.int32, False)),
      (dtypes.int32, dtypes.int64, (dtypes.int64, False)),
      (dtypes.int32, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.int32, dtypes.float16, (dtypes.float16, False)),
      (dtypes.int32, dtypes.float32, (dtypes.float32, False)),
      (dtypes.int32, dtypes.float64, (dtypes.float64, False)),
      (dtypes.int32, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.int32, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.int64, dtypes.int64, (dtypes.int64, False)),
      (dtypes.int64, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.int64, dtypes.float16, (dtypes.float16, False)),
      (dtypes.int64, dtypes.float32, (dtypes.float32, False)),
      (dtypes.int64, dtypes.float64, (dtypes.float64, False)),
      (dtypes.int64, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.int64, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.bfloat16, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.bfloat16, dtypes.float16, (dtypes.float32, False)),
      (dtypes.bfloat16, dtypes.float32, (dtypes.float32, False)),
      (dtypes.bfloat16, dtypes.float64, (dtypes.float64, False)),
      (dtypes.bfloat16, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.bfloat16, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.float16, dtypes.float16, (dtypes.float16, False)),
      (dtypes.float16, dtypes.float32, (dtypes.float32, False)),
      (dtypes.float16, dtypes.float64, (dtypes.float64, False)),
      (dtypes.float16, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.float16, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.float32, dtypes.float32, (dtypes.float32, False)),
      (dtypes.float32, dtypes.float64, (dtypes.float64, False)),
      (dtypes.float32, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.float32, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.float64, dtypes.float64, (dtypes.float64, False)),
      (dtypes.float64, dtypes.complex64, (dtypes.complex128, False)),
      (dtypes.float64, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.complex64, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.complex64, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.complex128, dtypes.complex128, (dtypes.complex128, False)),
  )
  def testResultTypeTFAndTF(self, a_dtype, b_dtype, res_dtype):
    with DtypeConversionTestEnv('all'):
      input_a = (
          constant_op.constant(1, dtype=a_dtype)
          if a_dtype != dtypes.bool
          else constant_op.constant(True)
      )
      input_b = (
          constant_op.constant(2, dtype=b_dtype)
          if b_dtype != dtypes.bool
          else constant_op.constant(False)
      )
      self.assertEqual(
          flexible_dtypes.result_type(input_a, input_b),
          res_dtype,
      )

  # Test NP types dtype inference.
  @parameterized.parameters(
      (np.bool_, np.uint8, dtypes.uint8),
      (np.uint8, np.int8, dtypes.int16),
      (np.uint16, np.int8, dtypes.int32),
      (np.uint32, np.float16, dtypes.float16),
      (np.uint64, np.float16, dtypes.float16),
      (np.uint64, np.complex64, dtypes.complex64),
      (np.int8, np.float16, dtypes.float16),
      (np.int16, np.float16, dtypes.float16),
      (np.int32, np.complex64, dtypes.complex64),
      (np.int64, np.float16, dtypes.float16),
      (np.float16, np.complex64, dtypes.complex64),
      (np.float32, np.float64, dtypes.float64),
      (np.float64, np.complex128, dtypes.complex128),
      (np.complex64, np.complex64, dtypes.complex64),
      (np.complex64, np.complex128, dtypes.complex128),
  )
  def testResultTypeNPAndNP(self, a_dtype, b_dtype, res_dtype):
    with DtypeConversionTestEnv('all'):
      self.assertEqual(
          flexible_dtypes.result_type(
              np.array(1, dtype=a_dtype), np.array(2, dtype=b_dtype)
          ),
          (res_dtype, False),
      )

  # Test np.array with default types.
  # np.array(int) => i64
  # np.array(float) => f64
  # np.array(complex) => complex128
  @parameterized.parameters(
      (1, np.bool_, (dtypes.int64, False)),
      (1, np.uint8, (dtypes.int64, False)),
      (1, np.uint16, (dtypes.int64, False)),
      (1, np.uint32, (dtypes.int64, False)),
      (1, np.uint64, (dtypes.float64, True)),
      (1, np.int8, (dtypes.int64, False)),
      (1, np.int16, (dtypes.int64, False)),
      (1, np.int32, (dtypes.int64, False)),
      (1, np.int64, (dtypes.int64, False)),
      (1, np.float16, (dtypes.float16, False)),
      (1, np.float32, (dtypes.float32, False)),
      (1, np.float64, (dtypes.float64, False)),
      (1, np.complex64, (dtypes.complex64, False)),
      (1, np.complex128, (dtypes.complex128, False)),
      (1.0, np.bool_, (dtypes.float64, False)),
      (1.0, np.uint8, (dtypes.float64, False)),
      (1.0, np.uint16, (dtypes.float64, False)),
      (1.0, np.uint32, (dtypes.float64, False)),
      (1.0, np.uint64, (dtypes.float64, False)),
      (1.0, np.int8, (dtypes.float64, False)),
      (1.0, np.int16, (dtypes.float64, False)),
      (1.0, np.int32, (dtypes.float64, False)),
      (1.0, np.int64, (dtypes.float64, False)),
      (1.0, np.float16, (dtypes.float64, False)),
      (1.0, np.float32, (dtypes.float64, False)),
      (1.0, np.float64, (dtypes.float64, False)),
      (1.0, np.complex64, (dtypes.complex128, False)),
      (1.0, np.complex128, (dtypes.complex128, False)),
      (1.0j, np.bool_, (dtypes.complex128, False)),
      (1.0j, np.uint8, (dtypes.complex128, False)),
      (1.0j, np.uint16, (dtypes.complex128, False)),
      (1.0j, np.uint32, (dtypes.complex128, False)),
      (1.0j, np.uint64, (dtypes.complex128, False)),
      (1.0j, np.int8, (dtypes.complex128, False)),
      (1.0j, np.int16, (dtypes.complex128, False)),
      (1.0j, np.int32, (dtypes.complex128, False)),
      (1.0j, np.int64, (dtypes.complex128, False)),
      (1.0j, np.float16, (dtypes.complex128, False)),
      (1.0j, np.float32, (dtypes.complex128, False)),
      (1.0j, np.float64, (dtypes.complex128, False)),
      (1.0j, np.complex64, (dtypes.complex128, False)),
      (1.0j, np.complex128, (dtypes.complex128, False)),
  )
  def testResultTypeNPDefaultArray(self, array_in, dtype, res_dtype):
    with DtypeConversionTestEnv('all'):
      self.assertEqual(
          flexible_dtypes.result_type(
              np.array(array_in), np.array(1, dtype=dtype)
          ),
          res_dtype,
      )

  # Test Python int inputs. Note that Python int literals are converted into
  # weak int32 type.
  @parameterized.parameters(
      (dtypes.bool, (dtypes.int32, True)),
      (dtypes.uint8, (dtypes.uint8, False)),
      (dtypes.uint16, (dtypes.uint16, False)),
      (dtypes.uint32, (dtypes.uint32, False)),
      (dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.int8, (dtypes.int8, False)),
      (dtypes.int16, (dtypes.int16, False)),
      (dtypes.int32, (dtypes.int32, False)),
      (dtypes.int64, (dtypes.int64, False)),
      (dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.float16, (dtypes.float16, False)),
      (dtypes.float32, (dtypes.float32, False)),
      (dtypes.float64, (dtypes.float64, False)),
      (dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.complex128, (dtypes.complex128, False)),
  )
  def testResultTypePythonInt(self, input_dtype, res_dtype):
    with DtypeConversionTestEnv('all'):
      t_input = (
          constant_op.constant(2, dtype=input_dtype)
          if input_dtype != dtypes.bool
          else constant_op.constant(True)
      )
      self.assertEqual(flexible_dtypes.result_type(1, t_input), res_dtype)

  # Test Python float inputs. Note that Python float literals are converted into
  # weak float32 type.
  @parameterized.parameters(
      (dtypes.bool, (dtypes.float32, True)),
      (dtypes.uint8, (dtypes.float64, True)),
      (dtypes.uint16, (dtypes.float64, True)),
      (dtypes.uint32, (dtypes.float64, True)),
      (dtypes.uint64, (dtypes.float64, True)),
      (dtypes.int8, (dtypes.float64, True)),
      (dtypes.int16, (dtypes.float64, True)),
      (dtypes.int32, (dtypes.float64, True)),
      (dtypes.int64, (dtypes.float64, True)),
      (dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.float16, (dtypes.float16, False)),
      (dtypes.float32, (dtypes.float32, False)),
      (dtypes.float64, (dtypes.float64, False)),
      (dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.complex128, (dtypes.complex128, False)),
  )
  def testResultTypePythonFloat(self, input_dtype, res_dtype):
    with DtypeConversionTestEnv('all'):
      t_input = (
          constant_op.constant(2, dtype=input_dtype)
          if input_dtype != dtypes.bool
          else constant_op.constant(True)
      )
      self.assertEqual(flexible_dtypes.result_type(1.0, t_input), res_dtype)

  # Test Python complex inputs. Note that Python complex literals are converted
  # into weak complex128 type.
  @parameterized.parameters(
      (dtypes.bool, (dtypes.complex128, True)),
      (dtypes.uint8, (dtypes.complex128, True)),
      (dtypes.uint16, (dtypes.complex128, True)),
      (dtypes.uint32, (dtypes.complex128, True)),
      (dtypes.uint64, (dtypes.complex128, True)),
      (dtypes.int8, (dtypes.complex128, True)),
      (dtypes.int16, (dtypes.complex128, True)),
      (dtypes.int32, (dtypes.complex128, True)),
      (dtypes.int64, (dtypes.complex128, True)),
      (dtypes.bfloat16, (dtypes.complex64, False)),
      (dtypes.float16, (dtypes.complex64, False)),
      (dtypes.float32, (dtypes.complex64, False)),
      (dtypes.float64, (dtypes.complex128, False)),
      (dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.complex128, (dtypes.complex128, False)),
  )
  def testResultTypePythonComplex(self, input_dtype, res_dtype):
    with DtypeConversionTestEnv('all'):
      t_input = (
          constant_op.constant(2, dtype=input_dtype)
          if input_dtype != dtypes.bool
          else constant_op.constant(True)
      )
      self.assertEqual(flexible_dtypes.result_type(1.0j, t_input), res_dtype)

  # Test every possible weak type + TF dtype.
  @parameterized.parameters(
      (dtypes.int32, dtypes.bool, (dtypes.int32, True)),
      (dtypes.int32, dtypes.uint8, (dtypes.uint8, False)),
      (dtypes.int32, dtypes.uint16, (dtypes.uint16, False)),
      (dtypes.int32, dtypes.uint32, (dtypes.uint32, False)),
      (dtypes.int32, dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.int32, dtypes.int8, (dtypes.int8, False)),
      (dtypes.int32, dtypes.int16, (dtypes.int16, False)),
      (dtypes.int32, dtypes.int32, (dtypes.int32, False)),
      (dtypes.int32, dtypes.int64, (dtypes.int64, False)),
      (dtypes.int32, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.int32, dtypes.float16, (dtypes.float16, False)),
      (dtypes.int32, dtypes.float32, (dtypes.float32, False)),
      (dtypes.int32, dtypes.float64, (dtypes.float64, False)),
      (dtypes.int32, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.int32, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.int64, dtypes.bool, (dtypes.int64, True)),
      (dtypes.int64, dtypes.uint8, (dtypes.uint8, False)),
      (dtypes.int64, dtypes.uint16, (dtypes.uint16, False)),
      (dtypes.int64, dtypes.uint32, (dtypes.uint32, False)),
      (dtypes.int64, dtypes.uint64, (dtypes.uint64, False)),
      (dtypes.int64, dtypes.int8, (dtypes.int8, False)),
      (dtypes.int64, dtypes.int16, (dtypes.int16, False)),
      (dtypes.int64, dtypes.int32, (dtypes.int32, False)),
      (dtypes.int64, dtypes.int64, (dtypes.int64, False)),
      (dtypes.int32, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.int64, dtypes.float16, (dtypes.float16, False)),
      (dtypes.int64, dtypes.float32, (dtypes.float32, False)),
      (dtypes.int64, dtypes.float64, (dtypes.float64, False)),
      (dtypes.int64, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.int64, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.float32, dtypes.bool, (dtypes.float32, True)),
      (dtypes.float32, dtypes.uint8, (dtypes.float64, True)),
      (dtypes.float32, dtypes.uint16, (dtypes.float64, True)),
      (dtypes.float32, dtypes.uint32, (dtypes.float64, True)),
      (dtypes.float32, dtypes.uint64, (dtypes.float64, True)),
      (dtypes.float32, dtypes.int8, (dtypes.float64, True)),
      (dtypes.float32, dtypes.int16, (dtypes.float64, True)),
      (dtypes.float32, dtypes.int32, (dtypes.float64, True)),
      (dtypes.float32, dtypes.int64, (dtypes.float64, True)),
      (dtypes.float32, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.float32, dtypes.float16, (dtypes.float16, False)),
      (dtypes.float32, dtypes.float32, (dtypes.float32, False)),
      (dtypes.float32, dtypes.float64, (dtypes.float64, False)),
      (dtypes.float32, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.float32, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.float64, dtypes.bool, (dtypes.float64, True)),
      (dtypes.float64, dtypes.uint8, (dtypes.float64, True)),
      (dtypes.float64, dtypes.uint16, (dtypes.float64, True)),
      (dtypes.float64, dtypes.uint32, (dtypes.float64, True)),
      (dtypes.float64, dtypes.uint64, (dtypes.float64, True)),
      (dtypes.float64, dtypes.int8, (dtypes.float64, True)),
      (dtypes.float64, dtypes.int16, (dtypes.float64, True)),
      (dtypes.float64, dtypes.int32, (dtypes.float64, True)),
      (dtypes.float64, dtypes.int64, (dtypes.float64, True)),
      (dtypes.float64, dtypes.bfloat16, (dtypes.bfloat16, False)),
      (dtypes.float64, dtypes.float16, (dtypes.float16, False)),
      (dtypes.float64, dtypes.float32, (dtypes.float32, False)),
      (dtypes.float64, dtypes.float64, (dtypes.float64, False)),
      (dtypes.float64, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.float64, dtypes.complex128, (dtypes.complex128, False)),
      (dtypes.complex128, dtypes.bool, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.uint8, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.uint16, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.uint32, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.uint64, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.int8, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.int16, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.int32, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.int64, (dtypes.complex128, True)),
      (dtypes.complex128, dtypes.bfloat16, (dtypes.complex64, False)),
      (dtypes.complex128, dtypes.float16, (dtypes.complex64, False)),
      (dtypes.complex128, dtypes.float32, (dtypes.complex64, False)),
      (dtypes.complex128, dtypes.float64, (dtypes.complex128, False)),
      (dtypes.complex128, dtypes.complex64, (dtypes.complex64, False)),
      (dtypes.complex128, dtypes.complex128, (dtypes.complex128, False)),
  )
  def testResultTypeWeakTypesWithTF(self, weak_dtype_a, dtype_b, res_dtype):
    with DtypeConversionTestEnv('all'):
      input_a = (
          constant_op.constant(1, dtype=weak_dtype_a)
          if weak_dtype_a != dtypes.bool
          else constant_op.constant(True)
      )
      input_b = (
          constant_op.constant(2, dtype=dtype_b)
          if dtype_b != dtypes.bool
          else constant_op.constant(True)
      )
      weak_input_a = weak_tensor.WeakTensor(input_a)
      self.assertEqual(
          flexible_dtypes.result_type(weak_input_a, input_b), res_dtype
      )

  # Test all the possible weak types + weak types.
  @parameterized.parameters(
      (dtypes.int32, dtypes.int32, dtypes.int32),
      (dtypes.int32, dtypes.int64, dtypes.int64),
      (dtypes.int32, dtypes.float32, dtypes.float32),
      (dtypes.int32, dtypes.float64, dtypes.float64),
      (dtypes.int32, dtypes.complex128, dtypes.complex128),
      (dtypes.int64, dtypes.int32, dtypes.int64),
      (dtypes.int64, dtypes.int64, dtypes.int64),
      (dtypes.int64, dtypes.float32, dtypes.float32),
      (dtypes.int64, dtypes.float64, dtypes.float64),
      (dtypes.int64, dtypes.complex128, dtypes.complex128),
      (dtypes.float32, dtypes.int32, dtypes.float32),
      (dtypes.float32, dtypes.int64, dtypes.float32),
      (dtypes.float32, dtypes.float32, dtypes.float32),
      (dtypes.float32, dtypes.float64, dtypes.float64),
      (dtypes.float32, dtypes.complex128, dtypes.complex128),
      (dtypes.float64, dtypes.int32, dtypes.float64),
      (dtypes.float64, dtypes.int64, dtypes.float64),
      (dtypes.float64, dtypes.float32, dtypes.float64),
      (dtypes.float64, dtypes.float64, dtypes.float64),
      (dtypes.float64, dtypes.complex128, dtypes.complex128),
      (dtypes.complex128, dtypes.int32, dtypes.complex128),
      (dtypes.complex128, dtypes.int64, dtypes.complex128),
      (dtypes.complex128, dtypes.float32, dtypes.complex128),
      (dtypes.complex128, dtypes.float64, dtypes.complex128),
      (dtypes.complex128, dtypes.complex128, dtypes.complex128),
  )
  def testResultTypeWeakTypesWithWeakTypes(self, dtype_a, dtype_b, res_dtype):
    with DtypeConversionTestEnv('all'):
      input_a = (
          constant_op.constant(1, dtype=dtype_a)
          if dtype_a != dtypes.bool
          else constant_op.constant(True)
      )
      input_b = (
          constant_op.constant(2, dtype=dtype_b)
          if dtype_b != dtypes.bool
          else constant_op.constant(True)
      )
      weak_input_a = weak_tensor.WeakTensor(input_a)
      weak_input_b = weak_tensor.WeakTensor(input_b)
      self.assertEqual(
          flexible_dtypes.result_type(weak_input_a, weak_input_b),
          (res_dtype, True),
      )

  # Test unallowed promotions in SAFE mode. Make sure exceptions are thrown.
  @parameterized.parameters(
      ((dtypes.uint8, False), (dtypes.int8, False)),
      ((dtypes.uint8, False), (dtypes.float32, True)),
      ((dtypes.uint16, False), (dtypes.int8, False)),
      ((dtypes.uint16, False), (dtypes.int16, False)),
      ((dtypes.uint16, False), (dtypes.bfloat16, False)),
      ((dtypes.uint16, False), (dtypes.float16, False)),
      ((dtypes.uint16, False), (dtypes.float32, True)),
      ((dtypes.uint32, False), (dtypes.int8, False)),
      ((dtypes.uint32, False), (dtypes.int16, False)),
      ((dtypes.uint32, False), (dtypes.int32, False)),
      ((dtypes.uint32, False), (dtypes.bfloat16, False)),
      ((dtypes.uint32, False), (dtypes.float32, False)),
      ((dtypes.uint32, False), (dtypes.complex64, False)),
      ((dtypes.uint32, False), (dtypes.float32, True)),
      ((dtypes.uint64, False), (dtypes.int8, False)),
      ((dtypes.uint64, False), (dtypes.int16, False)),
      ((dtypes.uint64, False), (dtypes.int32, False)),
      ((dtypes.uint64, False), (dtypes.int64, False)),
      ((dtypes.uint64, False), (dtypes.bfloat16, False)),
      ((dtypes.uint64, False), (dtypes.float16, False)),
      ((dtypes.uint64, False), (dtypes.float32, False)),
      ((dtypes.uint64, False), (dtypes.float64, False)),
      ((dtypes.uint64, False), (dtypes.complex64, False)),
      ((dtypes.uint64, False), (dtypes.complex128, False)),
      ((dtypes.uint64, False), (dtypes.float32, True)),
      ((dtypes.uint64, False), (dtypes.float64, True)),
      ((dtypes.uint64, False), (dtypes.complex128, True)),
      ((dtypes.int8, False), (dtypes.float32, True)),
      ((dtypes.int16, False), (dtypes.bfloat16, False)),
      ((dtypes.int16, False), (dtypes.float16, False)),
      ((dtypes.int16, False), (dtypes.float32, True)),
      ((dtypes.int32, False), (dtypes.bfloat16, False)),
      ((dtypes.int32, False), (dtypes.float16, False)),
      ((dtypes.int32, False), (dtypes.float32, False)),
      ((dtypes.int32, False), (dtypes.complex64, False)),
      ((dtypes.int32, False), (dtypes.float32, True)),
      ((dtypes.int64, False), (dtypes.bfloat16, False)),
      ((dtypes.int64, False), (dtypes.float16, False)),
      ((dtypes.int64, False), (dtypes.float32, False)),
      ((dtypes.int64, False), (dtypes.float64, False)),
      ((dtypes.int64, False), (dtypes.complex64, False)),
      ((dtypes.int64, False), (dtypes.complex128, False)),
      ((dtypes.int64, False), (dtypes.float32, True)),
      ((dtypes.int64, False), (dtypes.float64, True)),
      ((dtypes.int64, False), (dtypes.complex128, True)),
      ((dtypes.bfloat16, False), (dtypes.float16, False)),
      ((dtypes.bfloat16, False), (dtypes.complex128, True)),
  )
  def testResultTypeSafeModeUnallowedPromo(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv('safe'):
      # Create Tensor of input dtypes.
      input_a = (
          constant_op.constant(1, dtype=a_dtype[0])
          if a_dtype[0] != dtypes.bool
          else constant_op.constant(True)
      )
      input_b = (
          constant_op.constant(2, dtype=b_dtype[0])
          if b_dtype[0] != dtypes.bool
          else constant_op.constant(False)
      )
      # Create WeakTensors if weak = True.
      if a_dtype[1]:
        input_a = weak_tensor.WeakTensor(input_a)
      if b_dtype[1]:
        input_b = weak_tensor.WeakTensor(input_b)

      with self.assertRaises(TypeError):
        flexible_dtypes.result_type(input_a, input_b)

  # Test allowed promotions in SAFE mode. Make sure no exception is thrown.
  @parameterized.parameters(
      ((dtypes.bool, False), (dtypes.bool, False)),
      ((dtypes.bool, False), (dtypes.uint8, False)),
      ((dtypes.bool, False), (dtypes.uint16, False)),
      ((dtypes.bool, False), (dtypes.uint32, False)),
      ((dtypes.bool, False), (dtypes.uint64, False)),
      ((dtypes.bool, False), (dtypes.int8, False)),
      ((dtypes.bool, False), (dtypes.int16, False)),
      ((dtypes.bool, False), (dtypes.int32, False)),
      ((dtypes.bool, False), (dtypes.int64, False)),
      ((dtypes.bool, False), (dtypes.bfloat16, False)),
      ((dtypes.bool, False), (dtypes.float16, False)),
      ((dtypes.bool, False), (dtypes.float32, False)),
      ((dtypes.bool, False), (dtypes.float64, False)),
      ((dtypes.bool, False), (dtypes.complex64, False)),
      ((dtypes.bool, False), (dtypes.complex128, False)),
      ((dtypes.bool, False), (dtypes.int32, True)),
      ((dtypes.bool, False), (dtypes.int64, True)),
      ((dtypes.bool, False), (dtypes.float32, True)),
      ((dtypes.bool, False), (dtypes.float64, True)),
      ((dtypes.bool, False), (dtypes.complex128, True)),
      ((dtypes.uint8, False), (dtypes.uint8, False)),
      ((dtypes.uint8, False), (dtypes.uint16, False)),
      ((dtypes.uint8, False), (dtypes.uint32, False)),
      ((dtypes.uint8, False), (dtypes.uint64, False)),
      ((dtypes.uint8, False), (dtypes.int16, False)),
      ((dtypes.uint8, False), (dtypes.int32, False)),
      ((dtypes.uint8, False), (dtypes.int64, False)),
      ((dtypes.uint8, False), (dtypes.bfloat16, False)),
      ((dtypes.uint8, False), (dtypes.float16, False)),
      ((dtypes.uint8, False), (dtypes.float32, False)),
      ((dtypes.uint8, False), (dtypes.float64, False)),
      ((dtypes.uint8, False), (dtypes.complex64, False)),
      ((dtypes.uint8, False), (dtypes.complex128, False)),
      ((dtypes.uint8, False), (dtypes.int32, True)),
      ((dtypes.uint8, False), (dtypes.int64, True)),
      ((dtypes.uint8, False), (dtypes.float64, True)),
      ((dtypes.uint8, False), (dtypes.complex128, True)),
      ((dtypes.uint16, False), (dtypes.uint16, False)),
      ((dtypes.uint16, False), (dtypes.uint32, False)),
      ((dtypes.uint16, False), (dtypes.uint64, False)),
      ((dtypes.uint16, False), (dtypes.int32, False)),
      ((dtypes.uint16, False), (dtypes.int64, False)),
      ((dtypes.uint16, False), (dtypes.float32, False)),
      ((dtypes.uint16, False), (dtypes.float64, False)),
      ((dtypes.uint16, False), (dtypes.complex64, False)),
      ((dtypes.uint16, False), (dtypes.complex128, False)),
      ((dtypes.uint16, False), (dtypes.int32, True)),
      ((dtypes.uint16, False), (dtypes.int64, True)),
      ((dtypes.uint16, False), (dtypes.float64, True)),
      ((dtypes.uint16, False), (dtypes.complex128, True)),
      ((dtypes.uint32, False), (dtypes.uint32, False)),
      ((dtypes.uint32, False), (dtypes.uint64, False)),
      ((dtypes.uint32, False), (dtypes.int64, False)),
      ((dtypes.uint32, False), (dtypes.float64, False)),
      ((dtypes.uint32, False), (dtypes.complex128, False)),
      ((dtypes.uint32, False), (dtypes.int32, True)),
      ((dtypes.uint32, False), (dtypes.int64, True)),
      ((dtypes.uint32, False), (dtypes.float64, True)),
      ((dtypes.uint32, False), (dtypes.complex128, True)),
      ((dtypes.uint64, False), (dtypes.uint64, False)),
      ((dtypes.uint64, False), (dtypes.int32, True)),
      ((dtypes.uint64, False), (dtypes.int64, True)),
      ((dtypes.int8, False), (dtypes.int8, False)),
      ((dtypes.int8, False), (dtypes.int16, False)),
      ((dtypes.int8, False), (dtypes.int32, False)),
      ((dtypes.int8, False), (dtypes.int64, False)),
      ((dtypes.int8, False), (dtypes.bfloat16, False)),
      ((dtypes.int8, False), (dtypes.float16, False)),
      ((dtypes.int8, False), (dtypes.float32, False)),
      ((dtypes.int8, False), (dtypes.float64, False)),
      ((dtypes.int8, False), (dtypes.complex64, False)),
      ((dtypes.int8, False), (dtypes.complex128, False)),
      ((dtypes.int8, False), (dtypes.int32, True)),
      ((dtypes.int8, False), (dtypes.int64, True)),
      ((dtypes.int8, False), (dtypes.float64, True)),
      ((dtypes.int8, False), (dtypes.complex128, True)),
      ((dtypes.int16, False), (dtypes.int16, False)),
      ((dtypes.int16, False), (dtypes.int32, False)),
      ((dtypes.int16, False), (dtypes.int64, False)),
      ((dtypes.int16, False), (dtypes.float32, False)),
      ((dtypes.int16, False), (dtypes.float64, False)),
      ((dtypes.int16, False), (dtypes.complex64, False)),
      ((dtypes.int16, False), (dtypes.complex128, False)),
      ((dtypes.int16, False), (dtypes.int32, True)),
      ((dtypes.int16, False), (dtypes.int64, True)),
      ((dtypes.int16, False), (dtypes.float64, True)),
      ((dtypes.int16, False), (dtypes.complex128, True)),
      ((dtypes.int32, False), (dtypes.int32, False)),
      ((dtypes.int32, False), (dtypes.int64, False)),
      ((dtypes.int32, False), (dtypes.float64, False)),
      ((dtypes.int32, False), (dtypes.complex128, False)),
      ((dtypes.int32, False), (dtypes.int32, True)),
      ((dtypes.int32, False), (dtypes.int64, True)),
      ((dtypes.int32, False), (dtypes.float64, True)),
      ((dtypes.int32, False), (dtypes.complex128, True)),
      ((dtypes.int64, False), (dtypes.int64, False)),
      ((dtypes.int64, False), (dtypes.int32, True)),
      ((dtypes.int64, False), (dtypes.int64, True)),
      ((dtypes.bfloat16, False), (dtypes.bfloat16, False)),
      ((dtypes.bfloat16, False), (dtypes.float32, False)),
      ((dtypes.bfloat16, False), (dtypes.float64, False)),
      ((dtypes.bfloat16, False), (dtypes.complex64, False)),
      ((dtypes.bfloat16, False), (dtypes.complex128, False)),
      ((dtypes.bfloat16, False), (dtypes.int32, True)),
      ((dtypes.bfloat16, False), (dtypes.int64, True)),
      ((dtypes.bfloat16, False), (dtypes.float32, True)),
      ((dtypes.bfloat16, False), (dtypes.float64, True)),
      ((dtypes.float16, False), (dtypes.float16, False)),
      ((dtypes.float16, False), (dtypes.float32, False)),
      ((dtypes.float16, False), (dtypes.float64, False)),
      ((dtypes.float16, False), (dtypes.complex64, False)),
      ((dtypes.float16, False), (dtypes.complex128, False)),
      ((dtypes.float16, False), (dtypes.int32, True)),
      ((dtypes.float16, False), (dtypes.int64, True)),
      ((dtypes.float16, False), (dtypes.float32, True)),
      ((dtypes.float16, False), (dtypes.float64, True)),
      ((dtypes.float32, False), (dtypes.float32, False)),
      ((dtypes.float32, False), (dtypes.float64, False)),
      ((dtypes.float32, False), (dtypes.complex64, False)),
      ((dtypes.float32, False), (dtypes.complex128, False)),
      ((dtypes.float32, False), (dtypes.int32, True)),
      ((dtypes.float32, False), (dtypes.int64, True)),
      ((dtypes.float32, False), (dtypes.float32, True)),
      ((dtypes.float32, False), (dtypes.float64, True)),
      ((dtypes.float64, False), (dtypes.float64, False)),
      ((dtypes.float64, False), (dtypes.complex128, False)),
      ((dtypes.float64, False), (dtypes.int32, True)),
      ((dtypes.float64, False), (dtypes.int64, True)),
      ((dtypes.float64, False), (dtypes.float32, True)),
      ((dtypes.float64, False), (dtypes.float64, True)),
      ((dtypes.float64, False), (dtypes.complex128, True)),
      ((dtypes.complex64, False), (dtypes.complex64, False)),
      ((dtypes.complex64, False), (dtypes.complex128, False)),
      ((dtypes.complex64, False), (dtypes.int32, True)),
      ((dtypes.complex64, False), (dtypes.int64, True)),
      ((dtypes.complex64, False), (dtypes.float32, True)),
      ((dtypes.complex64, False), (dtypes.float64, True)),
      ((dtypes.complex64, False), (dtypes.complex128, True)),
      ((dtypes.complex128, False), (dtypes.complex128, False)),
      ((dtypes.complex128, False), (dtypes.int32, True)),
      ((dtypes.complex128, False), (dtypes.int64, True)),
      ((dtypes.complex128, False), (dtypes.float32, True)),
      ((dtypes.complex128, False), (dtypes.float64, True)),
      ((dtypes.complex128, False), (dtypes.complex128, True)),
      ((dtypes.int32, True), (dtypes.int32, True)),
      ((dtypes.int32, True), (dtypes.int64, True)),
      ((dtypes.int32, True), (dtypes.float32, True)),
      ((dtypes.int32, True), (dtypes.float64, True)),
      ((dtypes.int32, True), (dtypes.complex128, True)),
      ((dtypes.int64, True), (dtypes.int64, True)),
      ((dtypes.int64, True), (dtypes.float32, True)),
      ((dtypes.int64, True), (dtypes.float64, True)),
      ((dtypes.int64, True), (dtypes.complex128, True)),
      ((dtypes.float32, True), (dtypes.float32, True)),
      ((dtypes.float32, True), (dtypes.float64, True)),
      ((dtypes.float32, True), (dtypes.complex128, True)),
      ((dtypes.float64, True), (dtypes.float64, True)),
      ((dtypes.float64, True), (dtypes.complex128, True)),
      ((dtypes.complex128, True), (dtypes.complex128, True)),
  )
  def testResultTypeSafeModeAllowedPromo(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv('safe'):
      # Create Tensor of input dtypes.
      input_a = (
          constant_op.constant(1, dtype=a_dtype[0])
          if a_dtype[0] != dtypes.bool
          else constant_op.constant(True)
      )
      input_b = (
          constant_op.constant(2, dtype=b_dtype[0])
          if b_dtype[0] != dtypes.bool
          else constant_op.constant(False)
      )
      # Create WeakTensors if weak = True.
      if a_dtype[1]:
        input_a = weak_tensor.WeakTensor(input_a)
      if b_dtype[1]:
        input_b = weak_tensor.WeakTensor(input_b)
      flexible_dtypes.result_type(input_a, input_b)

  # Test Python nested structure type inference.
  def testResultTypePythonNestedStructure(self):
    with DtypeConversionTestEnv('all'):
      # i32* + f32* => f32*
      self.assertEqual(
          flexible_dtypes.result_type([1], [1.0]),
          (dtypes.float32, True),
      )
      # f32* + c128* => c128*
      self.assertEqual(
          flexible_dtypes.result_type([1, 2.0], [1.0j]),
          (dtypes.complex128, True),
      )
      self.assertEqual(
          flexible_dtypes.result_type([[1, 1.0], [1.0, 1.0]], [1.0j]),
          (dtypes.complex128, True),
      )

  # Test tf.variable type inference.
  def testResultTypeVariable(self):
    with DtypeConversionTestEnv('all'):
      v = variables.Variable(1.0, dtype=dtypes.float32)
      t = constant_op.constant(1, dtype=dtypes.float64)
      self.assertEqual(
          flexible_dtypes.result_type(v, t),
          (dtypes.float64, False),
      )

  # Test TF Dtypes type inference.
  def testResultTypeTFDtype(self):
    with DtypeConversionTestEnv('all'):
      d1 = dtypes.float32
      d2 = dtypes.float16
      self.assertEqual(
          flexible_dtypes.result_type(d1, d2),
          (dtypes.float32, False),
      )

  # Test NP dtype class type inference.
  def testResultTypeNPDtype(self):
    with DtypeConversionTestEnv('all'):
      d = np.dtype(np.float32)
      self.assertEqual(
          flexible_dtypes.result_type(d),
          (dtypes.float32, False),
      )

      d = np.dtype([('f1', np.int16)])
      with self.assertRaises(NotImplementedError):
        _ = flexible_dtypes.result_type(d)

      d = np.dtype([('a', 'f8'), ('b', 'S10')])
      with self.assertRaises(NotImplementedError):
        _ = flexible_dtypes.result_type(d)

  # Test bool type inference.
  def testResultTypeBool(self):
    with DtypeConversionTestEnv('all'):
      self.assertEqual(
          flexible_dtypes.result_type(True, False),
          (dtypes.bool, False),
      )

  # Test Tensor shape type inference.
  def testResultTypeTensorShape(self):
    with DtypeConversionTestEnv('all'):
      t = constant_op.constant([1, 2], dtype=dtypes.float64)
      self.assertEqual(
          flexible_dtypes.result_type(t.shape), (dtypes.int32, False)
      )

  # Test string types.
  def testResultTypeStr(self):
    with DtypeConversionTestEnv('all'):
      res = flexible_dtypes.result_type('foo', 'bar')
      self.assertEqual(res[0], dtypes.string)
      with self.assertRaisesRegex(
          NotImplementedError,
          "Implicit Conversion between <dtype: 'string'> and <dtype: 'int32'>"
          ' is not allowed. Please convert the input manually if you need to.',
      ):
        flexible_dtypes.result_type('foo', 1)

  # Test byte types.
  def testResultTypeBytes(self):
    with DtypeConversionTestEnv('all'):
      res = flexible_dtypes.result_type(b'foo', b'bar')
      self.assertEqual(res[0], dtypes.string)
      with self.assertRaisesRegex(
          NotImplementedError,
          "Implicit Conversion between <dtype: 'string'> and <dtype: 'int32'>"
          ' is not allowed. Please convert the input manually if you need to.',
      ):
        flexible_dtypes.result_type(b'foo', 1)

  # Test empty input.
  def testResultTypeEmptyInput(self):
    with DtypeConversionTestEnv('all'):
      dtype, is_weak = flexible_dtypes.result_type()
      self.assertEqual(dtype, dtypes.float32)
      self.assertTrue(is_weak)

  def testResultTypeUnsupportedInputType(self):
    class MyTensor(extension_type.ExtensionType):
      value: tensor.Tensor

    with DtypeConversionTestEnv('all'):
      a = MyTensor(constant_op.constant(1))
      with self.assertRaisesRegex(
          NotImplementedError,
          f'Auto dtype conversion semantics does not support {type(a)} type.',
      ):
        _ = flexible_dtypes.result_type(a)

  # Test v1 + v2 = v2 + v1.
  def testCommunicativity(self):
    with DtypeConversionTestEnv('all'):
      for v1 in _ALL_INPUT_TYPES:
        for v2 in _ALL_INPUT_TYPES:
          self.assertEqual(
              flexible_dtypes.result_type(v1, v2),
              flexible_dtypes.result_type(v2, v1),
          )

  # Test (v1 + v2) + v3 = v1 + (v2 + v3).
  def testAssociativity(self):
    with DtypeConversionTestEnv('all'):
      for v1 in _ALL_INPUT_TYPES:
        for v2 in _ALL_INPUT_TYPES:
          for v3 in _ALL_INPUT_TYPES:
            all_res = [
                flexible_dtypes.result_type(v1, v2, v3),
                flexible_dtypes.result_type(v1, v3, v2),
                flexible_dtypes.result_type(v2, v1, v3),
                flexible_dtypes.result_type(v2, v3, v1),
                flexible_dtypes.result_type(v3, v1, v2),
                flexible_dtypes.result_type(v3, v2, v1),
            ]
            self.assertAllEqual(all_res[:-1], all_res[1:])


if __name__ == '__main__':
  tf_test.main()
  ops.enable_eager_execution()
