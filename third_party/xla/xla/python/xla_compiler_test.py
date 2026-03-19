# Copyright 2024 The OpenXLA Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from xla.python import xla_extension


def create_random_array(shape, dtype: np.dtype) -> np.ndarray:
  if np.issubdtype(dtype, np.complexfloating):
    return np.random.rand(*shape).astype(dtype) + 1j * np.random.rand(*shape)
  else:
    return np.random.rand(*shape).astype(dtype)


@parameterized.parameters(
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.bool,
    np.complex64,
    np.complex128,
)
class ConstructLiteralTest(parameterized.TestCase):

  def test_create_literal_from_ndarray_rank_1(self, dtype):
    input_array = create_random_array([10], dtype)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape
    )
    literal = xla_extension.Literal(shape)
    # use `np.asarray` to ensure that the array is a view into the literal.
    array = np.asarray(literal)
    np.copyto(array, input_array)
    # Intentionally check against `np.array(literal)` instead of `array`
    # to ensure that the underlying literal is actually updated and not some
    # rebinding to a new object. (This also exersises the copy functionality)
    np.testing.assert_array_equal(np.array(literal), input_array)

  def test_create_literal_from_ndarray_rank_2(self, dtype):
    input_array = create_random_array([20, 5], dtype)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape, [1, 0]
    )
    literal = xla_extension.Literal(shape)
    array = np.asarray(literal)
    np.copyto(array, input_array)
    np.testing.assert_array_equal(np.array(literal), input_array)

  def test_create_literal_from_ndarray_rank_2_reverse_layout(self, dtype):
    input_array = create_random_array([25, 4], dtype)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape, [0, 1]
    )
    literal = xla_extension.Literal(shape)
    array = np.asarray(literal)
    np.copyto(array, input_array)
    np.testing.assert_array_equal(np.array(literal), input_array)


if __name__ == "__main__":
  absltest.main()
