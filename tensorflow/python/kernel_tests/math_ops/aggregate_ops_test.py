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
"""Tests for aggregate_ops."""

import numpy as np

from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class AddNTest(test.TestCase):
  # AddN special-cases adding the first M inputs to make (N - M) divisible by 8,
  # after which it adds the remaining (N - M) tensors 8 at a time in a loop.
  # Test N in [1, 10] so we check each special-case from 1 to 9 and one
  # iteration of the loop.
  _MAX_N = 10

  def _supported_types(self):
    if test.is_gpu_available():
      return [
          dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64,
          dtypes.complex64, dtypes.complex128, dtypes.int64
      ]
    return [
        dtypes.int8,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.bfloat16,
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.complex64,
        dtypes.complex128,
    ]

  def _buildData(self, shape, dtype):
    data = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testAddN(self):
    np.random.seed(12345)
    with self.session():
      for dtype in self._supported_types():
        for count in range(1, self._MAX_N + 1):
          data = [self._buildData((2, 2), dtype) for _ in range(count)]
          actual = self.evaluate(math_ops.add_n(data))
          expected = np.sum(np.vstack(
              [np.expand_dims(d, 0) for d in data]), axis=0)
          self.assertAllCloseAccordingToType(
              expected,
              actual,
              float_rtol=5e-6,
              float_atol=5e-6,
              half_rtol=5e-3,
              half_atol=5e-3,
          )

  @test_util.run_deprecated_v1
  def testUnknownShapes(self):
    np.random.seed(12345)
    with self.session() as sess:
      for dtype in self._supported_types():
        data = self._buildData((2, 2), dtype)
        for count in range(1, self._MAX_N + 1):
          data_ph = array_ops.placeholder(dtype=dtype)
          actual = sess.run(math_ops.add_n([data_ph] * count), {data_ph: data})
          expected = np.sum(np.vstack([np.expand_dims(data, 0)] * count),
                            axis=0)
          self.assertAllCloseAccordingToType(
              expected,
              actual,
              half_rtol=5e-3,
              half_atol=5e-3,
          )

  @test_util.run_deprecated_v1
  def testVariant(self):

    def create_constant_variant(value):
      return constant_op.constant(
          tensor_pb2.TensorProto(
              dtype=dtypes.variant.as_datatype_enum,
              tensor_shape=tensor_shape.TensorShape([]).as_proto(),
              variant_val=[
                  tensor_pb2.VariantTensorDataProto(
                      # Match registration in variant_op_registry.cc
                      type_name=b"int",
                      metadata=np.array(value, dtype=np.int32).tobytes())
              ]))

    # TODO(ebrevdo): Re-enable use_gpu=True once non-DMA Variant
    # copying between CPU and GPU is supported.
    with self.session(use_gpu=False):
      num_tests = 127
      values = list(range(100))
      variant_consts = [create_constant_variant(x) for x in values]
      sum_count_indices = np.random.randint(1, 29, size=num_tests)
      sum_indices = [
          np.random.randint(100, size=count) for count in sum_count_indices]
      expected_sums = [np.sum(x) for x in sum_indices]
      variant_sums = [math_ops.add_n([variant_consts[i] for i in x])
                      for x in sum_indices]

      # We use as_string() to get the Variant DebugString for the
      # variant_sums; we know its value so we can check via string equality
      # here.
      #
      # Right now, non-numpy-compatible objects cannot be returned from a
      # session.run call; similarly, objects that can't be converted to
      # native numpy types cannot be passed to ops.convert_to_tensor.
      variant_sums_string = string_ops.as_string(variant_sums)
      self.assertAllEqual(
          variant_sums_string,
          ["Variant<type: int value: {}>".format(s).encode("utf-8")
           for s in expected_sums])


if __name__ == "__main__":
  test.main()
