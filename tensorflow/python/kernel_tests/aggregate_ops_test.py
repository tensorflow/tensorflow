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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
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
          dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
          dtypes.complex128, dtypes.int64
      ]
    return [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
            dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
            dtypes.complex128]

  def _buildData(self, shape, dtype):
    data = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testAddN(self):
    np.random.seed(12345)
    with self.session(use_gpu=True) as sess:
      for dtype in self._supported_types():
        for count in range(1, self._MAX_N + 1):
          data = [self._buildData((2, 2), dtype) for _ in range(count)]
          actual = self.evaluate(math_ops.add_n(data))
          expected = np.sum(np.vstack(
              [np.expand_dims(d, 0) for d in data]), axis=0)
          tol = 5e-3 if dtype == dtypes.float16 else 5e-7
          self.assertAllClose(expected, actual, rtol=tol, atol=tol)

  def testUnknownShapes(self):
    np.random.seed(12345)
    with self.session(use_gpu=True) as sess:
      for dtype in self._supported_types():
        data = self._buildData((2, 2), dtype)
        for count in range(1, self._MAX_N + 1):
          data_ph = array_ops.placeholder(dtype=dtype)
          actual = sess.run(math_ops.add_n([data_ph] * count), {data_ph: data})
          expected = np.sum(np.vstack([np.expand_dims(data, 0)] * count),
                            axis=0)
          tol = 5e-3 if dtype == dtypes.float16 else 5e-7
          self.assertAllClose(expected, actual, rtol=tol, atol=tol)

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
      variant_const_3 = create_constant_variant(3)
      variant_const_4 = create_constant_variant(4)
      variant_const_5 = create_constant_variant(5)
      # 3 + 3 + 5 + 4 = 15.
      result = math_ops.add_n((variant_const_3, variant_const_3,
                               variant_const_5, variant_const_4))

      # Smoke test -- ensure this executes without trouble.
      # Right now, non-numpy-compatible objects cannot be returned from a
      # session.run call; similarly, objects that can't be converted to
      # native numpy types cannot be passed to ops.convert_to_tensor.
      # For now, run the test and examine the output to see that the result is
      # equal to 15.
      result_op = logging_ops.Print(
          result, [variant_const_3, variant_const_4, variant_const_5, result],
          message=("Variants stored an int: c(3), c(4), c(5), "
                   "add_n(c(3), c(3), c(5), c(4)): ")).op
      result_op.run()


if __name__ == "__main__":
  test.main()
