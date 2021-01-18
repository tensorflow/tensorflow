# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for DecodeRaw op from parsing_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class EncodeRawOpTest(test.TestCase):

  def testShapeInference(self):
    # Shape function requires placeholders and a graph.
    with ops.Graph().as_default():
      for dtype in [dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16,
                    dtypes.uint16, dtypes.int32, dtypes.int64, dtypes.float16,
                    dtypes.float32, dtypes.float64, dtypes.complex64,
                    dtypes.complex128]:
        in_bytes = array_ops.placeholder(dtype, shape=[None, None])
        encode = parsing_ops.encode_raw(in_bytes)
        self.assertEqual([None], encode.get_shape().as_list())

  def encodeThenDecode(self, val, dtype):
      self.assertAllEqual(
          val,
          parsing_ops.decode_raw(parsing_ops.encode_raw(val), dtype))

  def testUint8(self):
    self.encodeThenDecode(
        np.array([[ord("A")], [ord("a")]], dtype="u1"), dtype=dtypes.uint8)
    self.encodeThenDecode(
        np.array(
            [[ord("w"), ord("e"), ord("r")], [ord("X"), ord("Y"), ord("Z")]],
            dtype="u1"),
        dtype=dtypes.uint8)

  def testInt16(self):
    self.encodeThenDecode(
        np.array([[ord("A") + ord("a") * 256, ord("B") + ord("C") * 256]],
                 dtype="i2"),
        dtype=dtypes.int16)

  def testFloat16(self):
    self.encodeThenDecode(
        np.matrix([[1, -2, -3, 4]], dtype="f2"),
        dtype=dtypes.float16)

  def testBool(self):
    self.encodeThenDecode(
        np.matrix([[True, False, False, True]], dtype="b1"),
        dtype=dtypes.bool)

  def testToComplex64(self):
    self.encodeThenDecode(
        np.matrix([[1 + 1j, 2 - 2j, -3 + 3j, -4 - 4j]], dtype="c8"),
        dtype=dtypes.complex64)

  def testToComplex128(self):
    self.encodeThenDecode(
        np.matrix([[1 + 1j, 2 - 2j, -3 + 3j, -4 - 4j]], dtype="c16"),
        dtype=dtypes.complex128)

  def testToUInt16(self):
    # Use FF/EE/DD/CC so that decoded value is higher than 32768 for uint16
    self.assertAllEqual(
        [[0xFF + 0xEE * 256, 0xDD + 0xCC * 256]],
        parsing_ops.decode_raw([b"\xFF\xEE\xDD\xCC"], dtypes.uint16))


if __name__ == "__main__":
  test.main()
