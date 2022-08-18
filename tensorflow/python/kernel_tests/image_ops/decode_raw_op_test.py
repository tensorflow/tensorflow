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
"""Tests for DecodeRaw op from parsing_ops."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class DecodeRawOpTest(test.TestCase):

  def testShapeInference(self):
    # Shape function requires placeholders and a graph.
    with ops.Graph().as_default():
      for dtype in [dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16,
                    dtypes.uint16, dtypes.int32, dtypes.int64, dtypes.float16,
                    dtypes.float32, dtypes.float64, dtypes.complex64,
                    dtypes.complex128]:
        in_bytes = array_ops.placeholder(dtypes.string, shape=[None])
        decode = parsing_ops.decode_raw(in_bytes, dtype)
        self.assertEqual([None, None], decode.get_shape().as_list())

  def testToUint8(self):
    self.assertAllEqual(
        [[ord("A")], [ord("a")]],
        parsing_ops.decode_raw(["A", "a"], dtypes.uint8))

    self.assertAllEqual(
        [[ord("w"), ord("e"), ord("r")], [ord("X"), ord("Y"), ord("Z")]],
        parsing_ops.decode_raw(["wer", "XYZ"], dtypes.uint8))

    with self.assertRaisesOpError(
        "DecodeRaw requires input strings to all be the same size, but "
        "element 1 has size 5 != 6"):
      self.evaluate(parsing_ops.decode_raw(["short", "longer"], dtypes.uint8))

  def testToInt16(self):
    self.assertAllEqual(
        [[ord("A") + ord("a") * 256, ord("B") + ord("C") * 256]],
        parsing_ops.decode_raw(["AaBC"], dtypes.uint16))

    with self.assertRaisesOpError(
        "Input to DecodeRaw has length 3 that is not a multiple of 2, the "
        "size of int16"):
      self.evaluate(parsing_ops.decode_raw(["123", "456"], dtypes.int16))

  def testEndianness(self):
    self.assertAllEqual(
        [[0x04030201]],
        parsing_ops.decode_raw(
            ["\x01\x02\x03\x04"], dtypes.int32, little_endian=True))
    self.assertAllEqual(
        [[0x01020304]],
        parsing_ops.decode_raw(
            ["\x01\x02\x03\x04"], dtypes.int32, little_endian=False))
    self.assertAllEqual([[1 + 2j]],
                        parsing_ops.decode_raw([b"\x00\x00\x80?\x00\x00\x00@"],
                                               dtypes.complex64,
                                               little_endian=True))
    self.assertAllEqual([[1 + 2j]],
                        parsing_ops.decode_raw([b"?\x80\x00\x00@\x00\x00\x00"],
                                               dtypes.complex64,
                                               little_endian=False))

  def testToFloat16(self):
    result = np.matrix([[1, -2, -3, 4]], dtype="<f2")
    self.assertAllEqual(
        result, parsing_ops.decode_raw([result.tobytes()], dtypes.float16))

  def testToBool(self):
    result = np.matrix([[True, False, False, True]], dtype="<b1")
    self.assertAllEqual(result,
                        parsing_ops.decode_raw([result.tobytes()], dtypes.bool))

  def testToComplex64(self):
    result = np.matrix([[1 + 1j, 2 - 2j, -3 + 3j, -4 - 4j]], dtype="<c8")
    self.assertAllEqual(
        result, parsing_ops.decode_raw([result.tobytes()], dtypes.complex64))

  def testToComplex128(self):
    result = np.matrix([[1 + 1j, 2 - 2j, -3 + 3j, -4 - 4j]], dtype="<c16")
    self.assertAllEqual(
        result, parsing_ops.decode_raw([result.tobytes()], dtypes.complex128))

  def testEmptyStringInput(self):
    for num_inputs in range(3):
      result = parsing_ops.decode_raw([""] * num_inputs, dtypes.float16)
      self.assertEqual((num_inputs, 0), self.evaluate(result).shape)

  def testToUInt16(self):
    # Use FF/EE/DD/CC so that decoded value is higher than 32768 for uint16
    self.assertAllEqual(
        [[0xFF + 0xEE * 256, 0xDD + 0xCC * 256]],
        parsing_ops.decode_raw([b"\xFF\xEE\xDD\xCC"], dtypes.uint16))

    with self.assertRaisesOpError(
        "Input to DecodeRaw has length 3 that is not a multiple of 2, the "
        "size of uint16"):
      self.evaluate(parsing_ops.decode_raw(["123", "456"], dtypes.uint16))


if __name__ == "__main__":
  test.main()
