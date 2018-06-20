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
"""Tests for DecodeBmpOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test



class DecodeBmpOpTest(test.TestCase):

  def testex1(self):
    img_bytes = [[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [255, 255, 255]]]
    # Encoded BMP bytes from Wikipedia
    encoded_bytes = [
        0x42, 0x40,
        0x46, 0, 0, 0,
        0, 0,
        0, 0,
        0x36, 0, 0, 0,
        0x28, 0, 0, 0,
        0x2, 0, 0, 0,
        0x2, 0, 0, 0,
        0x1, 0,
        0x18, 0,
        0, 0, 0, 0,
        0x10, 0, 0, 0,
        0x13, 0xb, 0, 0,
        0x13, 0xb, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0xff,
        0xff, 0xff, 0xff,
        0, 0,
        0xff, 0, 0,
        0, 0xff, 0,
        0, 0,
    ]

    byte_string = bytes(bytearray(encoded_bytes))
    img_in = constant_op.constant(byte_string, dtype=dtypes.string)
    decode = array_ops.squeeze(image_ops.decode_bmp(img_in))

    with self.test_session():
      decoded = decode.eval()
      self.assertAllEqual(decoded, img_bytes)

  def testGrayscale(self):
    img_bytes = [[[255], [0]], [[255], [0]]]
    encoded_bytes = [
        0x42,
        0x40,
        0x3d,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0x36,
        0,
        0,
        0,
        0x28,
        0,
        0,
        0,
        0x2,
        0,
        0,
        0,
        0x2,
        0,
        0,
        0,
        0x1,
        0,
        0x8,
        0,
        0,
        0,
        0,
        0,
        0x10,
        0,
        0,
        0,
        0x13,
        0xb,
        0,
        0,
        0x13,
        0xb,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0xff,
        0,
        0,
        0,
        0xff,
        0,
        0,
        0,
    ]

    byte_string = bytes(bytearray(encoded_bytes))
    img_in = constant_op.constant(byte_string, dtype=dtypes.string)
    decode = image_ops.decode_bmp(img_in)

    with self.test_session():
      decoded = decode.eval()
      self.assertAllEqual(decoded, img_bytes)


if __name__ == "__main__":
  test.main()
