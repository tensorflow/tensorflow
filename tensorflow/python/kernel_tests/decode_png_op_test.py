# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for DecodePngOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class DecodePngOpTest(test.TestCase):

  def test16bit(self):
    img_bytes = [[0, 255], [1024, 1024 + 255]]
    # Encoded PNG bytes resulting from encoding the above img_bytes
    # using go's image/png encoder.
    encoded_bytes = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0,
        2, 0, 0, 0, 2, 16, 0, 0, 0, 0, 7, 77, 142, 187, 0, 0, 0, 21, 73, 68, 65,
        84, 120, 156, 98, 98, 96, 96, 248, 207, 194, 2, 36, 1, 1, 0, 0, 255,
        255, 6, 60, 1, 10, 68, 160, 26, 131, 0, 0, 0, 0, 73, 69, 78, 68, 174,
        66, 96, 130
    ]

    byte_string = bytes(bytearray(encoded_bytes))
    img_in = constant_op.constant(byte_string, dtype=dtypes.string)
    decode = array_ops.squeeze(
        image_ops.decode_png(
            img_in, dtype=dtypes.uint16))

    with self.cached_session():
      decoded = self.evaluate(decode)
      self.assertAllEqual(decoded, img_bytes)


if __name__ == "__main__":
  test.main()
