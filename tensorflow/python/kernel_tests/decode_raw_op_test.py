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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class DecodeRawOpTest(test.TestCase):

  def testToUint8(self):
    with self.test_session():
      in_bytes = array_ops.placeholder(dtypes.string, shape=[2])
      decode = parsing_ops.decode_raw(in_bytes, out_type=dtypes.uint8)
      self.assertEqual([2, None], decode.get_shape().as_list())

      result = decode.eval(feed_dict={in_bytes: ["A", "a"]})
      self.assertAllEqual([[ord("A")], [ord("a")]], result)

      result = decode.eval(feed_dict={in_bytes: ["wer", "XYZ"]})
      self.assertAllEqual([[ord("w"), ord("e"), ord("r")],
                           [ord("X"), ord("Y"), ord("Z")]], result)

      with self.assertRaisesOpError(
          "DecodeRaw requires input strings to all be the same size, but "
          "element 1 has size 5 != 6"):
        decode.eval(feed_dict={in_bytes: ["short", "longer"]})

  def testToInt16(self):
    with self.test_session():
      in_bytes = array_ops.placeholder(dtypes.string, shape=[None])
      decode = parsing_ops.decode_raw(in_bytes, out_type=dtypes.int16)
      self.assertEqual([None, None], decode.get_shape().as_list())

      result = decode.eval(feed_dict={in_bytes: ["AaBC"]})
      self.assertAllEqual(
          [[ord("A") + ord("a") * 256, ord("B") + ord("C") * 256]], result)

      with self.assertRaisesOpError(
          "Input to DecodeRaw has length 3 that is not a multiple of 2, the "
          "size of int16"):
        decode.eval(feed_dict={in_bytes: ["123", "456"]})

  def testToFloat16(self):
    with self.test_session():
      in_bytes = array_ops.placeholder(dtypes.string, shape=[None])
      decode = parsing_ops.decode_raw(in_bytes, out_type=dtypes.float16)
      self.assertEqual([None, None], decode.get_shape().as_list())

      expected_result = np.matrix([[1, -2, -3, 4]], dtype=np.float16)
      result = decode.eval(feed_dict={in_bytes: [expected_result.tostring()]})

      self.assertAllEqual(expected_result, result)


if __name__ == "__main__":
  test.main()
