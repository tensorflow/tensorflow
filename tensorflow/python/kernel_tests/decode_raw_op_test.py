# Copyright 2015 Google Inc. All Rights Reserved.
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

import tensorflow as tf


class DecodeRawOpTest(tf.test.TestCase):

  def testToUint8(self):
    with self.test_session():
      in_bytes = tf.placeholder(tf.string, shape=[2])
      decode = tf.decode_raw(in_bytes, out_type=tf.uint8)
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
      in_bytes = tf.placeholder(tf.string, shape=[None])
      decode = tf.decode_raw(in_bytes, out_type=tf.int16)
      self.assertEqual([None, None], decode.get_shape().as_list())

      result = decode.eval(feed_dict={in_bytes: ["AaBC"]})
      self.assertAllEqual([[ord("A") + ord("a") * 256,
                            ord("B") + ord("C") * 256]], result)

      with self.assertRaisesOpError(
          "Input to DecodeRaw has length 3 that is not a multiple of 2, the "
          "size of int16"):
        decode.eval(feed_dict={in_bytes: ["123", "456"]})

if __name__ == "__main__":
  tf.test.main()
