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

"""Tests for Substr op from string_ops."""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np


class SubstrOpTest(tf.test.TestCase):

  def testScalarString(self):
    test_string = b"Hello"
    position = 1
    length = 3
    expected_value = b"ell"

    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def testVectorStrings(self):
    test_string = [b"Hello", b"World"]
    position = 1
    length = 3
    expected_value = [b"ell", b"orl"]

    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def testMatrixStrings(self):
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = 1
    length = 4
    expected_value = [[b"en", b"leve", b"welv"],
                      [b"hirt", b"ourt", b"ifte"],
                      [b"ixte", b"even", b"ight"]]

    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def testOutOfRangeError(self):
    test_string = b"Hello"
    position = 7
    length = 3
    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        substr = substr_op.eval()

    test_string = [b"good", b"good", b"bad", b"good"]
    position = 3
    length = 1
    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        substr = substr_op.eval()

    test_string = b"Hello"
    position = -1
    length = 3
    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        substr = substr_op.eval()
      

if __name__ == "__main__":
  tf.test.main()
