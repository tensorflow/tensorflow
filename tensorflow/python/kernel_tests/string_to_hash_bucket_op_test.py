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

"""Tests for StringToHashBucket op from string_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class StringToHashBucketOpTest(tf.test.TestCase):

  def testStringToOneHashBucket(self):
    with self.test_session():
      input_string = tf.placeholder(tf.string)
      output = tf.string_to_hash_bucket(input_string, 1)
      result = output.eval(feed_dict={
          input_string: ['a', 'b', 'c']
      })

      self.assertAllEqual([0, 0, 0], result)

  def testStringToHashBuckets(self):
    with self.test_session():
      input_string = tf.placeholder(tf.string)
      output = tf.string_to_hash_bucket(input_string, 10)
      result = output.eval(feed_dict={
          input_string: ['a', 'b', 'c']
      })

      # Hash64('a') -> 2996632905371535868 -> mod 10 -> 8
      # Hash64('b') -> 5795986006276551370 -> mod 10 -> 0
      # Hash64('c') -> 14899841994519054197 -> mod 10 -> 7
      self.assertAllEqual([8, 0, 7], result)


if __name__ == '__main__':
  tf.test.main()
