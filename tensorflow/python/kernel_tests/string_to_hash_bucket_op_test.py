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
"""Tests for StringToHashBucket op from string_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringToHashBucketOpTest(test.TestCase):

  def testStringToOneHashBucketFast(self):
    with self.cached_session():
      input_string = array_ops.placeholder(dtypes.string)
      output = string_ops.string_to_hash_bucket_fast(input_string, 1)
      result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})

      self.assertAllEqual([0, 0, 0], result)

  def testStringToHashBucketsFast(self):
    with self.cached_session():
      input_string = array_ops.placeholder(dtypes.string)
      output = string_ops.string_to_hash_bucket_fast(input_string, 10)
      result = output.eval(feed_dict={input_string: ['a', 'b', 'c', 'd']})

      # Fingerprint64('a') -> 12917804110809363939 -> mod 10 -> 9
      # Fingerprint64('b') -> 11795596070477164822 -> mod 10 -> 2
      # Fingerprint64('c') -> 11430444447143000872 -> mod 10 -> 2
      # Fingerprint64('d') -> 4470636696479570465 -> mod 10 -> 5
      self.assertAllEqual([9, 2, 2, 5], result)

  def testStringToOneHashBucketLegacyHash(self):
    with self.cached_session():
      input_string = array_ops.placeholder(dtypes.string)
      output = string_ops.string_to_hash_bucket(input_string, 1)
      result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})

      self.assertAllEqual([0, 0, 0], result)

  def testStringToHashBucketsLegacyHash(self):
    with self.cached_session():
      input_string = array_ops.placeholder(dtypes.string)
      output = string_ops.string_to_hash_bucket(input_string, 10)
      result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})

      # Hash64('a') -> 2996632905371535868 -> mod 10 -> 8
      # Hash64('b') -> 5795986006276551370 -> mod 10 -> 0
      # Hash64('c') -> 14899841994519054197 -> mod 10 -> 7
      self.assertAllEqual([8, 0, 7], result)

  def testStringToOneHashBucketStrongOneHashBucket(self):
    with self.cached_session():
      input_string = constant_op.constant(['a', 'b', 'c'])
      output = string_ops.string_to_hash_bucket_strong(
          input_string, 1, key=[123, 345])
      self.assertAllEqual([0, 0, 0], output.eval())

  def testStringToHashBucketsStrong(self):
    with self.cached_session():
      input_string = constant_op.constant(['a', 'b', 'c'])
      output = string_ops.string_to_hash_bucket_strong(
          input_string, 10, key=[98765, 132])
      # key = [98765, 132]
      # StrongKeyedHash(key, 'a') -> 7157389809176466784 -> mod 10 -> 4
      # StrongKeyedHash(key, 'b') -> 15805638358933211562 -> mod 10 -> 2
      # StrongKeyedHash(key, 'c') -> 18100027895074076528 -> mod 10 -> 8
      self.assertAllEqual([4, 2, 8], output.eval())

  def testStringToHashBucketsStrongInvalidKey(self):
    with self.cached_session():
      input_string = constant_op.constant(['a', 'b', 'c'])
      with self.assertRaisesOpError('Key must have 2 elements'):
        string_ops.string_to_hash_bucket_strong(
            input_string, 10, key=[98765]).eval()


if __name__ == '__main__':
  test.main()
