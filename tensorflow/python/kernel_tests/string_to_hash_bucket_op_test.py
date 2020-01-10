"""Tests for StringToHashBucket op from string_ops."""
import tensorflow.python.platform

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
