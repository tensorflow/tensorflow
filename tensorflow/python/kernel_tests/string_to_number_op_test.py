"""Tests for StringToNumber op from parsing_ops."""

import tensorflow.python.platform

import tensorflow as tf


_ERROR_MESSAGE = "StringToNumberOp could not correctly convert string: "


class StringToNumberOpTest(tf.test.TestCase):

  def testToFloat(self):
    with self.test_session():
      input_string = tf.placeholder(tf.string)
      output = tf.string_to_number(
          input_string,
          out_type=tf.float32)

      result = output.eval(feed_dict={
          input_string: ["0",
                         "3",
                         "-1",
                         "1.12",
                         "0xF",
                         "   -10.5",
                         "3.40282e+38",
                         # The next two exceed maximum value for float, so we
                         # expect +/-INF to be returned instead.
                         "3.40283e+38",
                         "-3.40283e+38",
                         "NAN",
                         "INF"]
      })

      self.assertAllClose([0, 3, -1, 1.12, 0xF, -10.5, 3.40282e+38,
                           float("INF"), float("-INF"), float("NAN"),
                           float("INF")], result)

      with self.assertRaisesOpError(_ERROR_MESSAGE + "10foobar"):
        output.eval(feed_dict={input_string: ["10foobar"]})

  def testToInt32(self):
    with self.test_session():
      input_string = tf.placeholder(tf.string)
      output = tf.string_to_number(
          input_string,
          out_type=tf.int32)

      result = output.eval(feed_dict={
          input_string: ["0", "3", "-1", "    -10", "-2147483648", "2147483647"]
      })

      self.assertAllEqual([0, 3, -1, -10, -2147483648, 2147483647], result)

      with self.assertRaisesOpError(_ERROR_MESSAGE + "2.9"):
        output.eval(feed_dict={input_string: ["2.9"]})

      # The next two exceed maximum value of int32.
      for in_string in ["-2147483649", "2147483648"]:
        with self.assertRaisesOpError(_ERROR_MESSAGE + in_string):
          output.eval(feed_dict={input_string: [in_string]})


if __name__ == "__main__":
  tf.test.main()
