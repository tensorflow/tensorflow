import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class TruncateDivComplexTest(test.TestCase):

    def testTruncateDivComplex128(self):
        x = tf.constant([1+2j, -3.7+4.1j], dtype=tf.complex128)
        y = tf.constant([1+1j, 2-2j], dtype=tf.complex128)
        result = tf.math.truncatediv(x, y)
        expected = tf.complex(tf.math.trunc(tf.math.real(x/y)),
                              tf.math.trunc(tf.math.imag(x/y)))
        self.assertAllClose(result, expected)

    def testTruncateDivComplex64(self):
        x = tf.constant([1+2j, -3+4j], dtype=tf.complex64)
        y = tf.constant([1+1j, 2-2j], dtype=tf.complex64)
        result = tf.math.truncatediv(x, y)
        expected = tf.complex(tf.math.trunc(tf.math.real(x/y)),
                              tf.math.trunc(tf.math.imag(x/y)))
        self.assertAllClose(result, expected)

if __name__ == "__main__":
    test.main()
