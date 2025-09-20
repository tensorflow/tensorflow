import unittest
import tensorflow as tf


class SeparableConv2DPatchTest(unittest.TestCase):
    def setUp(self):
        self.x = tf.random.normal([1, 8, 8, 3])
        self.dw = tf.random.normal([3, 3, 3, 1])
        self.pw = tf.random.normal([1, 1, 3, 2])

    def test_negative_stride(self):
        with self.assertRaises(ValueError):
            tf.nn.separable_conv2d(
                self.x, self.dw, self.pw, strides=[1, -2, 1, 1], padding="VALID"
            )

    def test_large_stride(self):
        with self.assertRaises(ValueError):
            tf.nn.separable_conv2d(
                self.x, self.dw, self.pw, strides=[1, 2**40, 1, 1], padding="VALID"
            )

    def test_non_integer_stride(self):
        with self.assertRaises(TypeError):
            tf.nn.separable_conv2d(
                self.x, self.dw, self.pw, strides=[1, 1.5, 1, 1], padding="VALID"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
