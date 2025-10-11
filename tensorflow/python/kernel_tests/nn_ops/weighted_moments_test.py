import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class WeightedMomentsTest(test.TestCase):

  def test_tensor_axes_keepdims_false_shape_mismatch(self):
    x = tf.constant([[1., 2., 3., 4., 5.],
                     [6., 7., 8., 9., 10.]])
    weights = tf.constant([[1.], [1.]])
    axes = tf.constant([0], dtype=tf.int32)

    mean, var = tf.nn.weighted_moments(x, axes=axes, frequency_weights=weights, keepdims=False)

    self.assertEqual(mean.shape.as_list(), [5])
    self.assertEqual(var.shape.as_list(), [5])

if __name__ == "__main__":
    test.main()