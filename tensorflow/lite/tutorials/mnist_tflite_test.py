import unittest

import numpy as np

from tensorflow.lite.tutorials import mnist_tflite


class MnistTfliteTest(unittest.TestCase):

  def test_should_return_true_if_label_matches_argmax(self):
    output = np.array([0.1, 0.2, 0.7], dtype=np.float32)

    self.assertTrue(mnist_tflite.is_correct_prediction(output, 2))

  def test_should_return_false_if_label_does_not_match_argmax(self):
    output = np.array([0.1, 0.7, 0.2], dtype=np.float32)

    self.assertFalse(mnist_tflite.is_correct_prediction(output, 2))

  def test_should_raise_value_error_if_output_is_empty(self):
    output = np.array([], dtype=np.float32)

    with self.assertRaisesRegex(ValueError, 'Output must not be empty.'):
      mnist_tflite.predicted_label(output)


if __name__ == '__main__':
  unittest.main()
