# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
