# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for np_utils."""

import numpy as np

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.platform import test


class TestNPUtils(test.TestCase):

  def test_to_categorical(self):
    num_classes = 5
    shapes = [(1,), (3,), (4, 3), (5, 4, 3), (3, 1), (3, 2, 1)]
    expected_shapes = [(1, num_classes), (3, num_classes), (4, 3, num_classes),
                       (5, 4, 3, num_classes), (3, num_classes),
                       (3, 2, num_classes)]
    labels = [np.random.randint(0, num_classes, shape) for shape in shapes]
    one_hots = [
        np_utils.to_categorical(label, num_classes) for label in labels]
    for label, one_hot, expected_shape in zip(labels,
                                              one_hots,
                                              expected_shapes):
      # Check shape
      self.assertEqual(one_hot.shape, expected_shape)
      # Make sure there is only one 1 in a row
      self.assertTrue(np.all(one_hot.sum(axis=-1) == 1))
      # Get original labels back from one hots
      self.assertTrue(np.all(
          np.argmax(one_hot, -1).reshape(label.shape) == label))


if __name__ == '__main__':
  test.main()
