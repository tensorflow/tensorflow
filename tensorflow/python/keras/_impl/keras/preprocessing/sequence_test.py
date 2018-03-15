# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sequence data preprocessing utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test


class TestSequence(test.TestCase):

  def test_pad_sequences(self):
    a = [[1], [1, 2], [1, 2, 3]]

    # test padding
    b = keras.preprocessing.sequence.pad_sequences(a, maxlen=3, padding='pre')
    self.assertAllClose(b, [[0, 0, 1], [0, 1, 2], [1, 2, 3]])
    b = keras.preprocessing.sequence.pad_sequences(a, maxlen=3, padding='post')
    self.assertAllClose(b, [[1, 0, 0], [1, 2, 0], [1, 2, 3]])

    # test truncating
    b = keras.preprocessing.sequence.pad_sequences(
        a, maxlen=2, truncating='pre')
    self.assertAllClose(b, [[0, 1], [1, 2], [2, 3]])
    b = keras.preprocessing.sequence.pad_sequences(
        a, maxlen=2, truncating='post')
    self.assertAllClose(b, [[0, 1], [1, 2], [1, 2]])

    # test value
    b = keras.preprocessing.sequence.pad_sequences(a, maxlen=3, value=1)
    self.assertAllClose(b, [[1, 1, 1], [1, 1, 2], [1, 2, 3]])

  def test_pad_sequences_vector(self):
    a = [[[1, 1]], [[2, 1], [2, 2]], [[3, 1], [3, 2], [3, 3]]]

    # test padding
    b = keras.preprocessing.sequence.pad_sequences(a, maxlen=3, padding='pre')
    self.assertAllClose(b, [[[0, 0], [0, 0], [1, 1]], [[0, 0], [2, 1], [2, 2]],
                            [[3, 1], [3, 2], [3, 3]]])
    b = keras.preprocessing.sequence.pad_sequences(a, maxlen=3, padding='post')
    self.assertAllClose(b, [[[1, 1], [0, 0], [0, 0]], [[2, 1], [2, 2], [0, 0]],
                            [[3, 1], [3, 2], [3, 3]]])

    # test truncating
    b = keras.preprocessing.sequence.pad_sequences(
        a, maxlen=2, truncating='pre')
    self.assertAllClose(b, [[[0, 0], [1, 1]], [[2, 1], [2, 2]], [[3, 2], [3,
                                                                          3]]])

    b = keras.preprocessing.sequence.pad_sequences(
        a, maxlen=2, truncating='post')
    self.assertAllClose(b, [[[0, 0], [1, 1]], [[2, 1], [2, 2]], [[3, 1], [3,
                                                                          2]]])

    # test value
    b = keras.preprocessing.sequence.pad_sequences(a, maxlen=3, value=1)
    self.assertAllClose(b, [[[1, 1], [1, 1], [1, 1]], [[1, 1], [2, 1], [2, 2]],
                            [[3, 1], [3, 2], [3, 3]]])

  def test_make_sampling_table(self):
    a = keras.preprocessing.sequence.make_sampling_table(3)
    self.assertAllClose(
        a, np.asarray([0.00315225, 0.00315225, 0.00547597]), rtol=.1)

  def test_skipgrams(self):
    # test with no window size and binary labels
    couples, labels = keras.preprocessing.sequence.skipgrams(
        np.arange(3), vocabulary_size=3)
    for couple in couples:
      assert couple[0] in [0, 1, 2] and couple[1] in [0, 1, 2]

    # test window size and categorical labels
    couples, labels = keras.preprocessing.sequence.skipgrams(
        np.arange(5), vocabulary_size=5, window_size=1, categorical=True)
    for couple in couples:
      assert couple[0] - couple[1] <= 3
    for l in labels:
      assert len(l) == 2


if __name__ == '__main__':
  test.main()
