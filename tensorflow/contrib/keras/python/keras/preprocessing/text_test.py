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
"""Tests for text data preprocessing utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.python.platform import test


class TestText(test.TestCase):

  def test_one_hot(self):
    text = 'The cat sat on the mat.'
    encoded = keras.preprocessing.text.one_hot(text, 5)
    self.assertEqual(len(encoded), 6)
    assert np.max(encoded) <= 4
    assert np.min(encoded) >= 0

  def test_tokenizer(self):
    texts = [
        'The cat sat on the mat.',
        'The dog sat on the log.',
        'Dogs and cats living together.'
    ]
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=10)
    tokenizer.fit_on_texts(texts)

    sequences = []
    for seq in tokenizer.texts_to_sequences_generator(texts):
      sequences.append(seq)
    assert np.max(np.max(sequences)) < 10
    self.assertEqual(np.min(np.min(sequences)), 1)

    tokenizer.fit_on_sequences(sequences)

    for mode in ['binary', 'count', 'tfidf', 'freq']:
      matrix = tokenizer.texts_to_matrix(texts, mode)
      self.assertEqual(matrix.shape, (3, 10))


if __name__ == '__main__':
  test.main()
