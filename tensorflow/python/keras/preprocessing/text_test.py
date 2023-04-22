# -*- coding: utf-8 -*-
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

import numpy as np

from tensorflow.python.keras.preprocessing import text as preprocessing_text
from tensorflow.python.platform import test


class TestText(test.TestCase):

  def test_one_hot(self):
    text = 'The cat sat on the mat.'
    encoded = preprocessing_text.one_hot(text, 5)
    self.assertEqual(len(encoded), 6)
    self.assertLessEqual(np.max(encoded), 4)
    self.assertGreaterEqual(np.min(encoded), 0)

    # Test on unicode.
    text = u'The cat sat on the mat.'
    encoded = preprocessing_text.one_hot(text, 5)
    self.assertEqual(len(encoded), 6)
    self.assertLessEqual(np.max(encoded), 4)
    self.assertGreaterEqual(np.min(encoded), 0)

  def test_tokenizer(self):
    texts = [
        'The cat sat on the mat.',
        'The dog sat on the log.',
        'Dogs and cats living together.'
    ]
    tokenizer = preprocessing_text.Tokenizer(num_words=10)
    tokenizer.fit_on_texts(texts)

    sequences = []
    for seq in tokenizer.texts_to_sequences_generator(texts):
      sequences.append(seq)
    self.assertLess(np.max(np.max(sequences)), 10)
    self.assertEqual(np.min(np.min(sequences)), 1)

    tokenizer.fit_on_sequences(sequences)

    for mode in ['binary', 'count', 'tfidf', 'freq']:
      matrix = tokenizer.texts_to_matrix(texts, mode)
      self.assertEqual(matrix.shape, (3, 10))

  def test_hashing_trick_hash(self):
    text = 'The cat sat on the mat.'
    encoded = preprocessing_text.hashing_trick(text, 5)
    self.assertEqual(len(encoded), 6)
    self.assertLessEqual(np.max(encoded), 4)
    self.assertGreaterEqual(np.min(encoded), 1)

  def test_hashing_trick_md5(self):
    text = 'The cat sat on the mat.'
    encoded = preprocessing_text.hashing_trick(
        text, 5, hash_function='md5')
    self.assertEqual(len(encoded), 6)
    self.assertLessEqual(np.max(encoded), 4)
    self.assertGreaterEqual(np.min(encoded), 1)

  def test_tokenizer_oov_flag(self):
    x_train = ['This text has only known words']
    x_test = ['This text has some unknown words']  # 2 OOVs: some, unknown

    # Default, without OOV flag
    tokenizer = preprocessing_text.Tokenizer()
    tokenizer.fit_on_texts(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    self.assertEqual(len(x_test_seq[0]), 4)  # discards 2 OOVs

    # With OOV feature
    tokenizer = preprocessing_text.Tokenizer(oov_token='<unk>')
    tokenizer.fit_on_texts(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    self.assertEqual(len(x_test_seq[0]), 6)  # OOVs marked in place

  def test_sequential_fit(self):
    texts = [
        'The cat sat on the mat.', 'The dog sat on the log.',
        'Dogs and cats living together.'
    ]
    word_sequences = [['The', 'cat', 'is', 'sitting'],
                      ['The', 'dog', 'is', 'standing']]
    tokenizer = preprocessing_text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    tokenizer.fit_on_texts(word_sequences)

    self.assertEqual(tokenizer.document_count, 5)

    tokenizer.texts_to_matrix(texts)
    tokenizer.texts_to_matrix(word_sequences)

  def test_text_to_word_sequence(self):
    text = 'hello! ? world!'
    seq = preprocessing_text.text_to_word_sequence(text)
    self.assertEqual(seq, ['hello', 'world'])

  def test_text_to_word_sequence_multichar_split(self):
    text = 'hello!stop?world!'
    seq = preprocessing_text.text_to_word_sequence(text, split='stop')
    self.assertEqual(seq, ['hello', 'world'])

  def test_text_to_word_sequence_unicode(self):
    text = u'ali! veli? kırk dokuz elli'
    seq = preprocessing_text.text_to_word_sequence(text)
    self.assertEqual(seq, [u'ali', u'veli', u'kırk', u'dokuz', u'elli'])

  def test_text_to_word_sequence_unicode_multichar_split(self):
    text = u'ali!stopveli?stopkırkstopdokuzstopelli'
    seq = preprocessing_text.text_to_word_sequence(text, split='stop')
    self.assertEqual(seq, [u'ali', u'veli', u'kırk', u'dokuz', u'elli'])

  def test_tokenizer_unicode(self):
    texts = [
        u'ali veli kırk dokuz elli', u'ali veli kırk dokuz elli veli kırk dokuz'
    ]
    tokenizer = preprocessing_text.Tokenizer(num_words=5)
    tokenizer.fit_on_texts(texts)

    self.assertEqual(len(tokenizer.word_counts), 5)


if __name__ == '__main__':
  test.main()
