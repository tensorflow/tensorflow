# encoding: utf-8

#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.preprocessing import categorical_vocabulary


class CategoricalVocabularyTest(tf.test.TestCase):

  def testIntVocabulary(self):
    vocab = categorical_vocabulary.CategoricalVocabulary()
    self.assertEqual(vocab.get(1), 1)
    self.assertEqual(vocab.get(3), 2)
    self.assertEqual(vocab.get(2), 3)
    self.assertEqual(vocab.get(3), 2)
    # This vocab doesn't handle nan specially.
    self.assertEqual(vocab.get(float('nan')), 4)
    self.assertEqual(len(vocab), 5)

  def testWordVocabulary(self):
    vocab = categorical_vocabulary.CategoricalVocabulary()
    self.assertEqual(vocab.get('a'), 1)
    self.assertEqual(vocab.get('b'), 2)
    self.assertEqual(vocab.get('a'), 1)
    self.assertEqual(vocab.get('b'), 2)

  def testCountsTrim(self):
    vocab = categorical_vocabulary.CategoricalVocabulary()
    vocab.get('c')
    vocab.add('c', 5)
    vocab.get('a')
    vocab.add('a', 10)
    # not in vocab yet, skips.
    vocab.add('b', 5)
    vocab.add('d', 12)
    vocab.trim(7, 11)
    vocab.freeze()
    self.assertEqual(vocab.get('b'), 0)
    self.assertEqual(vocab.get('c'), 0)
    self.assertEqual(len(vocab), 2)
    self.assertEqual(vocab.get('a'), 1)


if __name__ == '__main__':
  tf.test.main()
