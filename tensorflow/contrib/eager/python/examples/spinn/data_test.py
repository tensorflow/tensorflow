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
"""Unit tests for SPINN data module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python.examples.spinn import data


class DataTest(tf.test.TestCase):

  def setUp(self):
    super(DataTest, self).setUp()
    self._temp_data_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._temp_data_dir)
    super(DataTest, self).tearDown()

  def testGenNonParenthesisWords(self):
    seq_with_parse = (
        "( Man ( ( ( ( ( wearing pass ) ( on ( a lanyard ) ) ) and "
        ") ( standing ( in ( ( a crowd ) ( of people ) ) ) ) ) . ) )")
    self.assertEqual(
        ["man", "wearing", "pass", "on", "a", "lanyard", "and", "standing",
         "in", "a", "crowd", "of", "people", "."],
        data.get_non_parenthesis_words(seq_with_parse.split(" ")))

  def testGetShiftReduce(self):
    seq_with_parse = (
        "( Man ( ( ( ( ( wearing pass ) ( on ( a lanyard ) ) ) and "
        ") ( standing ( in ( ( a crowd ) ( of people ) ) ) ) ) . ) )")
    self.assertEqual(
        [3, 3, 3, 2, 3, 3, 3, 2, 2, 2, 3, 2, 3, 3, 3, 3, 2, 3, 3, 2, 2, 2, 2, 2,
         3, 2, 2], data.get_shift_reduce(seq_with_parse.split(" ")))

  def testPadAndReverseWordIds(self):
    id_sequences = [[0, 2, 3, 4, 5],
                    [6, 7, 8],
                    [9, 10, 11, 12, 13, 14, 15, 16]]
    self.assertAllClose(
        [[1, 1, 1, 1, 5, 4, 3, 2, 0],
         [1, 1, 1, 1, 1, 1, 8, 7, 6],
         [1, 16, 15, 14, 13, 12, 11, 10, 9]],
        data.pad_and_reverse_word_ids(id_sequences))

  def testPadTransitions(self):
    unpadded = [[3, 3, 3, 2, 2, 2, 2],
                [3, 3, 2, 2, 2]]
    self.assertAllClose(
        [[3, 3, 3, 2, 2, 2, 2],
         [3, 3, 2, 2, 2, 1, 1]],
        data.pad_transitions(unpadded))

  def testCalculateBins(self):
    length2count = {
        1: 10,
        2: 15,
        3: 25,
        4: 40,
        5: 35,
        6: 10}
    self.assertEqual([2, 3, 4, 5, 6],
                     data.calculate_bins(length2count, 20))
    self.assertEqual([3, 4, 6], data.calculate_bins(length2count, 40))
    self.assertEqual([4, 6], data.calculate_bins(length2count, 60))

  def testLoadVoacbulary(self):
    snli_1_0_dir = os.path.join(self._temp_data_dir, "snli/snli_1.0")
    fake_train_file = os.path.join(snli_1_0_dir, "snli_1.0_train.txt")
    fake_dev_file = os.path.join(snli_1_0_dir, "snli_1.0_dev.txt")
    os.makedirs(snli_1_0_dir)

    with open(fake_train_file, "wt") as f:
      f.write("gold_label\tsentence1_binary_parse\tsentence2_binary_parse\t"
              "sentence1_parse\tsentence2_parse\tsentence1\tsentence2\t"
              "captionID\tpairID\tlabel1\tlabel2\tlabel3\tlabel4\tlabel5\n")
      f.write("neutral\t( ( Foo bar ) . )\t( ( foo baz ) . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")
    with open(fake_dev_file, "wt") as f:
      f.write("gold_label\tsentence1_binary_parse\tsentence2_binary_parse\t"
              "sentence1_parse\tsentence2_parse\tsentence1\tsentence2\t"
              "captionID\tpairID\tlabel1\tlabel2\tlabel3\tlabel4\tlabel5\n")
      f.write("neutral\t( ( Quux quuz ) ? )\t( ( Corge grault ) ! )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Quux quuz?\t.Corge grault!\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")

    vocab = data.load_vocabulary(self._temp_data_dir)
    self.assertSetEqual(
        {".", "?", "!", "foo", "bar", "baz", "quux", "quuz", "corge", "grault"},
        vocab)

  def testLoadVoacbularyWithoutFileRaisesError(self):
    with self.assertRaisesRegexp(ValueError, "Cannot find SNLI data files at"):
      data.load_vocabulary(self._temp_data_dir)

    os.makedirs(os.path.join(self._temp_data_dir, "snli"))
    with self.assertRaisesRegexp(ValueError, "Cannot find SNLI data files at"):
      data.load_vocabulary(self._temp_data_dir)

    os.makedirs(os.path.join(self._temp_data_dir, "snli/snli_1.0"))
    with self.assertRaisesRegexp(ValueError, "Cannot find SNLI data files at"):
      data.load_vocabulary(self._temp_data_dir)

  def testLoadWordVectors(self):
    glove_dir = os.path.join(self._temp_data_dir, "glove")
    os.makedirs(glove_dir)
    glove_file = os.path.join(glove_dir, "glove.42B.300d.txt")

    words = [".", ",", "foo", "bar", "baz"]
    with open(glove_file, "wt") as f:
      for i, word in enumerate(words):
        f.write("%s " % word)
        for j in range(data.WORD_VECTOR_LEN):
          f.write("%.5f" % (i * 0.1))
          if j < data.WORD_VECTOR_LEN - 1:
            f.write(" ")
          else:
            f.write("\n")

    vocab = {"foo", "bar", "baz", "qux", "."}
    # Notice that "qux" is not present in `words`.
    word2index, embed = data.load_word_vectors(self._temp_data_dir, vocab)

    self.assertEqual(6, len(word2index))
    self.assertEqual(0, word2index["<unk>"])
    self.assertEqual(1, word2index["<pad>"])
    self.assertEqual(2, word2index["."])
    self.assertEqual(3, word2index["foo"])
    self.assertEqual(4, word2index["bar"])
    self.assertEqual(5, word2index["baz"])
    self.assertEqual((6, data.WORD_VECTOR_LEN), embed.shape)
    self.assertAllClose([0.0] * data.WORD_VECTOR_LEN, embed[0, :])
    self.assertAllClose([0.0] * data.WORD_VECTOR_LEN, embed[1, :])
    self.assertAllClose([0.0] * data.WORD_VECTOR_LEN, embed[2, :])
    self.assertAllClose([0.2] * data.WORD_VECTOR_LEN, embed[3, :])
    self.assertAllClose([0.3] * data.WORD_VECTOR_LEN, embed[4, :])
    self.assertAllClose([0.4] * data.WORD_VECTOR_LEN, embed[5, :])

  def testLoadWordVectorsWithoutFileRaisesError(self):
    vocab = {"foo", "bar", "baz", "qux", "."}
    with self.assertRaisesRegexp(
        ValueError, "Cannot find GloVe embedding file at"):
      data.load_word_vectors(self._temp_data_dir, vocab)

    os.makedirs(os.path.join(self._temp_data_dir, "glove"))
    with self.assertRaisesRegexp(
        ValueError, "Cannot find GloVe embedding file at"):
      data.load_word_vectors(self._temp_data_dir, vocab)

  def _createFakeSnliData(self, fake_snli_file):
    # Four sentences in total.
    with open(fake_snli_file, "wt") as f:
      f.write("gold_label\tsentence1_binary_parse\tsentence2_binary_parse\t"
              "sentence1_parse\tsentence2_parse\tsentence1\tsentence2\t"
              "captionID\tpairID\tlabel1\tlabel2\tlabel3\tlabel4\tlabel5\n")
      f.write("neutral\t( ( Foo bar ) . )\t( ( foo . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")
      f.write("contradiction\t( ( Bar foo ) . )\t( ( baz . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")
      f.write("entailment\t( ( Quux quuz ) . )\t( ( grault . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")
      f.write("entailment\t( ( Quuz quux ) . )\t( ( garply . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")

  def _createFakeGloveData(self, glove_file):
    words = [".", "foo", "bar", "baz", "quux", "quuz", "grault", "garply"]
    with open(glove_file, "wt") as f:
      for i, word in enumerate(words):
        f.write("%s " % word)
        for j in range(data.WORD_VECTOR_LEN):
          f.write("%.5f" % (i * 0.1))
          if j < data.WORD_VECTOR_LEN - 1:
            f.write(" ")
          else:
            f.write("\n")

  def testEncodeSingleSentence(self):
    snli_1_0_dir = os.path.join(self._temp_data_dir, "snli/snli_1.0")
    fake_train_file = os.path.join(snli_1_0_dir, "snli_1.0_train.txt")
    os.makedirs(snli_1_0_dir)
    self._createFakeSnliData(fake_train_file)
    vocab = data.load_vocabulary(self._temp_data_dir)
    glove_dir = os.path.join(self._temp_data_dir, "glove")
    os.makedirs(glove_dir)
    glove_file = os.path.join(glove_dir, "glove.42B.300d.txt")
    self._createFakeGloveData(glove_file)
    word2index, _ = data.load_word_vectors(self._temp_data_dir, vocab)

    sentence_variants = [
        "( Foo ( ( bar baz ) . ) )",
        " ( Foo ( ( bar baz ) . ) ) ",
        "( Foo ( ( bar baz ) . )  )"]
    for sentence in sentence_variants:
      word_indices, shift_reduce = data.encode_sentence(sentence, word2index)
      self.assertEqual(np.int64, word_indices.dtype)
      self.assertEqual((5, 1), word_indices.shape)
      self.assertAllClose(
          np.array([[3, 3, 3, 2, 3, 2, 2]], dtype=np.int64).T, shift_reduce)

  def testSnliData(self):
    snli_1_0_dir = os.path.join(self._temp_data_dir, "snli/snli_1.0")
    fake_train_file = os.path.join(snli_1_0_dir, "snli_1.0_train.txt")
    os.makedirs(snli_1_0_dir)
    self._createFakeSnliData(fake_train_file)

    glove_dir = os.path.join(self._temp_data_dir, "glove")
    os.makedirs(glove_dir)
    glove_file = os.path.join(glove_dir, "glove.42B.300d.txt")
    self._createFakeGloveData(glove_file)

    vocab = data.load_vocabulary(self._temp_data_dir)
    word2index, _ = data.load_word_vectors(self._temp_data_dir, vocab)

    train_data = data.SnliData(fake_train_file, word2index)
    self.assertEqual(4, train_data.num_batches(1))
    self.assertEqual(2, train_data.num_batches(2))
    self.assertEqual(2, train_data.num_batches(3))
    self.assertEqual(1, train_data.num_batches(4))

    generator = train_data.get_generator(2)()
    for _ in range(2):
      label, prem, prem_trans, hypo, hypo_trans = next(generator)
      self.assertEqual(2, len(label))
      self.assertEqual((4, 2), prem.shape)
      self.assertEqual((5, 2), prem_trans.shape)
      self.assertEqual((3, 2), hypo.shape)
      self.assertEqual((3, 2), hypo_trans.shape)


if __name__ == "__main__":
  tf.test.main()
