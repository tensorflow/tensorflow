# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

# coding=utf-8
"""Tests for tensorflow.data.experimental.core.deprecated.text.subword_text_encoder."""
from __future__ import unicode_literals

import os

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from tensorflow_datasets import testing
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.deprecated.text import subword_text_encoder
from tensorflow.data.experimental.core.deprecated.text import text_encoder

TEST_DATA_DIR = os.path.join(utils.tfds_path(), 'testing', 'test_data')


class SubwordTextEncoderTest(parameterized.TestCase, testing.TestCase):

  def setUp(self):
    super(SubwordTextEncoderTest, self).setUp()
    # Vocab ids will be (offset for pad=0):
    #                  1       2       3      4      5
    self.vocab_list = ['foo_', 'bar_', 'foo', 'bar', '<EOS>']
    self.encoder = subword_text_encoder.SubwordTextEncoder(
        vocab_list=self.vocab_list)

  def test_vocab_size(self):
    # Bytes + pad + subwords
    self.assertEqual((256 + 1 + len(self.vocab_list)), self.encoder.vocab_size)

  @parameterized.parameters(
      ('foo bar', [1, 4]),
      ('foobar foo bar<EOS>bar', [3, 2, 1, 4, 5, 4]),
      # Respects whitespace
      ('bar <EOS>bar', [2, 5, 4]),
      ('bar <EOS> bar', [2, 5, 38, 4]),
      ('bar<EOS> bar', [4, 5, 38, 4]),
      # Invertible even with oov, respecting underscores and backslashes
      ('a_b!', [103, 101, 104, 39]),
      ('foo \\bar_!', [3, 38, 98, 4, 101, 39]),
      ('foo \\\\bar_!', [3, 38, 98, 98, 4, 101, 39]),
      ('hello world!', None),
      ('foo_ bar', None),
      ('foo _ bar', None),
      ('foo _bar', None),
      ('hello_world', None),
      ('hello_ world', None),
      ('hello _ world', None),
      ('hello _world', None),
      ('_', None),
      # Test that the underscore replacement string is unharmed
      ('\\&undsc', None),
      # Unicode encoded as bytes but decoded back to unicode character
      ('你', [234, 195, 166]),
  )
  def test_encode_decode(self, text, expected_ids):
    ids = self.encoder.encode(text)
    # Test ids match if ids provided
    if expected_ids:
      self.assertEqual(expected_ids, ids)
    # Test invertibility
    self.assertEqual(tf.compat.as_text(text), self.encoder.decode(ids))

  def test_bad_bytes(self):
    valid_unicode = '你'
    bad_bytes = [220 + len(self.vocab_list) + 1]
    bad_ids = self.encoder.encode('你') + bad_bytes
    text = self.encoder.decode(bad_ids)
    # Valid unicode character preserved
    self.assertEqual(valid_unicode, text[0])
    # Invalid byte converted to unknown character
    self.assertEqual('\uFFFD', text[1])

  def test_vocab_file(self):
    vocab_file = os.path.join(self.get_temp_dir(), 'vocab')
    self.encoder.save_to_file(vocab_file)
    encoder = subword_text_encoder.SubwordTextEncoder.load_from_file(vocab_file)
    self.assertEqual(encoder.subwords, self.vocab_list)


class SubwordTextEncoderBuildTest(testing.TestCase):

  def test_build(self):
    text_gen = lorem_ipsum_generator
    build_fn = subword_text_encoder.SubwordTextEncoder.build_from_corpus
    encoder = build_fn(text_gen(), 300)
    # Created some subwords
    self.assertGreater(encoder.vocab_size, text_encoder.NUM_BYTES + 1)

    base_encoder = subword_text_encoder.SubwordTextEncoder(vocab_list=[])
    for line in text_gen():
      # Invertible
      encoded = encoder.encode(line)
      self.assertEqual(line, encoder.decode(encoded))
      # Shorter than base
      if len(line) > 2:
        self.assertLess(len(encoded), len(base_encoder.encode(line)))

  def test_build_with_unicode(self):
    text_gen = lorem_ipsum_zh_generator
    build_fn = subword_text_encoder.SubwordTextEncoder.build_from_corpus
    encoder = build_fn(text_gen(), 300)
    # Created some subwords
    self.assertGreater(encoder.vocab_size, text_encoder.NUM_BYTES + 1)

    base_encoder = subword_text_encoder.SubwordTextEncoder(vocab_list=[])
    for line in text_gen():
      # Invertible
      encoded = encoder.encode(line)
      self.assertEqual(line, encoder.decode(encoded))
      # Shorter than base
      if len(line) > 2:
        self.assertLess(len(encoded), len(base_encoder.encode(line)))

  def test_max_subword_length(self):
    text_gen = lorem_ipsum_generator
    build_fn = subword_text_encoder.SubwordTextEncoder.build_from_corpus
    encoder = build_fn(text_gen(), 300, max_subword_length=1)
    # Created no subwords because there are no unicode characters in lorem ipsum
    # and single byte subwords are skipped because all bytes are in the vocab by
    # default.
    self.assertEqual(encoder.vocab_size, text_encoder.NUM_BYTES + 1)
    self.assertEqual(len(encoder.subwords), 0)

    # Not the case when there are unicode characters
    text_gen = lorem_ipsum_zh_generator
    build_fn = subword_text_encoder.SubwordTextEncoder.build_from_corpus
    encoder = build_fn(text_gen(), 300, max_subword_length=1)
    self.assertGreater(encoder.vocab_size, text_encoder.NUM_BYTES + 1)
    self.assertGreater(len(encoder.subwords), 0)

  def test_max_chars(self):
    text_gen = lorem_ipsum_zh_generator
    build_fn = subword_text_encoder.SubwordTextEncoder.build_from_corpus
    encoder = build_fn(text_gen(), 300, max_corpus_chars=1)
    self.assertGreater(encoder.vocab_size, text_encoder.NUM_BYTES + 1)
    self.assertEqual(1, len(encoder.subwords))
    first_letter = next(lorem_ipsum_zh_generator())[0]
    self.assertEqual(first_letter, encoder.subwords[0])

  def test_reserved_tokens(self):
    text_gen = lorem_ipsum_generator
    build_fn = subword_text_encoder.SubwordTextEncoder.build_from_corpus
    encoder = build_fn(text_gen(), 300, reserved_tokens=['<EOS>', '<EOD>'])
    self.assertEqual(2, encoder.encode('Lorem<EOD>')[-1])
    self.assertEqual(2, encoder.encode('Lorem<EOD>a')[-2])
    self.assertEqual(2, encoder.encode('Lorem<EOD>{')[-2])
    self.assertEqual(2, encoder.encode('Lorem<EOD> ')[-2])
    self.assertEqual('<EOS> <EOD>', encoder.decode([1, 78, 2]))
    self.assertEqual(['<EOS>', '<EOD>'], encoder.subwords[:2])


def _yield_lines_from_file(txt_file):
  with tf.io.gfile.GFile(txt_file, 'rb') as f:
    for line in f:
      yield tf.compat.as_text(line)


def lorem_ipsum_generator():
  txt_file = os.path.join(TEST_DATA_DIR, 'lorem_ipsum.txt')
  return _yield_lines_from_file(txt_file)


def lorem_ipsum_zh_generator():
  txt_file = os.path.join(TEST_DATA_DIR, 'lorem_ipsum_zh.txt')
  return _yield_lines_from_file(txt_file)


if __name__ == '__main__':
  testing.test_main()
