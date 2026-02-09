# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tests for tensorflow_text.python.tools.wordpiece_vocab.wordpiece_tokenizer_learner_lib."""

import collections
import logging

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow_text.tools.wordpiece_vocab import wordpiece_tokenizer_learner_lib as learner


class ExtractCharTokensTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {'testcase_name': 'UniqueChars',
       'word_counts': [('abc', 1), ('def', 1), ('ghi', 1)],
       'expected': {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'}},
      {'testcase_name': 'RepeatedChars',
       'word_counts': [('hello', 1), ('world!', 1)],
       'expected': {'h', 'e', 'l', 'o', 'w', 'r', 'd', '!'}})

  def testExtractCharTokens(self, word_counts, expected):
    actual = learner.extract_char_tokens(word_counts)
    self.assertEqual(expected, actual)


class EnsureAllTokensExistTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {'testcase_name': 'EmptyDict', 'input_tokens': {'a', 'b', 'c'},
       'output_tokens': {}, 'expected_tokens': {'a': 1, 'b': 1, 'c': 1},
       'include_joiner': False},
      {'testcase_name': 'SomeTokensExist', 'input_tokens': {'a', 'b', 'c'},
       'output_tokens': {'a': 2, 'd': 3, 'e': 1},
       'expected_tokens': {'a': 2, 'b': 1, 'c': 1, 'd': 3, 'e': 1},
       'include_joiner': False},
      {'testcase_name': 'SomeTokensExistWithJoiner',
       'input_tokens': {'a', 'b', 'c'},
       'output_tokens': {'a': 2, 'd': 3, 'e': 1},
       'expected_tokens': {'a': 2, 'b': 1, 'c': 1, 'd': 3, 'e': 1, '##a': 1,
                           '##b': 1, '##c': 1}, 'include_joiner': True})

  def testEnsureAllTokensExist(self, input_tokens, output_tokens,
                               expected_tokens, include_joiner):
    joiner = '##'
    new_tokens = learner.ensure_all_tokens_exist(input_tokens, output_tokens,
                                                 include_joiner, joiner)
    self.assertEqual(new_tokens, expected_tokens)


class GetSplitIndicesTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {'testcase_name': 'ValidWordNoJoiner', 'word': 'hello',
       'expected_indices': [2, 5], 'include_joiner': False},
      {'testcase_name': 'ValidWordWithJoiner', 'word': 'hello',
       'expected_indices': [2, 3, 5], 'include_joiner': True},
      {'testcase_name': 'InvalidSplitIndices', 'word': 'world',
       'expected_indices': None, 'include_joiner': False})

  def testGetSplitIndices(self, word, expected_indices, include_joiner):
    joiner = '##'
    curr_tokens = {'he': 1, 'llo': 1, '##l': 1, '##lo': 1, '!': 1}
    indices = learner.get_split_indices(word, curr_tokens, include_joiner,
                                        joiner)
    self.assertEqual(indices, expected_indices)


class GetSearchThreshsTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {'testcase_name': 'ThreshsWithinMinMax', 'upper_thresh': 200,
       'lower_thresh': 5, 'expected_upper': 200, 'expected_lower': 5},
      {'testcase_name': 'ThreshsOutsideMinMax', 'upper_thresh': 10000,
       'lower_thresh': 2, 'expected_upper': 292, 'expected_lower': 3})

  def testGetSearchThreshs(self, upper_thresh, lower_thresh, expected_upper,
                           expected_lower):
    word_counts = [('apple', 3), ('banana', 292), ('cucumber', 5)]
    upper, lower = learner.get_search_threshs(word_counts, upper_thresh,
                                              lower_thresh)
    self.assertEqual(upper, expected_upper)
    self.assertEqual(lower, expected_lower)


class GetInputWordsTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {'testcase_name': 'TokenTooLong',
       'word_counts': [('blah', 1), ('blehhhhhhhh', 2)],
       'expected_counts': [('blah', 1)]},
      {'testcase_name': 'TokenInReserved',
       'word_counts': [('q', 1), ('r', 2), ('<s>', 35), ('t', 3), ('u', 4)],
       'expected_counts': [('q', 1), ('r', 2), ('t', 3), ('u', 4)]})

  def testGetInputWords(self, word_counts, expected_counts):
    max_token_length = 10
    reserved_tokens = ['<unk>', '<s>', '</s>']
    new_counts = learner.get_input_words(word_counts, reserved_tokens,
                                         max_token_length)

    self.assertEqual(new_counts, expected_counts)


class GetAllowedCharsTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {'testcase_name': 'MoreCharsThanMax',
       'word_counts': [('hello', 1), ('world', 1)],
       'expected_chars': {'l', 'o', 'd', 'e', 'h'}},
      {'testcase_name': 'DifferentFrequency',
       'word_counts': [('hello', 1), ('world', 2)],
       'expected_chars': {'l', 'o', 'd', 'r', 'w'}})

  def testGetAllowedChars(self, word_counts, expected_chars):
    max_unique_chars = 5
    chars = learner.get_allowed_chars(word_counts, max_unique_chars)
    self.assertEqual(chars, expected_chars)


class FilterInputWordsTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {
          'testcase_name': 'TokenHasUnallowedChars',
          'word_counts': [('bad', 1), ('had', 2), ('bag', 1), ('cat', 5)],
          'expected_counts': [('had', 2), ('bad', 1), ('bag', 1)]
      }, {
          'testcase_name': 'TooManyInputTokens',
          'word_counts': [('bad', 1), ('had', 2), ('bag', 1), ('bed', 5),
                          ('head', 7)],
          'expected_counts': [('head', 7), ('bed', 5), ('had', 2), ('bad', 1)]
      })

  def testFilterInputWords(self, word_counts, expected_counts):
    allowed_chars = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}
    max_input_tokens = 4
    filtered_counts = learner.filter_input_words(word_counts, allowed_chars,
                                                 max_input_tokens)
    self.assertEqual(filtered_counts, expected_counts)


class GenerateFinalVocabularyTest(absltest.TestCase):

  def setUp(self):
    super(GenerateFinalVocabularyTest, self).setUp()
    self.reserved_tokens = ['<unk>', '<s>', '</s>']
    self.char_tokens = ['c', 'a', 'b']
    self.curr_tokens = {'my': 2, 'na': 5, '##me': 1, 'is': 2}
    self.vocab_array = ['<unk>', '<s>', '</s>', 'a', 'b', 'c', 'na', 'is', 'my',
                        '##me']

  def testGenerateFinalVocab(self):
    final_vocab = learner.generate_final_vocabulary(self.reserved_tokens,
                                                    self.char_tokens,
                                                    self.curr_tokens)
    self.assertEqual(final_vocab, self.vocab_array)


class LearnWithThreshTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {'testcase_name': 'LearnWithOneIteration',
       'word_counts': [('apple', 1), ('app', 1)],
       'thresh': 1,
       'expected_vocab': ['a', 'e', 'l', 'p', 'app', 'apple', 'le', 'ple', 'pp',
                          'pple'],
       'params': learner.Params(upper_thresh=4, lower_thresh=1,
                                num_iterations=1, max_input_tokens=1000,
                                max_token_length=50, max_unique_chars=5,
                                vocab_size=10, slack_ratio=0,
                                include_joiner_token=False, joiner='##',
                                reserved_tokens=[])},
      {'testcase_name': 'LearnWithTwoIterations',
       'word_counts': [('apple', 1), ('app', 1)],
       'thresh': 1,
       'expected_vocab': ['a', 'e', 'l', 'p', 'app', 'apple'],
       'params': learner.Params(upper_thresh=4, lower_thresh=1,
                                num_iterations=2, max_input_tokens=1000,
                                max_token_length=50, max_unique_chars=5,
                                vocab_size=10, slack_ratio=0,
                                include_joiner_token=False, joiner='##',
                                reserved_tokens=[])},
      {'testcase_name': 'LearnWithHigherThresh',
       'word_counts': [('apple', 1), ('app', 2)],
       'thresh': 2,
       'expected_vocab': ['a', 'e', 'l', 'p', 'app', 'pp'],
       'params': learner.Params(upper_thresh=4, lower_thresh=1,
                                num_iterations=1, max_input_tokens=1000,
                                max_token_length=50, max_unique_chars=5,
                                vocab_size=10, slack_ratio=0,
                                include_joiner_token=False, joiner='##',
                                reserved_tokens=[])})

  def testLearnWithThresh(self, word_counts, thresh, expected_vocab, params):
    vocab = learner.learn_with_thresh(word_counts, thresh, params)
    self.assertEqual(vocab, expected_vocab)


class LearnBinarySearchTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {'testcase_name': 'ReachesVocabSize',
       'word_counts': [('apple', 2), ('peach', 1), ('pear', 1)],
       'lower': 1, 'upper': 10, 'delta': 0,
       'expected_vocab': ['a', 'c', 'e', 'h', 'l', 'p', 'r', 'apple', 'peach',
                          'pear'],
       'params': learner.Params(upper_thresh=4, lower_thresh=1,
                                num_iterations=4, max_input_tokens=1000,
                                max_token_length=50, max_unique_chars=50,
                                vocab_size=10, slack_ratio=0,
                                include_joiner_token=False, joiner='##',
                                reserved_tokens=[])},
      {'testcase_name': 'VocabSizeWithinSlack',
       'word_counts': [('apple', 2), ('peach', 1), ('pear', 1), ('app', 2)],
       'lower': 1, 'upper': 10, 'delta': 6,
       'expected_vocab': ['a', 'c', 'e', 'h', 'l', 'p', 'r'],
       'params': learner.Params(upper_thresh=4, lower_thresh=1,
                                num_iterations=4, max_input_tokens=1000,
                                max_token_length=50, max_unique_chars=50,
                                vocab_size=12, slack_ratio=0.5,
                                include_joiner_token=False, joiner='##',
                                reserved_tokens=[])})

  def testBinarySearch(self, word_counts, lower, upper, delta, expected_vocab,
                       params):
    vocab = learner.learn_binary_search(word_counts, lower, upper, params)
    self.assertAlmostEqual(len(vocab), params.vocab_size, delta=delta)
    self.assertLessEqual(len(vocab), params.vocab_size)
    self.assertEqual(vocab, expected_vocab)


class CountWordsTest(parameterized.TestCase):

  def test_count_lists(self):
    data = [['aaa', 'bb', 'c'], ['aaa', 'aaa'], ['c']]
    counts = learner.count_words(data)

    self.assertEqual(counts, collections.Counter({'aaa': 3, 'bb': 1, 'c': 2}))

  def test_count_numpy_gen(self):

    def get_words():
      yield np.array(['aaa', 'bb', 'c'])
      yield np.array(['aaa', 'aaa'])
      yield np.array(['c'])

    counts = learner.count_words(get_words())

    self.assertEqual(counts, collections.Counter({'aaa': 3, 'bb': 1, 'c': 2}))

  def test_count_ragged_dataset(self):
    ds = dataset_ops.DatasetV2.from_tensor_slices(['aaa bb c', 'aaa aaa', 'c'])
    ds = ds.map(ragged_string_ops.string_split_v2)

    counts = learner.count_words(ds)

    self.assertEqual(counts, collections.Counter({'aaa': 3, 'bb': 1, 'c': 2}))


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  absltest.main()
