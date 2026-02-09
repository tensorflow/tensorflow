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

"""Tests for tensorflow_text.python.tools.wordpiece_vocab.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import tempfile

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import tensorflow.compat.v1 as tf
from tensorflow_text.tools.wordpiece_vocab import utils


class FilterTokensByLangTest(absltest.TestCase):

  def setUp(self):
    super(FilterTokensByLangTest, self).setUp()
    self.sample_input = [{'lang': 'en',
                          'tokens': ['I', 'like', 'pie', '.']}]

  def testLangInLangSet(self):
    with TestPipeline() as p:
      tokens = p | beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.FilterTokensByLang({'en'}))
      assert_that(result, equal_to([('I', 'en'),
                                    ('like', 'en'),
                                    ('pie', 'en'),
                                    ('.', 'en')]))

  def testLangNotInLangSet(self):
    with TestPipeline() as p:
      tokens = p | beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.FilterTokensByLang({'fr'}))
      assert_that(result, equal_to([]))

  def testLangNotInLangSetIncludeOthers(self):
    with TestPipeline() as p:
      tokens = p | beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.FilterTokensByLang({'fr'}, True))
      assert_that(result, equal_to([('I', 'other'),
                                    ('like', 'other'),
                                    ('pie', 'other'),
                                    ('.', 'other')]))


class CompareValues(beam.DoFn):

  def process(self, element):
    return [element['en'] < element['fr']]


class CalculateCoefficientsTest(absltest.TestCase):

  def setUp(self):
    super(CalculateCoefficientsTest, self).setUp()
    self.sample_input = [('I', 'en'), ('really', 'en'),
                         ('like', 'en'), ('pie', 'en'),
                         ('.', 'en'), ('Je', 'fr'),
                         ('suis', 'fr'), ('une', 'fr'),
                         ('fille', 'fr'), ('.', 'fr')]

  def testEqual(self):
    with TestPipeline() as p:
      tokens = p | beam.Create(self.sample_input)
      result = tokens | beam.CombineGlobally(utils.CalculateCoefficients(0.5))
      assert_that(result, equal_to([{'en': 1.0, 'fr': 1.0}]))

  def testNotEqual(self):
    with TestPipeline() as p:
      sample_input = [('I', 'en'), ('kind', 'en'), ('of', 'en'), ('like', 'en'),
                      ('to', 'en'), ('eat', 'en'), ('pie', 'en'), ('!', 'en'),
                      ('Je', 'fr'), ('suis', 'fr'), ('une', 'fr'),
                      ('fille', 'fr'), ('.', 'fr')]
      tokens = p | beam.Create(sample_input)
      result = (tokens
                | beam.CombineGlobally(utils.CalculateCoefficients(0.5))
                | beam.ParDo(CompareValues()))
      assert_that(result, equal_to([True]))


class ExponentialSmoothingTest(absltest.TestCase):

  def setUp(self):
    super(ExponentialSmoothingTest, self).setUp()
    self.sample_input = [('Hello', 'en'), (',', 'en'),
                         ('world', 'en'), ('!', 'en'),
                         ('Bonjour', 'fr'), ('.', 'fr')]
    self.coeffs = [{'en': 0.75, 'fr': 1.5}]

  def testBasic(self):
    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(self.sample_input)
      coeffs = p | 'CreateCoeffs' >> beam.Create(self.coeffs)
      result = tokens | beam.ParDo(
          utils.ExponentialSmoothing(), beam.pvalue.AsSingleton(coeffs))
      assert_that(result, equal_to([('Hello', 0.75), (',', 0.75),
                                    ('world', 0.75), ('!', 0.75),
                                    ('Bonjour', 1.5), ('.', 1.5)]))


class FilterByCountTest(absltest.TestCase):

  def setUp(self):
    super(FilterByCountTest, self).setUp()
    self.sample_input = [('one', 1), ('two', 2), ('three', 3), ('four', 4)]
    self.max_token_length = 50

  def testBelowThreshold(self):
    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.FilterByCount(self.max_token_length,
                                                       min_token_frequency=2))
      assert_that(result, equal_to([('three', 3), ('four', 4)]))

  def testTokenTooLong(self):
    sample_input = [('one', 1), ('two', 2), ('three', 3), ('four', 4),
                    ('qwertyuiopasdfghjklzxcvbnmqwertyuiopasdfghjklzxcvbnm', 5),
                    ('blah', 20)]

    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(sample_input)
      result = tokens | beam.ParDo(utils.FilterByCount(self.max_token_length,
                                                       min_token_frequency=2))
      assert_that(result, equal_to([('three', 3), ('four', 4), ('blah', 20)]))


class SortByCountTest(absltest.TestCase):

  def setUp(self):
    super(SortByCountTest, self).setUp()
    self.sample_input = [('a', 5), ('b', 2), ('c', 9), ('d', 4)]

  def testUnsorted(self):
    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(self.sample_input)
      result = tokens | beam.CombineGlobally(utils.SortByCount())
      assert_that(result, equal_to([[('c', 9), ('a', 5), ('d', 4), ('b', 2)]]))


class CompileTokenizationInfoTest(absltest.TestCase):

  def setUp(self):
    super(CompileTokenizationInfoTest, self).setUp()
    self.sample_input = [{'lang': 'en',
                          'num_non_unk_wordpieces': 4,
                          'num_dropped_chars': 2,
                          'num_preserved_chars': 13,
                          'wordpieces': ['the', 'app', '##le',
                                         'sauce', '[UNK]']},
                         {'lang': 'fr',
                          'num_non_unk_wordpieces': 5,
                          'num_dropped_chars': 0,
                          'num_preserved_chars': 14,
                          'wordpieces': ['bon', '##jour', 'bon', '##soir']}]

  def testTwoLangs(self):
    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.CompileTokenizationInfo())
      assert_that(result, equal_to([{
          'lang': 'en',
          'count': 1,
          'num_preserved_chars': 13,
          'num_dropped_chars': 2,
          'num_non_unk_wordpieces': 4,
          'preserved_ratio': [13/4],
          'dropped_ratio': [2/15],
          'wordpieces': collections.Counter(['the', 'app', '##le', 'sauce'])
      }, {
          'lang': 'fr',
          'count': 1,
          'num_preserved_chars': 14,
          'num_dropped_chars': 0,
          'num_non_unk_wordpieces': 5,
          'preserved_ratio': [14/5],
          'dropped_ratio': [0],
          'wordpieces': collections.Counter(['bon', '##jour', 'bon', '##soir'])
      }]))


class AggregateLangTest(absltest.TestCase):

  def setUp(self):
    super(AggregateLangTest, self).setUp()
    self.aggregator = utils.AggregateLang()
    self.sample_input = [{
        'lang': 'en',
        'count': 1,
        'num_preserved_chars': 13,
        'num_dropped_chars': 2,
        'num_non_unk_wordpieces': 4,
        'preserved_ratio': [13/4],
        'dropped_ratio': [2/15],
        'wordpieces': collections.Counter(['the', 'app', '##le', 'sauce'])
        }, {
            'lang': 'en',
            'count': 1,
            'num_preserved_chars': 11,
            'num_dropped_chars': 0,
            'num_non_unk_wordpieces': 4,
            'preserved_ratio': [11/4],
            'dropped_ratio': [0],
            'wordpieces': collections.Counter(['the', 'app', 'st', '##ore'])
            }]

  def testMultiEntryOneLang(self):
    expected_output = self.aggregator.create_accumulator()
    expected_output['en'] = {
        'count': 2,
        'num_preserved_chars': 24,
        'num_dropped_chars': 2,
        'num_non_unk_wordpieces': 8,
        'preserved_ratio': [13/4, 11/4],
        'dropped_ratio': [2/15, 0],
        'wordpieces': collections.Counter({'the': 2, 'app': 2, '##le': 1,
                                           'sauce': 1, 'st': 1, '##ore': 1})}
    # Test create_accumulator.
    accumulator = self.aggregator.create_accumulator()
    # Test add_input.
    for element in self.sample_input:
      accumulator = self.aggregator.add_input(accumulator, element)
    # Test merge_accumulators.
    merged = self.aggregator.merge_accumulators([
        accumulator, self.aggregator.create_accumulator()])
    # Test extract_output.
    output = self.aggregator.extract_output(merged)
    self.assertEqual(output, expected_output)


class CalculateMetricsTest(absltest.TestCase):

  def setUp(self):
    super(CalculateMetricsTest, self).setUp()
    self.info_dict = {
        'en': {
            'count': 2,
            'num_preserved_chars': 24,
            'num_dropped_chars': 2,
            'num_non_unk_wordpieces': 8,
            'preserved_ratio': [2, 3],
            'dropped_ratio': [0.5, 0],
            'wordpieces': collections.Counter({'the': 2, 'le': 1, '##sson': 1,
                                               'plan': 1, '##s': 1})},
        'fr': {
            'count': 2,
            'num_preserved_chars': 24,
            'num_dropped_chars': 2,
            'num_non_unk_wordpieces': 8,
            'preserved_ratio': [5, 7],
            'dropped_ratio': [0.4, 0.6],
            'wordpieces': collections.Counter({'bon': 2, 'le': 2, 'jour': 1,
                                               'soir': 1, 'homme': 1})}}
    self.metrics = utils.CalculateMetrics()

  def testListMean(self):
    test_list = [1, 2, 3, 4, 5]
    mean = self.metrics._get_list_mean(test_list)
    self.assertEqual(mean, 3)

  def testMicroCompressionRatio(self):
    fr_micro_compression = self.metrics._get_micro_compression_ratio(
        self.info_dict['fr'])
    self.assertEqual(fr_micro_compression, 3)

  def testMacroCompressionRatio(self):
    en_macro_compression = self.metrics._get_macro_compression_ratio(
        self.info_dict['en'])
    self.assertEqual(en_macro_compression, 2.5)

  def testMicroDroppedCharPercent(self):
    en_micro_dropped_char = self.metrics._get_micro_dropped_char_percent(
        self.info_dict['en'])
    self.assertEqual(en_micro_dropped_char, 100/13)

  def testMacroDroppedCharPercent(self):
    fr_macro_dropped_char = self.metrics._get_macro_dropped_char_percent(
        self.info_dict['fr'])
    self.assertEqual(fr_macro_dropped_char, 50.0)

  def testWordpieceOverlapFrench(self):
    fr_wp_overlap = self.metrics._get_wordpiece_overlap_percent(
        self.info_dict['fr']['wordpieces'], self.info_dict['en']['wordpieces'])
    self.assertEqual(fr_wp_overlap, 20.0)

  def testWordpieceOverlapFrenchWeighted(self):
    fr_wp_overlap = self.metrics._get_wordpiece_overlap_percent(
        self.info_dict['fr']['wordpieces'], self.info_dict['en']['wordpieces'],
        weighted=True)
    self.assertEqual(fr_wp_overlap, 200/7)

  def testWordpieceOverlapEnglish(self):
    en_wp_overlap = self.metrics._get_wordpiece_overlap_percent(
        self.info_dict['en']['wordpieces'], self.info_dict['en']['wordpieces'])
    self.assertEqual(en_wp_overlap, 100.0)

  def testFormatFloatOrNone(self):
    extra_digits = 0.12345
    self.assertEqual(self.metrics._format_float_or_none(extra_digits), '0.123')
    fewer_digits = 0.1
    self.assertEqual(self.metrics._format_float_or_none(fewer_digits), '0.100')
    non_float = ''
    self.assertEqual(self.metrics._format_float_or_none(non_float), None)


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class CountPreprocessingFnTest(absltest.TestCase):

  def setUp(self):
    super(CountPreprocessingFnTest, self).setUp()
    self.raw_data = {
        'text': ['Let\'s make this Chinese even though it\'s English.'],
        'language_code': ['zh'],
    }

  def testUseGivenLang(self):
    preprocessing_fn = utils.count_preprocessing_fn('text', 'language_code')
    with tf.Session() as sess:
      expected_tokens = ['Let', '\'', 's', 'make', 'this', 'Chinese', 'even',
                         'though', 'it', '\'', 's', 'English', '.']

      outputs = preprocessing_fn(self.raw_data)
      outputs = sess.run(outputs)
      self.assertEqual(outputs['lang'], 'zh')
      self.assertSequenceAlmostEqual(outputs['tokens'].values, expected_tokens)


class MetricsPreprocessingFnTest(absltest.TestCase):

  def setUp(self):
    super(MetricsPreprocessingFnTest, self).setUp()
    self.raw_data = {
        'label': ['1'],
        'text_a': ['The boy jumped into the air.'],
        'lang': ['en'],
    }
    self.vocab = ['The', 'jump', '##ed', 'in', '##to', 'the', 'air', '.', 'bo',
                  'jumped', 'to', 'cat', 'sat', 'on', 'a', 'h', '##at', 'c']
    self.expected_wordpieces = ['The', '[UNK]', 'jumped', 'in', '##to', 'the',
                                'air', '.']

  def testSingleElement(self):
    with tf.Session() as sess:
      with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as vocab:
        vocab.writelines([word + '\n' for word in self.vocab])
        vocab.flush()
        preprocessing_fn = utils.metrics_preprocessing_fn(
            vocab.name, 'text_a', 'lang')
        outputs = preprocessing_fn(self.raw_data)
        tf.tables_initializer().run()
        outputs = sess.run(outputs)

        self.assertEqual(outputs['lang'], 'en')
        self.assertEqual(outputs['num_non_unk_wordpieces'], 7)
        self.assertEqual(outputs['num_preserved_chars'], 20)
        self.assertEqual(outputs['num_dropped_chars'], 3)
        self.assertSequenceAlmostEqual(outputs['wordpieces'].values,
                                       self.expected_wordpieces)

  def testLargerBatchSize(self):
    with tf.Session() as sess:
      with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as vocab:
        raw_data = {
            'label': ['1', '2'],
            'text_a': ['The boy jumped into the air.', 'The cat sat on a hat.'],
            'lang': ['en', 'en'],
        }
        expected_wordpieces = ['The', '[UNK]', 'jumped', 'in', '##to', 'the',
                               'air', '.', 'The', 'cat', 'sat', 'on', 'a', 'h',
                               '##at', '.']
        vocab.writelines([word + '\n' for word in self.vocab])
        vocab.flush()
        preprocessing_fn = utils.metrics_preprocessing_fn(
            vocab.name, 'text_a', 'lang')
        outputs = preprocessing_fn(raw_data)
        tf.tables_initializer().run()
        outputs = sess.run(outputs)

        self.assertSequenceAlmostEqual(outputs['lang'], ['en', 'en'])
        self.assertSequenceAlmostEqual(outputs['num_preserved_chars'], [20, 16])
        self.assertSequenceAlmostEqual(outputs['num_dropped_chars'], [3, 0])
        self.assertSequenceAlmostEqual(outputs['wordpieces'].values,
                                       expected_wordpieces)
        self.assertSequenceAlmostEqual(outputs['num_non_unk_wordpieces'],
                                       [7, 8])


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  absltest.main()
