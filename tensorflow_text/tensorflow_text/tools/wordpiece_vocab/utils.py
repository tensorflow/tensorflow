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

"""PTransforms used for wordpiece vocabulary generation pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import apache_beam as beam
import tensorflow.compat.v1 as tf
from tensorflow_text.python.ops.bert_tokenizer import BertTokenizer
from tensorflow_text.python.ops.wordpiece_tokenizer import WordpieceTokenizer
from tensorflow_text.tools.wordpiece_vocab import wordpiece_tokenizer_learner_lib as learner


class FilterTokensByLang(beam.DoFn):
  """Filters out languages if necessary and yields each (token, lang) pair."""

  def __init__(self, lang_set, include_other_languages=False):
    self._lang_set = lang_set
    self._include_other_languages = include_other_languages

  def process(self, element):
    lang = element['lang']

    if lang in self._lang_set or self._include_other_languages:
      returned_lang = lang if lang in self._lang_set else 'other'

      for token in element['tokens']:
        yield token, returned_lang


class CalculateCoefficients(beam.CombineFn):
  """Calculates smoothing coefficient for each language."""

  def __init__(self, smoothing_exponent):
    self._smoothing_exponent = smoothing_exponent

  def create_accumulator(self):
    return {'total_count': 0, 'lang_count': collections.Counter()}

  def add_input(self, accumulator, element):
    _, lang = element
    accumulator['total_count'] += 1
    accumulator['lang_count'].update([lang])
    return accumulator

  def merge_accumulators(self, accumulators):
    merged = self.create_accumulator()
    for acc in accumulators:
      for key in merged:
        merged[key] += acc[key]
    return merged

  def extract_output(self, accumulator):
    lang_count = accumulator['lang_count']
    total = accumulator['total_count']
    probs, exp = {}, {}
    for lang in lang_count:
      probs[lang] = lang_count[lang] / total
      exp[lang] = pow(probs[lang], self._smoothing_exponent)
    total_weight = sum(exp.values())
    for lang in exp:
      exp[lang] = exp[lang] / (total_weight * probs[lang])
    return exp


class ExponentialSmoothing(beam.DoFn):
  """Applies exponential smoothing coefficients to the counts."""

  def __init__(self, corpus_multiplier=1):
    self._corpus_multiplier = corpus_multiplier

  def process(self, word_and_lang, coeffs):
    word, lang = word_and_lang
    count = coeffs[lang] * self._corpus_multiplier
    yield word, count


class FilterByCount(beam.DoFn):
  """Filters words with counts below some threshold."""

  def __init__(self, max_word_length, min_token_frequency=2):
    self._min_token_frequency = int(min_token_frequency)
    self._max_word_length = max_word_length

  def process(self, word_and_count):
    word, count = word_and_count
    if count > self._min_token_frequency and len(word) <= self._max_word_length:
      yield word, int(round(count))


class SortByCount(beam.CombineFn):
  """Sorts words by count."""

  def create_accumulator(self):
    return []

  def add_input(self, accumulator, element):
    if not accumulator:
      accumulator = self.create_accumulator()

    word, count = element
    accumulator.append((word, int(count)))
    return accumulator

  def merge_accumulators(self, accumulators):
    merged = self.create_accumulator()
    for accumulator in accumulators:
      if accumulator:
        merged.extend(accumulator)
    return merged

  def extract_output(self, accumulator):
    return sorted(sorted(accumulator, key=lambda x: x[0]), key=lambda x: x[1],
                  reverse=True)


class CompileTokenizationInfo(beam.DoFn):
  """Expands list of tokens and computes intermediate metrics."""

  def process(self, record):
    wordpiece_counter = collections.Counter(record['wordpieces'])
    del wordpiece_counter['[UNK]']
    dropped = record['num_dropped_chars']
    preserved = record['num_preserved_chars']
    non_unk = record['num_non_unk_wordpieces']
    preserved_ratio = [preserved / non_unk] if non_unk else []
    dropped_ratio = [dropped / (dropped + preserved)] if (dropped +
                                                          preserved) else []
    tokenization_info = {
        'lang': record['lang'],
        'count': 1,
        'num_preserved_chars': preserved,
        'num_dropped_chars': dropped,
        'num_non_unk_wordpieces': non_unk,
        'preserved_ratio': preserved_ratio,
        'dropped_ratio': dropped_ratio,
        'wordpieces': wordpiece_counter
    }
    yield tokenization_info


def default():
  return {
      'count': 0,
      'num_preserved_chars': 0,
      'num_dropped_chars': 0,
      'num_non_unk_wordpieces': 0,
      'preserved_ratio': [],
      'dropped_ratio': [],
      'wordpieces': collections.Counter()
  }


class AggregateLang(beam.CombineFn):
  """Aggregates intermediate metrics for each language."""

  def create_accumulator(self):
    return collections.defaultdict(default)

  def add_input(self, accumulator, element):
    lang = element['lang']
    for key in accumulator[lang].keys():
      accumulator[lang][key] += element[key]
    return accumulator

  def merge_accumulators(self, accumulators):
    merged = self.create_accumulator()
    for acc in accumulators:
      for lang in acc.keys():
        for key in acc[lang].keys():
          merged[lang][key] += acc[lang][key]
    return merged

  def extract_output(self, accumulator):
    return accumulator


class LearnVocab(beam.DoFn):

  def __init__(self, params):
    self._params = params

  def process(self, wordcounts):
    return learner.learn(wordcounts, self._params)


class CalculateMetrics(beam.DoFn):
  """Calculates metrics for each language given tokenization info."""

  def process(self, info_dict):
    for lang in info_dict.keys():
      infos = info_dict[lang]
      yield {
          'lang':
              lang,
          'sample_count':
              infos['count'],
          'micro_drop_char_percent':
              self._format_float_or_none(
                  self._get_micro_dropped_char_percent(infos)),
          'macro_drop_char_percent':
              self._format_float_or_none(
                  self._get_macro_dropped_char_percent(infos)),
          'micro_compress_ratio':
              self._format_float_or_none(
                  self._get_micro_compression_ratio(infos)),
          'macro_compress_ratio':
              self._format_float_or_none(
                  self._get_macro_compression_ratio(infos)),
          'unweighted_en_wp_overlap_percent':
              self._format_float_or_none(
                  self._get_wordpiece_overlap_percent(
                      infos['wordpieces'],
                      info_dict['en']['wordpieces'],
                      weighted=False)),
          'weighted_en_wp_overlap_percent':
              self._format_float_or_none(
                  self._get_wordpiece_overlap_percent(
                      infos['wordpieces'],
                      info_dict['en']['wordpieces'],
                      weighted=True))
      }

  def _get_list_mean(self, l):
    return sum(l) / len(l) if l else None

  def _get_micro_compression_ratio(self, infos):
    if infos['num_non_unk_wordpieces']:
      return infos['num_preserved_chars'] / infos['num_non_unk_wordpieces']
    else:
      return None

  def _get_macro_compression_ratio(self, infos):
    return self._get_list_mean(infos['preserved_ratio'])

  def _get_micro_dropped_char_percent(self, infos):
    if infos['num_preserved_chars'] + infos['num_dropped_chars']:
      return 100.0 * infos['num_dropped_chars'] / (
          infos['num_preserved_chars'] + infos['num_dropped_chars'])
    else:
      return None

  def _get_macro_dropped_char_percent(self, infos):
    return 100.0 * self._get_list_mean(infos['dropped_ratio'])

  def _get_wordpiece_overlap_percent(self,
                                     xx_wordpiece_counter,
                                     en_wordpiece_counter,
                                     weighted=False):
    numerator = 0
    denominator = 0
    for wordpiece, count in xx_wordpiece_counter.iteritems():
      if not weighted:
        count = 1
      denominator += count
      if wordpiece in en_wordpiece_counter:
        numerator += count

    if denominator:
      return 100.0 * numerator / denominator
    else:
      return None

  def _format_float_or_none(self, value):
    if isinstance(value, float):
      return '{:.3f}'.format(value)
    else:
      return None


def count_preprocessing_fn(text_key, language_code_key):
  """Generates a preprocessing function to be used in generating word counts.

  Args:
    text_key: feature key in tf.Example for text
    language_code_key: feature key in tf.Example for language_code

  Returns:
    a preprocessing function
  """

  def preprocessing_fn(inputs):
    """Function used to transform dataset using TF transform.

       Tokenizes input and detects language if there is no associated
       language_code.

    Args:
       inputs: dataset of tf.Examples containing text samples

    Returns:
       transformed outputs
    """

    outputs = {}

    tokenizer = BertTokenizer()
    tokens = tokenizer.tokenize(inputs[text_key])
    outputs['tokens'] = tokens.to_sparse()
    outputs['lang'] = tf.convert_to_tensor(inputs[language_code_key])

    return outputs

  return preprocessing_fn


def metrics_preprocessing_fn(vocab_file, text_key, language_code_key):
  """Generates a preprocessing function to be used in generating word counts.

  Args:
    vocab_file: path to file containing wordpiece vocabulary
    text_key: feature key in tf.Example for text
    language_code_key: feature key in tf.Example for language_code

  Returns:
    a preprocessing function
  """

  def preprocessing_fn(inputs):
    """Preprocessing function used in TF Transform.

    Args:
       inputs: the input dataset of tf.Examples

    Returns:
       preprocessed outputs
    """
    vocab_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        vocab_file, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER), -1)

    tokenizer = BertTokenizer()
    tokens = tokenizer.tokenize(inputs[text_key])
    wordpiece_tokenizer = WordpieceTokenizer(vocab_table,
                                             token_out_type=tf.string)
    wordpieces = wordpiece_tokenizer.tokenize(tokens)
    wordpieces_flat = wordpieces.flat_values
    wordpieces_flat.set_shape([None])
    wordpieces = tf.RaggedTensor.from_nested_row_splits(
        wordpieces_flat, wordpieces.nested_row_splits)

    known_mask = tf.cast(tf.not_equal(wordpieces, '[UNK]'), tf.int32)
    num_non_unk_wordpieces = tf.reduce_sum(known_mask, axis=[1, 2])

    wordpiece_is_unknown = tf.equal(wordpieces, '[UNK]')
    token_has_unknown = tf.reduce_any(wordpiece_is_unknown, axis=-1)
    unknown_tokens = tf.ragged.boolean_mask(tokens, token_has_unknown)
    unknown_lengths = tf.strings.length(unknown_tokens)
    num_dropped_chars = tf.math.reduce_sum(unknown_lengths, axis=1)

    token_lengths = tf.strings.length(tokens)
    total_chars = tf.reduce_sum(token_lengths, axis=-1)
    num_preserved_chars = total_chars - num_dropped_chars

    flattened = tf.RaggedTensor.from_row_splits(
        wordpieces.flat_values, tf.gather(wordpieces.values.row_splits,
                                          wordpieces.row_splits))

    outputs = {}
    outputs['num_non_unk_wordpieces'] = tf.cast(num_non_unk_wordpieces,
                                                tf.int64)
    outputs['num_dropped_chars'] = tf.cast(num_dropped_chars, tf.int64)
    outputs['num_preserved_chars'] = tf.cast(num_preserved_chars, tf.int64)
    outputs['wordpieces'] = flattened.to_sparse()
    outputs['lang'] = tf.convert_to_tensor(inputs[language_code_key])

    return outputs

  return preprocessing_fn
