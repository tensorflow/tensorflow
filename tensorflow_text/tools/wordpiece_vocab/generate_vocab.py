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

"""Generate wordpiece vocab and compute metrics over dataset of tf.Examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
from absl import app
from absl import flags
import apache_beam as beam
import tensorflow.compat.v1 as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_text.tools.wordpiece_vocab import utils
from tensorflow_text.tools.wordpiece_vocab import wordpiece_tokenizer_learner_lib as learner


FLAGS = flags.FLAGS

flags.DEFINE_string('data_file', None, 'The input data file path.')
flags.DEFINE_string('vocab_file', None, 'The output vocab file path.')
flags.DEFINE_string('metrics_file', None, 'The output metrics file path.')
flags.DEFINE_string(
    'lang_set', 'en,es,ru,ar,de,fr,it,pt,ja,pl,fa,zh',
    'Set of languages used to build wordpiece model, '
    'given as a comma-separated list.')
flags.DEFINE_string('text_key', 'text', 'Text feature key in input examples.')
flags.DEFINE_string(
    'language_code_key', 'language_code', 'Language code feature key.')
flags.DEFINE_float(
    'smoothing_exponent', 0.5,
    'Exponent used in calculating exponential smoothing coefficients.')
flags.DEFINE_integer('max_word_length', 50,
                     'Discard words of length greater than max_word_length.')
flags.DEFINE_integer('upper_thresh', 10000000,
                     'Upper threshold for binary search.')
flags.DEFINE_integer('lower_thresh', 10, 'Lower threshold for binary search.')
flags.DEFINE_integer('num_iterations', 4,
                     'Number of iterations in wordpiece learning algorithm.')
flags.DEFINE_integer('num_pad_tokens', 100, 'Number of padding tokens to '
                     'include in vocab.')
flags.DEFINE_integer('max_input_tokens', 5000000,
                     'Maximum number of input tokens, where -1 means no max.')
flags.DEFINE_integer('max_token_length', 50, 'Maximum length of a token.')
flags.DEFINE_integer('max_unique_chars', 1000,
                     'Maximum number of unique characters as tokens.')
flags.DEFINE_integer('vocab_size', 110000, 'Target size of generated vocab, '
                     'where vocab_size is an upper bound and the size of vocab '
                     'can be within slack_ratio less than the vocab_size.')
flags.DEFINE_float('slack_ratio', 0.05,
                   'Difference permitted between target and actual vocab size.')
flags.DEFINE_bool('include_joiner_token', True,
                  'Whether to include joiner token in word suffixes.')
flags.DEFINE_string('joiner', '##', 'Joiner token in word suffixes.')
flags.DEFINE_list('reserved_tokens',
                  ['<unk>', '<s>', '</s>', '<mask>',
                   '<cls>', '<sep>', '<S>', '<T>'],
                  'Reserved tokens to be included in vocab.')


def generate_vocab(data_file, vocab_file, metrics_file, raw_metadata, params,
                   min_token_frequency=2):
  """Returns a pipeline generating a vocab and writing the output.

  Args:
    data_file: recordio file to read
    vocab_file: path in which to write the vocab
    metrics_file: path in which to write the metrics
    raw_metadata: schema for dataset
    params: parameters for wordpiece vocab learning algorithm
    min_token_frequency: the min frequency for a token to be included
  """

  lang_set = set(FLAGS.lang_set.split(','))

  # Schema to format metrics as CSV.
  csv_schema = schema_utils.schema_from_feature_spec({
      'lang': tf.FixedLenFeature([], tf.string),
      'sample_count': tf.FixedLenFeature([], tf.int64),
      'micro_drop_char_percent': tf.FixedLenFeature([], tf.string),
      'macro_drop_char_percent': tf.FixedLenFeature([], tf.string),
      'micro_compress_ratio': tf.FixedLenFeature([], tf.string),
      'macro_compress_ratio': tf.FixedLenFeature([], tf.string),
      'unweighted_en_wp_overlap_percent': tf.FixedLenFeature([], tf.string),
      'weighted_en_wp_overlap_percent': tf.FixedLenFeature([], tf.string),
  })

  columns = ['lang',
             'sample_count',
             'micro_drop_char_percent',
             'macro_drop_char_percent',
             'micro_compress_ratio',
             'macro_compress_ratio',
             'unweighted_en_wp_overlap_percent',
             'weighted_en_wp_overlap_percent']

  example_converter = tft.coders.ExampleProtoCoder(raw_metadata.schema,
                                                   serialized=False)

  def run_vocab():
    """Creates a pipeline to generate wordpiece vocab over a corpus."""

    vocab_pipeline = beam.Pipeline()

    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      # Read raw data and convert to TF Transform encoded dict.
      raw_data = (
          vocab_pipeline
          | 'ReadInputData' >> beam.io.tfrecordio.ReadFromTFRecord(
              data_file, coder=beam.coders.ProtoCoder(tf.train.Example))
          | 'DecodeInputData' >> beam.Map(example_converter.decode))

      # Apply TF Transform.
      (transformed_data, _), _ = (
          (raw_data, raw_metadata)
          | 'FilterLangAndExtractToken' >> tft_beam.AnalyzeAndTransformDataset(
              utils.count_preprocessing_fn(FLAGS.text_key,
                                           FLAGS.language_code_key)))

      # Filter by languages.
      tokens = (
          transformed_data
          | 'FilterByLang' >> beam.ParDo(utils.FilterTokensByLang(lang_set)))

      # Calculate smoothing coefficients.
      coeffs = (
          tokens
          | 'CalculateSmoothingCoefficients' >> beam.CombineGlobally(
              utils.CalculateCoefficients(FLAGS.smoothing_exponent)))

      # Apply smoothing, aggregate counts, and sort words by count.
      _ = (
          tokens
          | 'ApplyExponentialSmoothing' >> beam.ParDo(
              utils.ExponentialSmoothing(), beam.pvalue.AsSingleton(coeffs))
          | 'SumCounts' >> beam.CombinePerKey(sum)
          | 'FilterLowCounts' >> beam.ParDo(utils.FilterByCount(
              FLAGS.max_word_length, min_token_frequency))
          | 'MergeAndSortCounts' >> beam.CombineGlobally(utils.SortByCount())
          | 'LearnVocab' >> beam.ParDo(utils.LearnVocab(params))
          | 'Flatten' >> beam.FlatMap(lambda x: x + '\n')
          | 'WriteVocab' >> beam.io.WriteToText(vocab_file,
                                                shard_name_template='',
                                                append_trailing_newlines=False))
    return vocab_pipeline

  def run_metrics():
    """Creates a pipeline to measure wordpiece vocab metrics over a corpus."""

    metrics_pipeline = beam.Pipeline()

    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      # Read raw data and convert to TF Transform encoded dict.
      raw_data = (
          metrics_pipeline
          | 'ReadInputData' >> beam.io.tfrecordio.ReadFromTFRecord(
              data_file, coder=beam.coders.ProtoCoder(tf.train.Example))
          | 'DecodeInputData' >> beam.Map(example_converter.decode))

      # Apply transform to wordpiece-tokenize input.
      (metrics_transformed_data, _), _ = (
          (raw_data, raw_metadata)
          | 'WordpieceTokenizeInput' >> tft_beam.AnalyzeAndTransformDataset(
              utils.metrics_preprocessing_fn(FLAGS.vocab_file,
                                             FLAGS.text_key,
                                             FLAGS.language_code_key)))

      # Initialize CSV coder. Aggregate values for each lang, calculate metrics,
      # and write to output to a CSV file.
      csv_converter = tft.coders.CsvCoder(columns, csv_schema)
      _ = (
          metrics_transformed_data
          | 'CompileTokenInfo' >> beam.ParDo(utils.CompileTokenizationInfo())
          | 'CombineStatsForLang' >> beam.CombineGlobally(utils.AggregateLang())
          | 'CalculateMetrics' >> beam.ParDo(utils.CalculateMetrics())
          | 'EncodeMetrics' >> beam.Map(csv_converter.encode)
          | 'WriteMetrics' >> beam.io.WriteToText(
              metrics_file, shard_name_template='', header=','.join(columns)))
    return metrics_pipeline

  vocab_pipeline = run_vocab()
  vocab_pipeline.run().wait_until_finish()

  metrics_pipeline = run_metrics()
  metrics_pipeline.run().wait_until_finish()


def main(_):
  # Define schema.
  raw_metadata = dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec({
          'text': tf.FixedLenFeature([], tf.string),
          'language_code': tf.FixedLenFeature([], tf.string),
      }))

  # Add in padding tokens.
  reserved_tokens = FLAGS.reserved_tokens
  if FLAGS.num_pad_tokens:
    padded_tokens = ['<pad>']
    padded_tokens += ['<pad%d>' % i for i in range(1, FLAGS.num_pad_tokens)]
    reserved_tokens = padded_tokens + reserved_tokens

  params = learner.Params(FLAGS.upper_thresh, FLAGS.lower_thresh,
                          FLAGS.num_iterations, FLAGS.max_input_tokens,
                          FLAGS.max_token_length, FLAGS.max_unique_chars,
                          FLAGS.vocab_size, FLAGS.slack_ratio,
                          FLAGS.include_joiner_token, FLAGS.joiner,
                          reserved_tokens)

  generate_vocab(FLAGS.data_file, FLAGS.vocab_file, FLAGS.metrics_file,
                 raw_metadata, params)


if __name__ == '__main__':
  app.run(main)
