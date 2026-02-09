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

"""Read text from RecordIO of tf.Examples and generate sorted word counts."""

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

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None, 'The input file path.')
flags.DEFINE_string('output_path', None, 'The output file path.')
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


def word_count(input_path, output_path, raw_metadata, min_token_frequency=2):
  """Returns a pipeline counting words and writing the output.

  Args:
    input_path: recordio file to read
    output_path: path in which to write the output
    raw_metadata: metadata of input tf.Examples
    min_token_frequency: the min frequency for a token to be included
  """

  lang_set = set(FLAGS.lang_set.split(','))

  # Create pipeline.
  pipeline = beam.Pipeline()

  with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    converter = tft.coders.ExampleProtoCoder(
        raw_metadata.schema, serialized=False)

    # Read raw data and convert to TF Transform encoded dict.
    raw_data = (
        pipeline
        | 'ReadInputData' >> beam.io.tfrecordio.ReadFromTFRecord(
            input_path, coder=beam.coders.ProtoCoder(tf.train.Example))
        | 'DecodeInputData' >> beam.Map(converter.decode))

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
        | 'Flatten' >> beam.FlatMap(lambda x: x)
        | 'FormatCounts' >> beam.Map(lambda tc: '%s\t%s' % (tc[0], tc[1]))
        | 'WriteSortedCount' >> beam.io.WriteToText(
            output_path, shard_name_template=''))

  return pipeline


def main(_):
  # Generate schema of input data.
  raw_metadata = dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec({
          'text': tf.FixedLenFeature([], tf.string),
          'language_code': tf.FixedLenFeature([], tf.string),
      }))

  pipeline = word_count(FLAGS.input_path, FLAGS.output_path, raw_metadata)
  pipeline.run().wait_until_finish()


if __name__ == '__main__':
  app.run(main)
