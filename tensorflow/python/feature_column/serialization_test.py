# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for feature_column and DenseFeatures serialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.feature_column import dense_features
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.feature_column import serialization
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class FeatureColumnSerializationTest(test.TestCase):
  """Tests for serialization, deserialization helpers."""

  def test_serialize_non_feature_column(self):

    class NotAFeatureColumn(object):
      pass

    with self.assertRaisesRegexp(ValueError, 'is not a FeatureColumn'):
      serialization.serialize_feature_column(NotAFeatureColumn())

  def test_deserialize_invalid_config(self):
    with self.assertRaisesRegexp(ValueError, 'Improper config format: {}'):
      serialization.deserialize_feature_column({})

  def test_deserialize_config_missing_key(self):
    config_missing_key = {
        'config': {
            # Dtype is missing and should cause a failure.
            # 'dtype': 'int32',
            'default_value': None,
            'key': 'a',
            'normalizer_fn': None,
            'shape': (2,)
        },
        'class_name': 'NumericColumn'
    }

    with self.assertRaisesRegexp(
        ValueError, 'Invalid config:.*expected keys.*dtype'):
      serialization.deserialize_feature_column(config_missing_key)

  def test_deserialize_invalid_class(self):
    with self.assertRaisesRegexp(
        ValueError, 'Unknown feature_column_v2: NotExistingFeatureColumnClass'):
      serialization.deserialize_feature_column({
          'class_name': 'NotExistingFeatureColumnClass',
          'config': {}
      })

  def test_deserialization_deduping(self):
    price = fc.numeric_column('price')
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 1])

    configs = serialization.serialize_feature_columns([price, bucketized_price])

    deserialized_feature_columns = serialization.deserialize_feature_columns(
        configs)
    self.assertLen(deserialized_feature_columns, 2)
    new_price = deserialized_feature_columns[0]
    new_bucketized_price = deserialized_feature_columns[1]

    # Ensure these are not the original objects:
    self.assertIsNot(price, new_price)
    self.assertIsNot(bucketized_price, new_bucketized_price)
    # But they are equivalent:
    self.assertEqual(price, new_price)
    self.assertEqual(bucketized_price, new_bucketized_price)

    # Check that deduping worked:
    self.assertIs(new_bucketized_price.source_column, new_price)

  def deserialization_custom_objects(self):
    # Note that custom_objects is also tested extensively above per class, this
    # test ensures that the public wrappers also handle it correctly.
    def _custom_fn(input_tensor):
      return input_tensor + 42.

    price = fc.numeric_column('price', normalizer_fn=_custom_fn)

    configs = serialization.serialize_feature_columns([price])

    deserialized_feature_columns = serialization.deserialize_feature_columns(
        configs)

    self.assertLen(deserialized_feature_columns, 1)
    new_price = deserialized_feature_columns[0]

    # Ensure these are not the original objects:
    self.assertIsNot(price, new_price)
    # But they are equivalent:
    self.assertEqual(price, new_price)

    # Check that normalizer_fn points to the correct function.
    self.assertIs(new_price.normalizer_fn, _custom_fn)


@test_util.run_all_in_graph_and_eager_modes
class DenseFeaturesSerializationTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', None, None),
      ('trainable', True, 'trainable'),
      ('not_trainable', False, 'frozen'))
  def test_get_config(self, trainable, name):
    cols = [fc.numeric_column('a'),
            fc.embedding_column(fc.categorical_column_with_identity(
                key='b', num_buckets=3), dimension=2)]
    orig_layer = dense_features.DenseFeatures(
        cols, trainable=trainable, name=name)
    config = orig_layer.get_config()

    self.assertEqual(config['name'], orig_layer.name)
    self.assertEqual(config['trainable'], trainable)
    self.assertLen(config['feature_columns'], 2)
    self.assertEqual(
        config['feature_columns'][0]['class_name'], 'NumericColumn')
    self.assertEqual(config['feature_columns'][0]['config']['shape'], (1,))
    self.assertEqual(
        config['feature_columns'][1]['class_name'], 'EmbeddingColumn')

  @parameterized.named_parameters(
      ('default', None, None),
      ('trainable', True, 'trainable'),
      ('not_trainable', False, 'frozen'))
  def test_from_config(self, trainable, name):
    cols = [fc.numeric_column('a'),
            fc.embedding_column(fc.categorical_column_with_vocabulary_list(
                'b', vocabulary_list=['1', '2', '3']), dimension=2),
            fc.indicator_column(fc.categorical_column_with_hash_bucket(
                key='c', hash_bucket_size=3))]
    orig_layer = dense_features.DenseFeatures(
        cols, trainable=trainable, name=name)
    config = orig_layer.get_config()

    new_layer = dense_features.DenseFeatures.from_config(config)

    self.assertEqual(new_layer.name, orig_layer.name)
    self.assertEqual(new_layer.trainable, trainable)
    self.assertLen(new_layer._feature_columns, 3)
    self.assertEqual(new_layer._feature_columns[0].name, 'a')
    self.assertEqual(new_layer._feature_columns[1].initializer.mean, 0.0)
    self.assertEqual(new_layer._feature_columns[1].categorical_column.name, 'b')
    self.assertIsInstance(new_layer._feature_columns[2], fc.IndicatorColumn)

  def test_crossed_column(self):
    a = fc.categorical_column_with_vocabulary_list(
        'a', vocabulary_list=['1', '2', '3'])
    b = fc.categorical_column_with_vocabulary_list(
        'b', vocabulary_list=['1', '2', '3'])
    ab = fc.crossed_column([a, b], hash_bucket_size=2)
    cols = [fc.indicator_column(ab)]

    orig_layer = dense_features.DenseFeatures(cols)
    config = orig_layer.get_config()

    new_layer = dense_features.DenseFeatures.from_config(config)

    self.assertLen(new_layer._feature_columns, 1)
    self.assertEqual(new_layer._feature_columns[0].name, 'a_X_b_indicator')


@test_util.run_all_in_graph_and_eager_modes
class SequenceFeaturesSerializationTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('default', None, None),
                                  ('trainable', True, 'trainable'),
                                  ('not_trainable', False, 'frozen'))
  def test_get_config(self, trainable, name):
    cols = [sfc.sequence_numeric_column('a')]
    orig_layer = sfc.SequenceFeatures(cols, trainable=trainable, name=name)
    config = orig_layer.get_config()

    self.assertEqual(config['name'], orig_layer.name)
    self.assertEqual(config['trainable'], trainable)
    self.assertLen(config['feature_columns'], 1)
    self.assertEqual(config['feature_columns'][0]['class_name'],
                     'SequenceNumericColumn')
    self.assertEqual(config['feature_columns'][0]['config']['shape'], (1,))

  @parameterized.named_parameters(('default', None, None),
                                  ('trainable', True, 'trainable'),
                                  ('not_trainable', False, 'frozen'))
  def test_from_config(self, trainable, name):
    cols = [sfc.sequence_numeric_column('a')]
    orig_layer = sfc.SequenceFeatures(cols, trainable=trainable, name=name)
    config = orig_layer.get_config()

    new_layer = sfc.SequenceFeatures.from_config(config)

    self.assertEqual(new_layer.name, orig_layer.name)
    self.assertEqual(new_layer.trainable, trainable)
    self.assertLen(new_layer._feature_columns, 1)
    self.assertEqual(new_layer._feature_columns[0].name, 'a')


@test_util.run_all_in_graph_and_eager_modes
class LinearModelLayerSerializationTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', 1, 'sum', None, None),
      ('trainable', 6, 'mean', True, 'trainable'),
      ('not_trainable', 10, 'sum', False, 'frozen'))
  def test_get_config(self, units, sparse_combiner, trainable, name):
    cols = [fc.numeric_column('a'),
            fc.categorical_column_with_identity(key='b', num_buckets=3)]
    layer = fc._LinearModelLayer(
        cols, units=units, sparse_combiner=sparse_combiner,
        trainable=trainable, name=name)
    config = layer.get_config()

    self.assertEqual(config['name'], layer.name)
    self.assertEqual(config['trainable'], trainable)
    self.assertEqual(config['units'], units)
    self.assertEqual(config['sparse_combiner'], sparse_combiner)
    self.assertLen(config['feature_columns'], 2)
    self.assertEqual(
        config['feature_columns'][0]['class_name'], 'NumericColumn')
    self.assertEqual(
        config['feature_columns'][1]['class_name'], 'IdentityCategoricalColumn')

  @parameterized.named_parameters(
      ('default', 1, 'sum', None, None),
      ('trainable', 6, 'mean', True, 'trainable'),
      ('not_trainable', 10, 'sum', False, 'frozen'))
  def test_from_config(self, units, sparse_combiner, trainable, name):
    cols = [fc.numeric_column('a'),
            fc.categorical_column_with_vocabulary_list(
                'b', vocabulary_list=('1', '2', '3')),
            fc.categorical_column_with_hash_bucket(
                key='c', hash_bucket_size=3)]
    orig_layer = fc._LinearModelLayer(
        cols, units=units, sparse_combiner=sparse_combiner,
        trainable=trainable, name=name)
    config = orig_layer.get_config()

    new_layer = fc._LinearModelLayer.from_config(config)

    self.assertEqual(new_layer.name, orig_layer.name)
    self.assertEqual(new_layer._units, units)
    self.assertEqual(new_layer._sparse_combiner, sparse_combiner)
    self.assertEqual(new_layer.trainable, trainable)
    self.assertLen(new_layer._feature_columns, 3)
    self.assertEqual(new_layer._feature_columns[0].name, 'a')
    self.assertEqual(
        new_layer._feature_columns[1].vocabulary_list, ('1', '2', '3'))
    self.assertEqual(new_layer._feature_columns[2].num_buckets, 3)


if __name__ == '__main__':
  test.main()
