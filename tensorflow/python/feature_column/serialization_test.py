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


from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import serialization
from tensorflow.python.platform import test


class FeatureColumnSerializationTest(test.TestCase):
  """Tests for serialization, deserialization helpers."""

  def test_serialize_non_feature_column(self):

    class NotAFeatureColumn(object):
      pass

    with self.assertRaisesRegex(ValueError, 'is not a FeatureColumn'):
      serialization.serialize_feature_column(NotAFeatureColumn())

  def test_deserialize_invalid_config(self):
    with self.assertRaisesRegex(ValueError, 'Improper config format: {}'):
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

    with self.assertRaisesRegex(ValueError,
                                'Invalid config:.*expected keys.*dtype'):
      serialization.deserialize_feature_column(config_missing_key)

  def test_deserialize_invalid_class(self):
    with self.assertRaisesRegex(
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


if __name__ == '__main__':
  test.main()
