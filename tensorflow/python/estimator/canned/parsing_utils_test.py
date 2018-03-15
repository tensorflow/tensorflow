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
"""Tests for parsing_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator.canned import parsing_utils
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class ClassifierParseExampleSpec(test.TestCase):
  """Tests tf.estimator.classifier_parse_example_spec."""

  def test_defaults(self):
    parsing_spec = parsing_utils.classifier_parse_example_spec(
        feature_columns=[fc.numeric_column('a')], label_key='b')
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((1,), dtype=dtypes.int64),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_string(self):
    parsing_spec = parsing_utils.classifier_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        label_dtype=dtypes.string)
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((1,), dtype=dtypes.string),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  # TODO(ispir): test label_default_value compatibility with label_dtype
  def test_label_default_value(self):
    parsing_spec = parsing_utils.classifier_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        label_default=0)
    expected_spec = {
        'a':
            parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b':
            parsing_ops.FixedLenFeature(
                (1,), dtype=dtypes.int64, default_value=0),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_weight_column_as_string(self):
    parsing_spec = parsing_utils.classifier_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        weight_column='c')
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((1,), dtype=dtypes.int64),
        'c': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_weight_column_as_numeric_column(self):
    parsing_spec = parsing_utils.classifier_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        weight_column=fc.numeric_column('c'))
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((1,), dtype=dtypes.int64),
        'c': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_label_key_should_not_be_used_as_feature(self):
    with self.assertRaisesRegexp(ValueError,
                                 'label should not be used as feature'):
      parsing_utils.classifier_parse_example_spec(
          feature_columns=[fc.numeric_column('a')], label_key='a')

  def test_weight_column_should_not_be_used_as_feature(self):
    with self.assertRaisesRegexp(ValueError,
                                 'weight_column should not be used as feature'):
      parsing_utils.classifier_parse_example_spec(
          feature_columns=[fc.numeric_column('a')],
          label_key='b',
          weight_column=fc.numeric_column('a'))

  def test_weight_column_should_be_a_numeric_column(self):
    with self.assertRaisesRegexp(ValueError,
                                 'tf.feature_column.numeric_column'):
      not_a_numeric_column = 3
      parsing_utils.classifier_parse_example_spec(
          feature_columns=[fc.numeric_column('a')],
          label_key='b',
          weight_column=not_a_numeric_column)


class RegressorParseExampleSpec(test.TestCase):
  """Tests tf.estimator.classifier_parse_example_spec."""

  def test_defaults(self):
    parsing_spec = parsing_utils.regressor_parse_example_spec(
        feature_columns=[fc.numeric_column('a')], label_key='b')
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_int64(self):
    parsing_spec = parsing_utils.regressor_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        label_dtype=dtypes.int64)
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((1,), dtype=dtypes.int64),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_label_default_value(self):
    parsing_spec = parsing_utils.regressor_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        label_default=0.)
    expected_spec = {
        'a':
            parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b':
            parsing_ops.FixedLenFeature(
                (1,), dtype=dtypes.float32, default_value=0.),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_label_dimension(self):
    parsing_spec = parsing_utils.regressor_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        label_dimension=3)
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((3,), dtype=dtypes.float32),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_weight_column_as_string(self):
    parsing_spec = parsing_utils.regressor_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        weight_column='c')
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'c': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_weight_column_as_numeric_column(self):
    parsing_spec = parsing_utils.regressor_parse_example_spec(
        feature_columns=[fc.numeric_column('a')],
        label_key='b',
        weight_column=fc.numeric_column('c'))
    expected_spec = {
        'a': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'b': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
        'c': parsing_ops.FixedLenFeature((1,), dtype=dtypes.float32),
    }
    self.assertDictEqual(expected_spec, parsing_spec)

  def test_label_key_should_not_be_used_as_feature(self):
    with self.assertRaisesRegexp(ValueError,
                                 'label should not be used as feature'):
      parsing_utils.regressor_parse_example_spec(
          feature_columns=[fc.numeric_column('a')], label_key='a')

  def test_weight_column_should_not_be_used_as_feature(self):
    with self.assertRaisesRegexp(ValueError,
                                 'weight_column should not be used as feature'):
      parsing_utils.regressor_parse_example_spec(
          feature_columns=[fc.numeric_column('a')],
          label_key='b',
          weight_column=fc.numeric_column('a'))

  def test_weight_column_should_be_a_numeric_column(self):
    with self.assertRaisesRegexp(ValueError,
                                 'tf.feature_column.numeric_column'):
      not_a_numeric_column = 3
      parsing_utils.regressor_parse_example_spec(
          feature_columns=[fc.numeric_column('a')],
          label_key='b',
          weight_column=not_a_numeric_column)


if __name__ == '__main__':
  test.main()
