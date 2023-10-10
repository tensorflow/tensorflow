# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.pad_to_cardinality()."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import pad_to_cardinality
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


pad_to_cardinality = pad_to_cardinality.pad_to_cardinality


class PadToCardinalityTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    data = [1, 2, 3, 4, 5]
    target = 12
    ds = dataset_ops.Dataset.from_tensor_slices({'a': data})
    ds = ds.apply(pad_to_cardinality(target))
    expected_data = [{'a': data[i], 'valid': True} for i in range(len(data))]
    expected_padding = [
        {'a': 0, 'valid': False} for _ in range(target - len(data))
    ]
    self.assertAllEqual(
        self.getDatasetOutput(ds), expected_data + expected_padding
    )

  @combinations.generate(test_base.default_test_combinations())
  def testNoPadding(self):
    data = [1, 2, 3, 4, 5]
    target = 5
    ds = dataset_ops.Dataset.from_tensor_slices({'a': data})
    ds = ds.apply(pad_to_cardinality(target))
    expected_data = [{'a': data[i], 'valid': True} for i in range(len(data))]
    self.assertAllEqual(self.getDatasetOutput(ds), expected_data)

  @combinations.generate(test_base.default_test_combinations())
  def testStructuredData(self):
    data = {
        'a': [1, 2, 3, 4, 5],
        'b': ([b'a', b'b', b'c', b'd', b'e'], [-1, -2, -3, -4, -5]),
    }
    data_len = len(data['a'])
    target = 12
    ds = dataset_ops.Dataset.from_tensor_slices(data)
    ds = ds.apply(pad_to_cardinality(target))
    expected_data = [
        {
            'a': data['a'][i],
            'b': (data['b'][0][i], data['b'][1][i]),
            'valid': True,
        }
        for i in range(data_len)
    ]
    expected_padding = [
        {'a': 0, 'b': (b'', 0), 'valid': False}
        for _ in range(target - data_len)
    ]
    self.assertAllEqual(
        self.getDatasetOutput(ds), expected_data + expected_padding
    )

  @combinations.generate(test_base.v2_eager_only_combinations())
  def testUnknownCardinality(self):
    ds = dataset_ops.Dataset.from_tensors({'a': 1}).filter(lambda x: True)
    with self.assertRaisesRegex(
        ValueError,
        'The dataset passed into `pad_to_cardinality` must have a '
        'known cardinalty, but has cardinality -2',
    ):
      ds = ds.apply(pad_to_cardinality(5))
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.v2_eager_only_combinations())
  def testInfiniteCardinality(self):
    ds = dataset_ops.Dataset.from_tensors({'a': 1}).repeat()
    with self.assertRaisesRegex(
        ValueError,
        'The dataset passed into `pad_to_cardinality` must have a '
        'known cardinalty, but has cardinality -1',
    ):
      ds = ds.apply(pad_to_cardinality(5))
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.v2_only_combinations())
  def testNonMapping(self):
    ds = dataset_ops.Dataset.from_tensors(1)
    with self.assertRaisesRegex(
        ValueError,
        '`pad_to_cardinality` requires its input dataset to be a dictionary',
    ):
      ds = ds.apply(pad_to_cardinality(5))
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.v2_eager_only_combinations())
  def testRequestedCardinalityTooShort(self):
    ds = dataset_ops.Dataset.from_tensors({'a': 1}).repeat(5)
    with self.assertRaisesRegex(
        ValueError,
        r'The dataset passed into `pad_to_cardinality` must have a cardinalty '
        r'less than the target cardinality \(3\), but has cardinality 5',
    ):
      ds = ds.apply(pad_to_cardinality(3))
      self.getDatasetOutput(ds)


if __name__ == '__main__':
  test.main()
