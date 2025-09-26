# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_config
from tensorflow.python.platform import test


class RaggedFeatureTest(test.TestCase):

  def test_validate_and_recreate_partition_row_splits(self):
    """Tests _validate_and_recreate_partition with RowSplits."""
    partition_types = parsing_config.RaggedFeature._PARTITION_TYPES
    partition = parsing_config.RaggedFeature.RowSplits(key="test_key")
    recreated_partition = (
        parsing_config.RaggedFeature._validate_and_recreate_partition(
            partition, partition_types
        )
    )
    self.assertIsInstance(
        recreated_partition, parsing_config.RaggedFeature.RowSplits
    )
    self.assertEqual(recreated_partition.key, "test_key")

  def test_validate_and_recreate_partition_row_lengths(self):
    """Tests _validate_and_recreate_partition with RowLengths."""
    partition_types = parsing_config.RaggedFeature._PARTITION_TYPES
    partition = parsing_config.RaggedFeature.RowLengths(key="test_key2")
    recreated_partition = (
        parsing_config.RaggedFeature._validate_and_recreate_partition(
            partition, partition_types
        )
    )
    self.assertIsInstance(
        recreated_partition, parsing_config.RaggedFeature.RowLengths
    )
    self.assertEqual(recreated_partition.key, "test_key2")

  def test_validate_and_recreate_partition_uniform_row_length(self):
    """Tests _validate_and_recreate_partition with UniformRowLength."""
    partition_types = parsing_config.RaggedFeature._PARTITION_TYPES
    partition = parsing_config.RaggedFeature.UniformRowLength(length=5)
    recreated_partition = (
        parsing_config.RaggedFeature._validate_and_recreate_partition(
            partition, partition_types
        )
    )
    self.assertIsInstance(
        recreated_partition, parsing_config.RaggedFeature.UniformRowLength
    )
    self.assertEqual(recreated_partition.length, 5)

  def test_validate_and_recreate_partition_invalid(self):
    """Tests _validate_and_recreate_partition with an invalid partition type."""
    partition_types = parsing_config.RaggedFeature._PARTITION_TYPES

    class InvalidPartition:

      def __init__(self, key):
        self.key = key

    partition = InvalidPartition(key="invalid_key")
    with self.assertRaisesRegex(
        TypeError, "cannot be cast to any of the partition"
    ):
      parsing_config.RaggedFeature._validate_and_recreate_partition(
          partition, partition_types
      )

  def test_ragged_feature_new_with_valid_partitions(self):
    """Tests RaggedFeature constructor with valid partitions."""
    partitions = [
        parsing_config.RaggedFeature.RowSplits(key="s1"),
        parsing_config.RaggedFeature.UniformRowLength(length=3),
    ]
    feature = parsing_config.RaggedFeature(
        dtype=dtypes.int64, partitions=partitions
    )
    self.assertEqual(feature.dtype, dtypes.int64)
    self.assertLen(feature.partitions, 2)
    self.assertIsInstance(
        feature.partitions[0], parsing_config.RaggedFeature.RowSplits
    )
    self.assertEqual(feature.partitions[0].key, "s1")
    self.assertIsInstance(
        feature.partitions[1], parsing_config.RaggedFeature.UniformRowLength
    )
    self.assertEqual(feature.partitions[1].length, 3)

  def test_ragged_feature_new_with_invalid_partitions(self):
    """Tests RaggedFeature constructor with invalid partitions."""

    class InvalidPartition:

      def __init__(self, key):
        self.key = key

    partitions = [
        InvalidPartition(key="invalid_key"),
    ]
    with self.assertRaisesRegex(
        TypeError, "cannot be cast to any of the partition"
    ):
      parsing_config.RaggedFeature(dtype=dtypes.int64, partitions=partitions)


if __name__ == "__main__":
  test.main()
