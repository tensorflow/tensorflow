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
"""Tests for Keras' base preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test


class PreprocessingLayerTest(test.TestCase):
  """Base test class for preprocessing layer API validation."""

  def assert_accumulator_equal(self, combiner, acc1, acc2, message=None):
    data_1 = combiner.extract(acc1)
    data_2 = combiner.extract(acc2)
    self.assertAllClose(data_1, data_2, msg=message)

  def validate_accumulator_computation(self, combiner, data, expected):
    """Validate that various combinations of compute and merge are identical."""
    if len(data) < 4:
      raise AssertionError("Data must have at least 4 elements.")
    data_0 = np.array([data[0]])
    data_1 = np.array([data[1]])
    data_2 = np.array(data[2:])

    single_compute = combiner.compute(data)

    all_merge = combiner.merge([
        combiner.compute(data_0),
        combiner.compute(data_1),
        combiner.compute(data_2)
    ])
    self.assert_accumulator_equal(
        combiner,
        single_compute,
        all_merge,
        message="Sharding data should not change the data output.")

    unordered_all_merge = combiner.merge([
        combiner.compute(data_1),
        combiner.compute(data_2),
        combiner.compute(data_0)
    ])
    self.assert_accumulator_equal(
        combiner,
        all_merge,
        unordered_all_merge,
        message="The order of merge arguments should not change the data "
        "output."
    )

    hierarchical_merge = combiner.merge([
        combiner.compute(data_1),
        combiner.merge([combiner.compute(data_2),
                        combiner.compute(data_0)])
    ])
    self.assert_accumulator_equal(
        combiner,
        all_merge,
        hierarchical_merge,
        message="Nesting merge arguments should not change the data output.")

    nested_compute = combiner.compute(
        data_0, combiner.compute(data_1, combiner.compute(data_2)))
    self.assert_accumulator_equal(
        combiner,
        all_merge,
        nested_compute,
        message="Nesting compute arguments should not change the data output.")

    mixed_compute = combiner.merge([
        combiner.compute(data_0),
        combiner.compute(data_1, combiner.compute(data_2))
    ])
    self.assert_accumulator_equal(
        combiner,
        all_merge,
        mixed_compute,
        message="Mixing merge and compute calls should not change the data "
        "output.")

    self.assertAllClose(expected, combiner.extract(all_merge))

  def validate_accumulator_extract_and_restore(self, combiner, data, expected):
    """Validate that the extract<->restore loop loses no data."""
    acc = combiner.compute(data)
    extracted_data = combiner.extract(acc)
    restored_acc = combiner.restore(extracted_data)
    self.assert_accumulator_equal(combiner, acc, restored_acc)
    self.assertAllClose(expected, combiner.extract(restored_acc))

  def validate_accumulator_serialize_and_deserialize(self, combiner, data,
                                                     expected):
    """Validate that the serialize<->deserialize loop loses no data."""
    acc = combiner.compute(data)
    extracted_data = combiner.serialize(acc)
    restored_acc = combiner.deserialize(extracted_data)
    self.assert_accumulator_equal(combiner, acc, restored_acc)
    self.assertAllClose(expected, combiner.extract(restored_acc))

  def validate_accumulator_uniqueness(self, combiner, data, expected):
    """Validate that every call to compute creates a unique accumulator."""
    acc = combiner.compute(data)
    acc2 = combiner.compute(data)
    self.assertIsNot(acc, acc2)
    self.assertAllClose(expected, combiner.extract(acc))
