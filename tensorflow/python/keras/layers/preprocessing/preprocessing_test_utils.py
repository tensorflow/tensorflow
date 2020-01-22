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

import collections
import numpy as np

from tensorflow.python.platform import test


class PreprocessingLayerTest(test.TestCase):
  """Base test class for preprocessing layer API validation."""

  # TODO(b/137303934): Consider incorporating something like this Close vs All
  # behavior into core tf.test.TestCase.
  def assertAllCloseOrEqual(self, a, b, msg=None):
    """Asserts that elements are close (if numeric) or equal (if string)."""
    if isinstance(a, collections.Mapping):
      self.assertEqual(len(a), len(b))
      for key, a_value in a.items():
        b_value = b[key]
        error_message = "{} ({})".format(msg, key) if msg else None
        self.assertAllCloseOrEqual(a_value, b_value, error_message)
    elif isinstance(a, (list, tuple)):
      self.assertEqual(len(a), len(b))
      for a_value, b_value in zip(a, b):
        self.assertAllCloseOrEqual(a_value, b_value, msg)
    else:
      comparison_fn = (
          self.assertAllClose
          if np.issubdtype(a.dtype, np.number) else self.assertAllEqual)
      comparison_fn(a, b, msg=msg)

  def assert_extracted_output_equal(self, combiner, acc1, acc2, msg=None):
    data_1 = combiner.extract(acc1)
    data_2 = combiner.extract(acc2)
    self.assertAllCloseOrEqual(data_1, data_2, msg=msg)

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
    self.assertAllCloseOrEqual(
        single_compute,
        all_merge,
        msg="Sharding data should not change the data output.")

    unordered_all_merge = combiner.merge([
        combiner.compute(data_1),
        combiner.compute(data_2),
        combiner.compute(data_0)
    ])
    self.assertAllCloseOrEqual(
        all_merge,
        unordered_all_merge,
        msg="The order of merge arguments should not change the data "
        "output.")

    hierarchical_merge = combiner.merge([
        combiner.compute(data_1),
        combiner.merge([combiner.compute(data_2),
                        combiner.compute(data_0)])
    ])
    self.assertAllCloseOrEqual(
        all_merge,
        hierarchical_merge,
        msg="Nesting merge arguments should not change the data output.")

    nested_compute = combiner.compute(
        data_0, combiner.compute(data_1, combiner.compute(data_2)))
    self.assertAllCloseOrEqual(
        all_merge,
        nested_compute,
        msg="Nesting compute arguments should not change the data output.")

    mixed_compute = combiner.merge([
        combiner.compute(data_0),
        combiner.compute(data_1, combiner.compute(data_2))
    ])
    self.assertAllCloseOrEqual(
        all_merge,
        mixed_compute,
        msg="Mixing merge and compute calls should not change the data "
        "output.")
    self.assertAllCloseOrEqual(expected, all_merge)

  def validate_accumulator_extract(self, combiner, data, expected):
    """Validate that the expected results of computing and extracting."""
    acc = combiner.compute(data)
    extracted_data = combiner.extract(acc)
    self.assertAllCloseOrEqual(expected, extracted_data)

  def validate_accumulator_extract_and_restore(self, combiner, data, expected):
    """Validate that the extract<->restore loop loses no data."""
    acc = combiner.compute(data)
    extracted_data = combiner.extract(acc)
    restored_acc = combiner.restore(extracted_data)
    self.assert_extracted_output_equal(combiner, acc, restored_acc)
    self.assertAllCloseOrEqual(expected, combiner.extract(restored_acc))

  def validate_accumulator_serialize_and_deserialize(self, combiner, data,
                                                     expected):
    """Validate that the serialize<->deserialize loop loses no data."""
    acc = combiner.compute(data)
    serialized_data = combiner.serialize(acc)
    deserialized_data = combiner.deserialize(serialized_data)
    self.assertAllCloseOrEqual(acc, deserialized_data)

  def validate_accumulator_uniqueness(self, combiner, data):
    """Validate that every call to compute creates a unique accumulator."""
    acc = combiner.compute(data)
    acc2 = combiner.compute(data)
    self.assertIsNot(acc, acc2)
    self.assertAllCloseOrEqual(acc, acc2)
