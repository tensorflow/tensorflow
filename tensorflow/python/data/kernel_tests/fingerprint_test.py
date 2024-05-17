# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `dataset.fingerprint()`."""

from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


def dataset_fn_test_cases():
  cases = [
      ("range1", lambda: dataset_ops.Dataset.range(10)),
      (
          "flat_map1",
          lambda: dataset_ops.Dataset.range(10).flat_map(
              dataset_ops.Dataset.range
          ),
      ),
      ("tfrecord1", lambda: readers.TFRecordDataset(["f1.txt", "f2.txt"])),
      (
          "tfrecord2",
          lambda: readers.TFRecordDataset(["f1.txt", "f2.txt"]).repeat(2),
      ),
  ]

  named_cases = []
  for case in cases:
    name, dataset_fn = case
    named_cases.append(combinations.NamedObject(name=name, obj=dataset_fn))

  return combinations.combine(dataset_fn=named_cases)


def dataset_pair_fn_test_cases():
  dataset = dataset_ops.Dataset

  cases = [
      ("range1", lambda: (dataset.range(10), dataset.range(11))),
      (
          "flat_map1",
          lambda: (
              dataset.range(10).flat_map(dataset.range),
              dataset.range(10).flat_map(lambda x: dataset.range(x + 1)),
          ),
      ),
      (
          "tfrecord1",
          lambda: (
              readers.TFRecordDataset(["f1.txt", "f2.txt"]),
              readers.TFRecordDataset(["f1.txt", "f3.txt"]),
          ),
      ),
      (
          "tfrecord2",
          lambda: (
              readers.TFRecordDataset(["f1.txt", "f2.txt"]).repeat(2),
              readers.TFRecordDataset(["f1.txt", "f3.txt"]).repeat(2),
          ),
      ),
  ]

  named_cases = []
  for case in cases:
    name, dataset_pair_fn = case
    named_cases.append(combinations.NamedObject(name=name, obj=dataset_pair_fn))

  return combinations.combine(dataset_pair_fn=named_cases)


class FingerprintTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(), dataset_fn_test_cases()
      )
  )
  def testSameDatasetSameFingerprint(self, dataset_fn):
    fingerprint1 = self.evaluate(dataset_fn().fingerprint())
    fingerprint2 = self.evaluate(dataset_fn().fingerprint())
    self.assertEqual(fingerprint1, fingerprint2)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          dataset_pair_fn_test_cases(),
      )
  )
  def testDifferentDatasetDifferentFingerprint(self, dataset_pair_fn):
    lhs, rhs = dataset_pair_fn()
    lhs_fingerprint = self.evaluate(lhs.fingerprint())
    rhs_fingerprint = self.evaluate(rhs.fingerprint())
    self.assertNotEqual(lhs_fingerprint, rhs_fingerprint)


if __name__ == "__main__":
  test.main()
