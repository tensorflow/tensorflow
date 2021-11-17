# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.assert_cardinality()`."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class AssertCardinalityTest(test_base.DatasetTestBase, parameterized.TestCase):
  """Tests for `tf.data.experimental.assert_cardinality()`."""

  @combinations.generate(test_base.default_test_combinations())
  def testCorrectCardinality(self):
    dataset = dataset_ops.Dataset.range(10).filter(lambda x: True)
    self.assertEqual(
        self.evaluate(cardinality.cardinality(dataset)), cardinality.UNKNOWN)
    self.assertDatasetProduces(dataset, expected_output=range(10))
    dataset = dataset.apply(cardinality.assert_cardinality(10))
    self.assertEqual(self.evaluate(cardinality.cardinality(dataset)), 10)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_elements=10,
              asserted_cardinality=20,
              expected_error="Input dataset was expected to contain 20 "
              "elements but contained only 10 elements.") +
          combinations.combine(
              num_elements=1,
              asserted_cardinality=20,
              expected_error="Input dataset was expected to contain 20 "
              "elements but contained only 1 element.") +
          combinations.combine(
              num_elements=10,
              asserted_cardinality=cardinality.INFINITE,
              expected_error="Input dataset was expected to contain an "
              "infinite number of elements but contained only 10 elements.") +
          combinations.combine(
              num_elements=1,
              asserted_cardinality=cardinality.INFINITE,
              expected_error="Input dataset was expected to contain an "
              "infinite number of elements but contained only 1 element.") +
          combinations.combine(
              num_elements=10,
              asserted_cardinality=5,
              expected_error="Input dataset was expected to contain 5 "
              "elements but contained at least 6 elements.") +
          combinations.combine(
              num_elements=10,
              asserted_cardinality=1,
              expected_error="Input dataset was expected to contain 1 "
              "element but contained at least 2 elements.")))
  def testIncorrectCardinality(self, num_elements, asserted_cardinality,
                               expected_error):
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(
        cardinality.assert_cardinality(asserted_cardinality))
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(errors.FailedPreconditionError, expected_error):
      while True:
        self.evaluate(get_next())


class AssertCardinalityCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                      parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):

    def build_dataset(num_elements):
      return dataset_ops.Dataset.range(num_elements).apply(
          cardinality.assert_cardinality(num_elements))

    verify_fn(self, lambda: build_dataset(200), num_outputs=200)


if __name__ == "__main__":
  test.main()
