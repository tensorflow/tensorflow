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
"""Tests for `tf.data.experimental.unique()`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import unique
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class UniqueTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _testSimpleHelper(self, dtype, test_cases):
    """Test the `unique()` transformation on a list of test cases.

    Args:
      dtype: The `dtype` of the elements in each test case.
      test_cases: A list of pairs of lists. The first component is the test
        input that will be passed to the transformation; the second component
        is the expected sequence of outputs from the transformation.
    """

    # The `current_test_case` will be updated when we loop over `test_cases`
    # below; declare it here so that the generator can capture it once.
    current_test_case = []
    dataset = dataset_ops.Dataset.from_generator(lambda: current_test_case,
                                                 dtype).apply(unique.unique())

    for test_case, expected in test_cases:
      current_test_case = test_case
      self.assertDatasetProduces(dataset, [
          compat.as_bytes(element) if dtype == dtypes.string else element
          for element in expected
      ])

  @combinations.generate(test_base.graph_only_combinations())
  def testSimpleInt(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      self._testSimpleHelper(dtype, [
          ([], []),
          ([1], [1]),
          ([1, 1, 1, 1, 1, 1, 1], [1]),
          ([1, 2, 3, 4], [1, 2, 3, 4]),
          ([1, 2, 4, 3, 2, 1, 2, 3, 4], [1, 2, 4, 3]),
          ([[1], [1, 1], [1, 1, 1]], [[1], [1, 1], [1, 1, 1]]),
          ([[1, 1], [1, 1], [2, 2], [3, 3], [1, 1]], [[1, 1], [2, 2], [3, 3]]),
      ])

  @combinations.generate(test_base.graph_only_combinations())
  def testSimpleString(self):
    self._testSimpleHelper(dtypes.string, [
        ([], []),
        (["hello"], ["hello"]),
        (["hello", "hello", "hello"], ["hello"]),
        (["hello", "world"], ["hello", "world"]),
        (["foo", "bar", "baz", "baz", "bar", "foo"], ["foo", "bar", "baz"]),
    ])

  def _checkDatasetRaises(self, dtype, test_cases, error):
    """Test whether the dataset raises the appropriate errors
    while generating the outputs.

    Args:
      dtype: The expected `dtype` of the elements in each test case.
      test_cases: A list of lists. The dataset will be created from the
        list items.
      error: The expected error to be raised.
    """

    current_test_case = []
    dataset = dataset_ops.Dataset.from_generator(lambda: current_test_case,
                                                 dtype).apply(unique.unique())

    for test_case in test_cases:
      current_test_case = test_case
      with self.assertRaises(error):
        self.getDatasetOutput(dataset)

  @combinations.generate(test_base.graph_only_combinations())
  def testStringTypeMismatch(self):
    """Should raise InternalError when element type doesn't match
    with dtypes.string."""

    test_cases = [
        ["hello", 1],
        ["hello", "hello", "world", 3],
        ["hello", 1, 1],
        ["hello", "world", 1, 2],
        [1, "hello"],
        [1, 2, "hello"],
        [1, 3, "hello", "world"],
        [1, 1, "hello", "hello"]
    ]
    self._checkDatasetRaises(dtype=dtypes.string, test_cases=test_cases,
                             error=errors.InternalError)

  @combinations.generate(combinations.times(
      test_base.graph_only_combinations(),
      combinations.combine(dtype=[dtypes.int32, dtypes.int64])))
  def testIntTypeMismatch(self, dtype):
    """Should raise InvalidArgumentError when element type doesn't
    match with dtypes.int32, dtypes.int64"""

    test_cases = [
        [1, "foo"],
        [1, 2, "bar"],
        [1, 3, "foo", "bar"],
        [1, 4, "foo", "foo"],
        ["bar", 1],
        ["bar", "foo", 2],
        ["bar", "bar", "foo", 3],
        ["foo", 1, 1],
        ["bar", "bar", 1, 1],
    ]
    self._checkDatasetRaises(dtype=dtype, test_cases=test_cases,
                             error=errors.InvalidArgumentError)

  @combinations.generate(test_base.graph_only_combinations())
  def testUnsupportedTypes(self):
    """Should raise TypeError when element type doesn't match with the
    dtypes.int64, dtypes.int32 or dtypes.string (supported types)."""

    for dtype in [dtypes.bool, dtypes.double, dtypes.complex64,
                  dtypes.float32, dtypes.float64, dtypes.qint16, dtypes.qint32]:
      with self.assertRaises(TypeError):
        _ = dataset_ops.Dataset.from_generator(lambda: [],
                                               dtype).apply(unique.unique())


if __name__ == "__main__":
  test.main()
