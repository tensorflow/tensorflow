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

  @combinations.generate(test_base.graph_only_combinations())
  def testTypeMismatch(self):

    # Placeholder values are needed to fill in the expected array with dummy value so that,
    # when the dataset generates the element and observes that there is a type mismatch,
    # it raises the proper error and not an OutOfRangeError which occurs when it is unable
    # to fetch an element to compare from the expected array in the first place.
    string_placeholder = ""
    int32_placeholder = 0
    int64_placeholder = 0

    # raises InternalError when element type doesn't match with dtypes.string.
    string_cases = [
        (["hello", 1, 2, 1], ["hello"]),
        (["hello", "world", 1], ["hello", "world"]),
        (["hello", "hello", "world", 1, 2], ["hello", "world"]),
        (["hello", "world", 1, 1, 2], ["hello", "world"]),
        # In the following cases, when the first element (i.e 1) of the dataset is generated,
        # it validates the type and immediately raises the error. This is unlike the above cases,
        # wherein the dtype of the starting elements are as expected to start with,
        # and the dataset has to loop until it reaches the incorrect dtype element.
        # Until then we need to make sure that data with correct type has to match
        # for testing purposes. Similar logic applies to dtype.int32 and dtype.64 as well.
        ([1, 2, "hello"], [string_placeholder]),
        ([1, 1, 2, 3, 3, "hello"], [string_placeholder]),
    ]

    # handle each case independently so that an error raised by a single case doesn't interfere
    # with the other ones. As per self._testSimpleHelper functionality.
    for case in string_cases:
      with self.assertRaises(errors.InternalError):
        self._testSimpleHelper(dtypes.string, [case])

    # raises InvalidArgumentError when element type doesn't match with dtypes.int32.
    int32_cases = [
        ([1, "hello", "world"], [1]),
        ([1, 2, 1, "hello", "hello", "world"], [1, 2]),
        (["hello", 1, 2], [int32_placeholder]),
        (["hello", 1, 1, 2, 3, 3], [int32_placeholder]),
    ]
    for case in int32_cases:
      with self.assertRaises(errors.InvalidArgumentError):
        self._testSimpleHelper(dtypes.int32, [case])

    # raises InvalidArgumentError when element type doesn't match with dtypes.int64.
    int64_cases = [
        ([2, 3, "hello", "world"], [2, 3]),
        ([2, 3, 3, "hello", "hello", "world"], [2, 3]),
        (["hello", 2, 2], [int64_placeholder]),
        (["hello", "hello", 1, 1, 2, 3], [int64_placeholder]),
    ]
    for case in int64_cases:
      with self.assertRaises(errors.InvalidArgumentError):
        self._testSimpleHelper(dtypes.int64, [case])


if __name__ == "__main__":
  test.main()
