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

from tensorflow.python.data.experimental.ops import unique
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class UniqueTest(test_base.DatasetTestBase):

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
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      for test_case, expected in test_cases:
        current_test_case = test_case
        self.evaluate(iterator.initializer)
        for element in expected:
          if dtype == dtypes.string:
            element = compat.as_bytes(element)
          self.assertAllEqual(element, self.evaluate(next_element))
        with self.assertRaises(errors.OutOfRangeError):
          self.evaluate(next_element)

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def testSimpleString(self):
    self._testSimpleHelper(dtypes.string, [
        ([], []),
        (["hello"], ["hello"]),
        (["hello", "hello", "hello"], ["hello"]),
        (["hello", "world"], ["hello", "world"]),
        (["foo", "bar", "baz", "baz", "bar", "foo"], ["foo", "bar", "baz"]),
    ])


if __name__ == "__main__":
  test.main()
