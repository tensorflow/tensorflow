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
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
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
          ([1, 1, 1, 1, 0], [1, 0]),
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
  def testUnsupportedTypes(self):
    """Should raise TypeError when element type doesn't match with the

    dtypes.int64, dtypes.int32 or dtypes.string (supported types).
    """

    for dtype in [
        dtypes.bool, dtypes.double, dtypes.complex64, dtypes.float32,
        dtypes.float64, dtypes.qint16, dtypes.qint32
    ]:
      with self.assertRaises(TypeError):
        _ = dataset_ops.Dataset.from_generator(lambda: [],
                                               dtype).apply(unique.unique())


class UniqueCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                           parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testUnique(self):

    def build_dataset(num_elements, unique_elem_range):
      return dataset_ops.Dataset.range(num_elements).map(
          lambda x: x % unique_elem_range).apply(unique.unique())

    self.run_core_tests(lambda: build_dataset(200, 100), 100)


if __name__ == "__main__":
  test.main()
