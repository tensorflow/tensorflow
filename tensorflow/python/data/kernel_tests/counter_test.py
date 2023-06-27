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
"""Tests for `tf.data.Dataset.counter`."""
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class CounterTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(start=3, step=4, expected_output=[[3, 7, 11]]) +
          combinations.combine(start=0, step=-1, expected_output=[[0, -1, -2]]))
  )
  def testCounter(self, start, step, expected_output):
    dataset = dataset_ops.Dataset.counter(start, step)
    self.assertEqual(
        [], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    self.assertEqual(dtypes.int64, dataset_ops.get_legacy_output_types(dataset))
    get_next = self.getNext(dataset)
    for expected in expected_output:
      self.assertEqual(expected, self.evaluate(get_next()))


class CounterCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                            parameterized.TestCase):

  def _build_counter_dataset(self, start, step, num_outputs, options=None):
    counter_dataset = dataset_ops.Dataset.counter(start, step)
    range_dataset = dataset_ops.Dataset.range(num_outputs)
    dataset = dataset_ops.Dataset.zip((counter_dataset, range_dataset))
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    num_outputs = 10
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self, lambda: self._build_counter_dataset(
            start=2, step=10, num_outputs=num_outputs, options=options),
        num_outputs)


if __name__ == "__main__":
  test.main()
