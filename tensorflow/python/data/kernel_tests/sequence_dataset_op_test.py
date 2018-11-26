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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class SequenceDatasetTest(test_base.DatasetTestBase):

  def testRepeatTensorDataset(self):
    """Test a dataset that repeats its input multiple times."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    # This placeholder can be fed when dataset-definition subgraph
    # runs (i.e. `init_op` below) to configure the number of
    # repetitions used in a particular iterator.

    def do_test(count):
      dataset = dataset_ops.Dataset.from_tensors(components).repeat(count)
      self.assertEqual([c.shape for c in components],
                       [shape for shape in dataset.output_shapes])
      self.assertDatasetProduces(dataset, [components] * count)

    # Test a finite repetition.
    do_test(3)

    # test a different finite repetition.
    do_test(7)

    # Test an empty repetition.
    do_test(0)

    # Test an infinite repetition.
    # NOTE(mrry): There's not a good way to test that the sequence
    # actually is infinite.
    dataset = dataset_ops.Dataset.from_tensors(components).repeat(-1)
    self.assertEqual([c.shape for c in components],
                     [shape for shape in dataset.output_shapes])
    get_next = self.getNext(dataset)
    for _ in range(17):
      results = self.evaluate(get_next())
      for component, result_component in zip(components, results):
        self.assertAllEqual(component, result_component)

  def testTakeTensorDataset(self):
    components = (np.arange(10),)

    def do_test(count):
      dataset = dataset_ops.Dataset.from_tensor_slices(components).take(count)
      self.assertEqual([c.shape[1:] for c in components],
                       [shape for shape in dataset.output_shapes])
      num_output = min(count, 10) if count != -1 else 10
      self.assertDatasetProduces(
          dataset, [tuple(components[0][i:i + 1]) for i in range(num_output)])

    # Take fewer than input size
    do_test(4)

    # Take more than input size
    do_test(25)

    # Take all of input
    do_test(-1)

    # Take nothing
    do_test(0)

  def testSkipTensorDataset(self):
    components = (np.arange(10),)

    def do_test(count):
      dataset = dataset_ops.Dataset.from_tensor_slices(components).skip(count)
      self.assertEqual([c.shape[1:] for c in components],
                       [shape for shape in dataset.output_shapes])
      start_range = min(count, 10) if count != -1 else 10
      self.assertDatasetProduces(
          dataset,
          [tuple(components[0][i:i + 1]) for i in range(start_range, 10)])

    # Skip fewer than input size, we should skip
    # the first 4 elements and then read the rest.
    do_test(4)

    # Skip more than input size: get nothing.
    do_test(25)

    # Skip exactly input size.
    do_test(10)

    # Set -1 for 'count': skip the entire dataset.
    do_test(-1)

    # Skip nothing
    do_test(0)

  def testRepeatRepeatTensorDataset(self):
    """Test the composition of repeat datasets."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    inner_count, outer_count = 7, 14

    dataset = dataset_ops.Dataset.from_tensors(components).repeat(
        inner_count).repeat(outer_count)
    self.assertEqual([c.shape for c in components],
                     [shape for shape in dataset.output_shapes])
    self.assertDatasetProduces(dataset,
                               [components] * (inner_count * outer_count))

  def testRepeatEmptyDataset(self):
    """Test that repeating an empty dataset does not hang."""
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10).skip(10).repeat(-1)
    self.assertDatasetProduces(dataset, [])


if __name__ == "__main__":
  test.main()
