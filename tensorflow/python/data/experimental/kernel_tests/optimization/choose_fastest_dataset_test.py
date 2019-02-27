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
"""Tests for `tf.data.experimental._ChooseFastestDataset`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ChooseFastestDatasetTest(test_base.DatasetTestBase,
                               parameterized.TestCase):

  def testChooseFastestSimple(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3, 4])
    merge = optimization._ChooseFastestDataset([dataset, dataset])
    self.assertDatasetProduces(
        merge,
        expected_output=[0, 1, 2, 3, 4],
        expected_shapes=dataset_ops.get_legacy_output_shapes(dataset))

  def testChooseFastestManyInputs(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3, 4])
    merge = optimization._ChooseFastestDataset([dataset for _ in range(5)])
    self.assertDatasetProduces(
        merge,
        expected_output=[0, 1, 2, 3, 4],
        expected_shapes=dataset_ops.get_legacy_output_shapes(dataset))

  def testChooseFastest(self):
    dataset = dataset_ops.Dataset.range(600)
    f = lambda x: 2 * x
    dataset_a = dataset.batch(50).map(f)
    dataset_b = dataset.map(f).batch(50)
    merge = optimization._ChooseFastestDataset([dataset_a, dataset_b])
    self.assertDatasetProduces(
        merge,
        expected_output=[
            [i * 2 for i in range(j * 50, (j + 1) * 50)] for j in range(12)
        ],
        expected_shapes=dataset_ops.get_legacy_output_shapes(dataset_a))

  @parameterized.named_parameters(
      ("Shapes", [0], [[1, 2, 3]], "must have compatible output shapes."),
      ("Types", [0], [0.0], "must have the same output types."),
      ("NumComponents", [0], ([0], [1]), "must have the same output types."),
      ("Cardinality", [1, 2, 3], [1], "must have compatible cardinalities."))
  def testChooseFastestErrorWithIncompatibleInput(self, slices_a, slices_b,
                                                  error_msg):
    dataset_a = dataset_ops.Dataset.from_tensor_slices(slices_a)
    dataset_b = dataset_ops.Dataset.from_tensor_slices(slices_b)

    # The error is raised at dataset creation time.
    if context.executing_eagerly():
      with self.assertRaises(errors.InvalidArgumentError):
        merge = optimization._ChooseFastestDataset([dataset_a, dataset_b])
    else:
      merge = optimization._ChooseFastestDataset([dataset_a, dataset_b])
      self.assertDatasetProduces(
          merge, expected_error=(errors.InvalidArgumentError, error_msg))


if __name__ == "__main__":
  test.main()
