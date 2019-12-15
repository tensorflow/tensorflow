# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for metrics_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSizeOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      {
          'x_list': [1],
          'y_list': [2]
      },
      {
          'x_list': [1, 2],
          'y_list': [2, 3]
      },
      {
          'x_list': [1, 2, 4],
          'y_list': [2, 3, 5]
      },
      {
          'x_list': [[1, 2], [3, 4]],
          'y_list': [[2, 3], [5, 6]]
      },
  ])
  def test_passing_dense_tensors(self, x_list, y_list):
    x = constant_op.constant(x_list)
    y = constant_op.constant(y_list)
    [x,
     y], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y])
    x.shape.assert_is_compatible_with(y.shape)

  @parameterized.parameters([
      {
          'x_list': [1],
      },
      {
          'x_list': [1, 2],
      },
      {
          'x_list': [1, 2, 4],
      },
      {
          'x_list': [[1, 2], [3, 4]],
      },
  ])
  def test_passing_one_dense_tensor(self, x_list):
    x = constant_op.constant(x_list)
    [x], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x])

  @parameterized.parameters([
      {
          'x_list': [1],
          'y_list': [2]
      },
      {
          'x_list': [1, 2],
          'y_list': [2, 3]
      },
      {
          'x_list': [1, 2, 4],
          'y_list': [2, 3, 5]
      },
      {
          'x_list': [[1, 2], [3, 4]],
          'y_list': [[2, 3], [5, 6]]
      },
      {
          'x_list': [[1, 2], [3, 4], [1]],
          'y_list': [[2, 3], [5, 6], [3]]
      },
      {
          'x_list': [[1, 2], [], [1]],
          'y_list': [[2, 3], [], [3]]
      },
  ])
  def test_passing_both_ragged(self, x_list, y_list):
    x = ragged_factory_ops.constant(x_list)
    y = ragged_factory_ops.constant(y_list)
    [x,
     y], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y])
    x.shape.assert_is_compatible_with(y.shape)

  @parameterized.parameters([
      {
          'x_list': [1],
      },
      {
          'x_list': [1, 2],
      },
      {
          'x_list': [1, 2, 4],
      },
      {
          'x_list': [[1, 2], [3, 4]],
      },
      {
          'x_list': [[1, 2], [3, 4], [1]],
      },
      {
          'x_list': [[1, 2], [], [1]],
      },
  ])
  def test_passing_one_ragged(self, x_list):
    x = ragged_factory_ops.constant(x_list)
    [x], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x])

  @parameterized.parameters([
      {
          'x_list': [1],
          'y_list': [2],
          'mask_list': [0]
      },
      {
          'x_list': [1, 2],
          'y_list': [2, 3],
          'mask_list': [0, 1]
      },
      {
          'x_list': [1, 2, 4],
          'y_list': [2, 3, 5],
          'mask_list': [1, 1, 1]
      },
      {
          'x_list': [[1, 2], [3, 4]],
          'y_list': [[2, 3], [5, 6]],
          'mask_list': [[1, 1], [0, 1]]
      },
      {
          'x_list': [[1, 2], [3, 4], [1]],
          'y_list': [[2, 3], [5, 6], [3]],
          'mask_list': [[1, 1], [0, 0], [1]]
      },
      {
          'x_list': [[1, 2], [], [1]],
          'y_list': [[2, 3], [], [3]],
          'mask_list': [[1, 1], [], [0]]
      },
  ])
  def test_passing_both_ragged_with_mask(self, x_list, y_list, mask_list):
    x = ragged_factory_ops.constant(x_list)
    y = ragged_factory_ops.constant(y_list)
    mask = ragged_factory_ops.constant(mask_list)
    [x, y], mask = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y], mask)
    x.shape.assert_is_compatible_with(y.shape)
    y.shape.assert_is_compatible_with(mask.shape)

  @parameterized.parameters([
      {
          'x_list': [1],
          'mask_list': [0]
      },
      {
          'x_list': [1, 2],
          'mask_list': [0, 1]
      },
      {
          'x_list': [1, 2, 4],
          'mask_list': [1, 1, 1]
      },
      {
          'x_list': [[1, 2], [3, 4]],
          'mask_list': [[1, 1], [0, 1]]
      },
      {
          'x_list': [[1, 2], [3, 4], [1]],
          'mask_list': [[1, 1], [0, 0], [1]]
      },
      {
          'x_list': [[1, 2], [], [1]],
          'mask_list': [[1, 1], [], [0]]
      },
  ])
  def test_passing_one_ragged_with_mask(self, x_list, mask_list):
    x = ragged_factory_ops.constant(x_list)
    mask = ragged_factory_ops.constant(mask_list)
    [x], mask = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values([x], mask)
    x.shape.assert_is_compatible_with(mask.shape)

  @parameterized.parameters([
      {
          'x_list': [[[1, 3]]],
          'y_list': [[2, 3]]
      },
  ])
  def test_failing_different_ragged_and_dense_ranks(self, x_list, y_list):
    x = ragged_factory_ops.constant(x_list)
    y = ragged_factory_ops.constant(y_list)
    with self.assertRaises(ValueError):  # pylint: disable=g-error-prone-assert-raises
      [x, y
      ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y])

  @parameterized.parameters([
      {
          'x_list': [[[1, 3]]],
          'y_list': [[[2, 3]]],
          'mask_list': [[0, 1]]
      },
  ])
  def test_failing_different_mask_ranks(self, x_list, y_list, mask_list):
    x = ragged_factory_ops.constant(x_list)
    y = ragged_factory_ops.constant(y_list)
    mask = ragged_factory_ops.constant(mask_list)
    with self.assertRaises(ValueError):  # pylint: disable=g-error-prone-assert-raises
      [x, y
      ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y],
                                                                        mask)

  # we do not support such cases that ragged_ranks are different but overall
  # dimension shapes and sizes are identical due to adding too much performance
  # overheads to the overall use cases.
  def test_failing_different_ragged_ranks(self):
    dt = constant_op.constant([[[1, 2]]])
    # adding a ragged dimension
    x = ragged_tensor.RaggedTensor.from_row_splits(dt, row_splits=[0, 1])
    y = ragged_factory_ops.constant([[[[1, 2]]]])
    with self.assertRaises(ValueError):  # pylint: disable=g-error-prone-assert-raises
      [x, y], _ = \
          metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y])


if __name__ == '__main__':
  googletest.main()
