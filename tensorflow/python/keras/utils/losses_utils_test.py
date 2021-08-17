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
"""Tests for losses_utils."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class RemoveSqueezableTest(test_util.TensorFlowTestCase):
  """Test remove_squeezable_dimensions"""

  def test_ragged_3d_same_shape(self):
    """ shape (2, (sequence={1, 2}), 3)"""
    x = ragged_factory_ops.constant([[[1, 2, 3]], [[4, 5, 6], [7, 8, 9]]])
    rank = x.shape.ndims
    x_p, _ = losses_utils.remove_squeezable_dimensions(x, x)
    self.assertEqual(x_p.shape.ndims, rank)

  def test_ragged_3d_4d_squeezable(self):
    """ shapes:

        x: (2, (sequence={1, 2}), 3)
        y: (2, (sequence={1, 2}), 3, 1)
    """
    x = ragged_factory_ops.constant([[[1, 2, 3]], [[4, 5, 6], [7, 8, 9]]])
    y = array_ops.expand_dims(x, axis=-1)
    self.assertEqual(x.shape.ndims, 3)
    self.assertEqual(y.shape.ndims, 4)
    _, y_p = losses_utils.remove_squeezable_dimensions(x, y)
    y_p.shape.assert_is_compatible_with(x.shape)
    self.assertEqual(y_p.shape.ndims, 3)

    x_p, _ = losses_utils.remove_squeezable_dimensions(y, x)
    x_p.shape.assert_is_compatible_with(x.shape)
    self.assertEqual(x_p.shape.ndims, 3)

  def test_dense_2d_3d_squeezable(self):
    x = constant_op.constant([[1, 2], [3, 4]])
    y = constant_op.constant([[[1], [2]], [[3], [4]]])
    _, y_p = losses_utils.remove_squeezable_dimensions(x, y)
    y_p.shape.assert_is_compatible_with(x.shape)
    self.assertEqual(y_p.shape.ndims, x.shape.ndims)
    x_p, _ = losses_utils.remove_squeezable_dimensions(y, x)
    x_p.shape.assert_is_compatible_with(x.shape)


class RemoveSqueezableTestGraphOnly(test_util.TensorFlowTestCase):
  """Test remove_squeezable_dimensions (graph-mode only)."""

  def test_placeholder(self):
    """Test dynamic rank tensors."""
    with ops.Graph().as_default():
      x = array_ops.placeholder_with_default([1., 2., 3.], shape=None)
      y = array_ops.placeholder_with_default([[1.], [2.], [3.]], shape=None)
      _, y_p = losses_utils.remove_squeezable_dimensions(x, y)
      y_p.shape.assert_is_compatible_with(x.shape)
      self.assertAllEqual(array_ops.shape(x), array_ops.shape(y_p))
      x_p, _ = losses_utils.remove_squeezable_dimensions(y, x)
      x_p.shape.assert_is_compatible_with(x.shape)


if __name__ == '__main__':
  test.main()
