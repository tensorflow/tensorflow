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
"""Tests for array operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ArrayOpTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def testGatherGradHasPartialStaticShape(self):
    # Create a tensor with an unknown dim 1.
    x = random_ops.random_normal([4, 10, 10])
    x = array_ops.gather(
        x,
        array_ops.reshape(array_ops.where_v2(x[0, :, 0] > 0.5), [-1]),
        axis=1)
    self.assertAllEqual(x.shape.as_list(), [4, None, 10])

    a = array_ops.gather(array_ops.gather(x, [0, 1]), [0, 1])
    b = array_ops.gather(array_ops.gather(x, [2, 3], axis=2), [0, 1])
    grad_a = ops.convert_to_tensor(gradients.gradients(a, x)[0])
    grad_b = ops.convert_to_tensor(gradients.gradients(b, x)[0])

    # We make sure that the representation of the shapes are correct; the shape
    # equality check will always eval to false due to the shapes being partial.
    self.assertAllEqual(grad_a.shape.as_list(), [None, None, 10])
    self.assertAllEqual(grad_b.shape.as_list(), [4, None, 10])

  @test_util.deprecated_graph_mode_only
  def testReshapeShapeInference(self):
    # Create a tensor with an unknown dim 1.
    x = random_ops.random_normal([4, 10, 10])
    x = array_ops.gather(
        x,
        array_ops.reshape(array_ops.where_v2(x[0, :, 0] > 0.5), [-1]),
        axis=1)
    self.assertAllEqual(x.shape.as_list(), [4, None, 10])
    a = array_ops.reshape(x, array_ops.shape(x))
    self.assertAllEqual(a.shape.as_list(), [4, None, 10])
    b = array_ops.reshape(x, math_ops.cast(array_ops.shape(x), dtypes.int64))
    self.assertAllEqual(b.shape.as_list(), [4, None, 10])

    # We do not shape-infer across a tf.cast into anything that's not tf.int32
    # or tf.int64, since they might end up mangling the shape.
    c = array_ops.reshape(
        x,
        math_ops.cast(
            math_ops.cast(array_ops.shape(x), dtypes.float32), dtypes.int32))
    self.assertAllEqual(c.shape.as_list(), [None, None, None])

  def testEmptyMeshgrid(self):
    self.assertEqual(array_ops.meshgrid(), [])


if __name__ == "__main__":
  test.main()
