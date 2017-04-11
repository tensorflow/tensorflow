# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for common shapes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class CommonShapesTest(test_util.TensorFlowTestCase):

  def testBroadcast_one_dimension(self):
    s1 = tensor_shape.vector(5)
    s2 = tensor_shape.vector(7)

    unknown = tensor_shape.unknown_shape()
    scalar = tensor_shape.scalar()
    expanded_scalar = tensor_shape.TensorShape([1])

    # Tensors with same shape should have the same broadcast result.
    self.assertEqual(s1, common_shapes.broadcast_shape(s1, s1))
    self.assertEqual(s2, common_shapes.broadcast_shape(s2, s2))
    self.assertEqual(unknown, common_shapes.broadcast_shape(unknown, unknown))
    self.assertEqual(scalar, common_shapes.broadcast_shape(scalar, scalar))
    self.assertEqual(expanded_scalar, common_shapes.broadcast_shape(
        expanded_scalar, expanded_scalar))

    # [] acts like an identity.
    self.assertEqual(s1, common_shapes.broadcast_shape(s1, scalar))
    self.assertEqual(s2, common_shapes.broadcast_shape(s2, scalar))

    self.assertEqual(s1, common_shapes.broadcast_shape(s1, expanded_scalar))
    self.assertEqual(s2, common_shapes.broadcast_shape(s2, expanded_scalar))

    self.assertEqual(unknown, common_shapes.broadcast_shape(s1, unknown))
    self.assertEqual(unknown, common_shapes.broadcast_shape(s2, unknown))

    self.assertEqual(expanded_scalar, common_shapes.broadcast_shape(
        scalar, expanded_scalar))

    with self.assertRaises(ValueError):
      common_shapes.broadcast_shape(s1, s2)
      common_shapes.broadcast_shape(s2, s1)


if __name__ == "__main__":
  googletest.main()
