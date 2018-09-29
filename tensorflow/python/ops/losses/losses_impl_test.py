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
"""Tests for losses util."""

from __future__ import absolute_import

from tensorflow.python.framework import constant_op
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test


class CosineDistanceTest(test.TestCase):

  def testReshapeInput(self):
    # Two logical orthogonal vectors with cosine distance 1.
    vec_a = [1, 1, 1, 1]
    vec_b = [1, -1, 1, -1]

    # Cosine distance, viewing vec_a and vec_b as 1D vectors.
    with self.test_session():
      self.assertEqual(
          1.0,
          losses.cosine_distance(
              constant_op.constant(vec_a),
              constant_op.constant(vec_b),
              axis=0).eval())

    # Same as above, but viewing vec_a and vec_b as 2D tensors.
    with self.test_session():
      self.assertEqual(
          1.0,
          losses.cosine_distance(
              constant_op.constant(vec_a, shape=[2, 2]),
              constant_op.constant(vec_b, shape=[2, 2]),
              axis=[0, 1]).eval())


if __name__ == '__main__':
  test.main()
