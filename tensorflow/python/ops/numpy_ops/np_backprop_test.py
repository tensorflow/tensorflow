# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for backpropgration on tf-numpy functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import backprop
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_array_ops
# Required for operator overloads
from tensorflow.python.ops.numpy_ops import np_math_ops  # pylint: disable=unused-import
from tensorflow.python.platform import test


class BackpropTest(test.TestCase):

  def test_setitem(self):
    # Single integer index.
    a = np_array_ops.array([1., 2., 3.])
    b = np_array_ops.array(5.)
    c = np_array_ops.array(10.)

    tensors = [arr.data for arr in [a, b, c]]
    with backprop.GradientTape() as g:
      g.watch(tensors)
      a[1] = b + c
      loss = np_array_ops.sum(a)

    gradients = g.gradient(loss.data, tensors)
    self.assertSequenceEqual(
        np_array_ops.array(gradients[0]).tolist(), [1., 0., 1.])
    self.assertEqual(np_array_ops.array(gradients[1]).tolist(), 1.)
    self.assertEqual(np_array_ops.array(gradients[2]).tolist(), 1.)

    # Tuple index.
    a = np_array_ops.array([[[1., 2.], [3., 4.]], [[5., 6.],
                                                   [7., 8.]]])  # 2x2x2 array.
    b = np_array_ops.array([10., 11.])

    tensors = [arr.data for arr in [a, b]]
    with backprop.GradientTape() as g:
      g.watch(tensors)
      a[(1, 0)] = b
      loss = np_array_ops.sum(a)

    gradients = g.gradient(loss.data, tensors)
    self.assertSequenceEqual(
        np_array_ops.array(gradients[0]).tolist(),
        [[[1., 1.], [1., 1.]], [[0., 0.], [1., 1.]]])
    self.assertEqual(np_array_ops.array(gradients[1]).tolist(), [1., 1.])


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
