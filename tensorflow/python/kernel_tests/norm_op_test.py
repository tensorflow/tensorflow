# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.norm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _AddTest(test, test_name, fn):
  test_name = "_".join(["test", test_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class NormOpTest(tf.test.TestCase):

  def testBadOrder(self):
    matrix = [[0., 1.], [2., 3.]]
    for order_ in "foo", -7, -1.1, 0:
      with self.assertRaisesRegexp(ValueError,
                                   "'order' must be a supported vector norm"):
        tf.norm(matrix, order="fro")

    for order_ in "foo", -7, -1.1, 0:
      with self.assertRaisesRegexp(ValueError,
                                   "'order' must be a supported vector norm"):
        tf.norm(matrix, order=order_, axis=-1)

    for order_ in 1.1, 2:
      with self.assertRaisesRegexp(ValueError,
                                   "'order' must be a supported matrix norm"):
        tf.norm(matrix, order=order_, axis=[-2, -1])

  def testInvalidAxis(self):
    matrix = [[0., 1.], [2., 3.]]
    for axis_ in [], [1, 2, 3], [[1]], [[1], [2]], [3.1415], [1, 1]:
      error_prefix = ("'axis' must be None, an integer, or a tuple of 2 unique "
                      "integers")
      with self.assertRaisesRegexp(ValueError, error_prefix):
        tf.norm(matrix, axis=axis_)


def _GetNormOpTest(dtype_, shape_, order_, axis_, keep_dims_,
                   use_static_shape_):

  def _CompareNorm(self, matrix):
    np_norm = np.linalg.norm(
        matrix, ord=order_, axis=axis_, keepdims=keep_dims_)
    with self.test_session(use_gpu=True) as sess:
      if use_static_shape_:
        tf_matrix = tf.constant(matrix)
        tf_norm = tf.norm(
            tf_matrix, order=order_, axis=axis_, keep_dims=keep_dims_)
        tf_norm_val = sess.run(tf_norm)
      else:
        tf_matrix = tf.placeholder(dtype_)
        tf_norm = tf.norm(
            tf_matrix, order=order_, axis=axis_, keep_dims=keep_dims_)
        tf_norm_val = sess.run(tf_norm, feed_dict={tf_matrix: matrix})
    self.assertAllClose(np_norm, tf_norm_val)

  def Test(self):
    is_matrix_norm = (isinstance(axis_, tuple) or
                      isinstance(axis_, list)) and len(axis_) == 2
    is_fancy_p_norm = np.isreal(order_) and np.floor(order_) != order_
    if ((not is_matrix_norm and order_ == "fro") or
        (is_matrix_norm and is_fancy_p_norm)):
      self.skipTest("Not supported by neither numpy.linalg.norm nor tf.norm")
    if is_matrix_norm and order_ == 2:
      self.skipTest("Not supported by tf.norm")
    if axis_ is None and len(shape) > 2:
      self.skipTest("Not supported by numpy.linalg.norm")
    matrix = np.random.randn(*shape_).astype(dtype_)
    _CompareNorm(self, matrix)

  return Test


if __name__ == "__main__":
  for use_static_shape in False, True:
    for dtype in np.float32, np.float64, np.complex64, np.complex128:
      for rows in 2, 5:
        for cols in 2, 5:
          for batch in [], [2], [2, 3]:
            shape = batch + [rows, cols]
            for order in "fro", 0.5, 1, 2, np.inf:
              for axis in [
                  None, (-2, -1), (-1, -2), -len(shape), 0, len(shape) - 1
              ]:
                for keep_dims in False, True:
                  name = "%s_%s_ord_%s_axis_%s_%s_%s" % (
                      dtype.__name__, "_".join(map(str, shape)), order, axis,
                      keep_dims, use_static_shape)
                  _AddTest(NormOpTest, "Norm_" + name,
                           _GetNormOpTest(dtype, shape, order, axis, keep_dims,
                                          use_static_shape))

  tf.test.main()
