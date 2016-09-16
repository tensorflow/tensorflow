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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class MatrixBandPartTest(tf.test.TestCase):
  pass  # Filled in below


def _GetMatrixBandPartTest(dtype_, batch_shape_, shape_):

  def Test(self):
    mat = np.ones(shape_).astype(dtype_)
    batch_mat = np.tile(mat, batch_shape + (1, 1))
    with self.test_session(use_gpu=True):
      for lower in -1, 0, 1, shape_[-2] - 1:
        for upper in -1, 0, 1, shape_[-1] - 1:
          band_np = mat
          if lower >= 0:
            band_np = np.triu(band_np, -lower)
          if upper >= 0:
            band_np = np.tril(band_np, upper)
          if batch_shape is not ():
            band_np = np.tile(band_np, batch_shape + (1, 1))
          band = tf.matrix_band_part(batch_mat, lower, upper)
          self.assertAllEqual(band_np, band.eval())

  return Test


class MatrixBandPartGradTest(tf.test.TestCase):
  pass  # Filled in below


def _GetMatrixBandPartGradTest(dtype_, batch_shape_, shape_):

  def Test(self):
    shape = batch_shape_ + shape_
    x = tf.constant(np.random.rand(*shape), dtype=dtype_)
    with self.test_session(use_gpu=True):
      for lower in -1, 0, 1, shape_[-2] - 1:
        for upper in -1, 0, 1, shape_[-1] - 1:
          y = tf.matrix_band_part(x, lower, upper)
          error = tf.test.compute_gradient_error(x, x.get_shape().as_list(), y,
                                                 y.get_shape().as_list())
          self.assertLess(error, 1e-4)

  return Test


if __name__ == '__main__':
  for dtype in np.int32, np.int64, np.float32, np.float64:
    for batch_shape in ((), (2,), (1, 3, 2)):
      for rows in 1, 2, 7:
        for cols in 1, 2, 7:
          shape = (rows, cols)
          name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
          setattr(MatrixBandPartTest, 'testMatrixBandPart_' + name,
                  _GetMatrixBandPartTest(dtype, batch_shape, shape))
          if dtype == np.float32 or dtype == np.float64:
            setattr(MatrixBandPartGradTest, 'testMatrixBandPartGrad_' + name,
                    _GetMatrixBandPartGradTest(dtype, batch_shape, shape))

  tf.test.main()
