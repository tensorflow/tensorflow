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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SvdOpTest(tf.test.TestCase):

  def testWrongDimensions(self):
    # The input to svd should be 2-dimensional tensor.
    scalar = tf.constant(1.)
    with self.assertRaises(ValueError):
      tf.svd(scalar)
    vector = tf.constant([1., 2.])
    with self.assertRaises(ValueError):
      tf.svd(vector)
    tensor = tf.constant([[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]])
    with self.assertRaises(ValueError):
      tf.svd(tensor)

    # The input to batch_svd should be a tensor of at least rank 2.
    scalar = tf.constant(1.)
    with self.assertRaises(ValueError):
      tf.batch_svd(scalar)
    vector = tf.constant([1., 2.])
    with self.assertRaises(ValueError):
      tf.batch_svd(vector)


def _GetSvdOpTest(dtype_, shape_):

  def _CompareSingularVectors(self, x, y, atol):
    # Singular vectors are only unique up to sign (complex phase factor for
    # complex matrices), so we normalize the signs first.
    signs = np.sign(np.sum(np.divide(x, y), -2, keepdims=True))
    x *= signs
    self.assertAllClose(x, y, atol=atol)

  def Test(self):
    np.random.seed(1)
    x = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    if dtype_ == np.float32:
      atol = 1e-4
    else:
      atol = 1e-14
    for compute_uv in False, True:
      for full_matrices in False, True:
        with self.test_session():
          if x.ndim == 2:
            if compute_uv:
              tf_s, tf_u, tf_v = tf.svd(tf.constant(x),
                                        compute_uv=compute_uv,
                                        full_matrices=full_matrices)
            else:
              tf_s = tf.svd(tf.constant(x),
                            compute_uv=compute_uv,
                            full_matrices=full_matrices)
          else:
            if compute_uv:
              tf_s, tf_u, tf_v = tf.batch_svd(
                  tf.constant(x),
                  compute_uv=compute_uv,
                  full_matrices=full_matrices)
            else:
              tf_s = tf.batch_svd(
                  tf.constant(x),
                  compute_uv=compute_uv,
                  full_matrices=full_matrices)
          if compute_uv:
            np_u, np_s, np_v = np.linalg.svd(x,
                                             compute_uv=compute_uv,
                                             full_matrices=full_matrices)
          else:
            np_s = np.linalg.svd(x,
                                 compute_uv=compute_uv,
                                 full_matrices=full_matrices)
          self.assertAllClose(np_s, tf_s.eval(), atol=atol)
          if compute_uv:
            _CompareSingularVectors(self, np_u, tf_u.eval(), atol)
            _CompareSingularVectors(self, np.swapaxes(np_v, -2, -1),
                                    tf_v.eval(), atol)

  return Test


if __name__ == '__main__':
  for dtype in np.float32, np.float64:
    for m in 1, 2, 5, 10:
      for n in 1, 2, 5, 10:
        for batch_dims in [(), (3,)] + [(3, 2)] * (max(m, n) < 10):
          shape = batch_dims + (m, n)
          name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
          setattr(SvdOpTest, 'testSvd_' + name, _GetSvdOpTest(dtype, shape))
  tf.test.main()
