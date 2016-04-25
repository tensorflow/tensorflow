# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.ops.tf.self_adjoint_eig."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class SelfAdjointEigOpTest(tf.test.TestCase):

  def _testEigs(self, x, d, tf_ans, use_gpu=False):
    np_eig_val, np_eig_vec = np.linalg.eig(x)

    # First check the eigenvalues
    self.assertAllClose(sorted(np_eig_val), sorted(tf_ans[0, :]))

    # need to make things canonical. This test may still fail in case there are
    # two equal eigenvalues, so that there is indeterminacy in the eigenvectors.
    # For now, assume that we will only test matrices with distinct eigenvalues.
    np_arg = np.argsort(np_eig_val)
    tf_arg = np.argsort(tf_ans[0, :])

    np_eig_vecs_sorted = np.array([np_eig_vec[:, i] for i in np_arg]).T
    tf_eig_vecs_sorted = np.array([tf_ans[1:, i] for i in tf_arg]).T
    np_eig_vecs_signed_sorted = np.array([np_eig_vecs_sorted[:, i] *
                                          np.sign(np_eig_vecs_sorted[0, i])
                                          for i in xrange(d)]).T
    tf_eig_vecs_signed_sorted = np.array([tf_eig_vecs_sorted[:, i] *
                                          np.sign(tf_eig_vecs_sorted[0, i])
                                          for i in xrange(d)]).T
    self.assertAllClose(np_eig_vecs_signed_sorted, tf_eig_vecs_signed_sorted)

  def _compareSelfAdjointEig(self, x, use_gpu=False):
    with self.test_session() as sess:
      tf_eig = tf.self_adjoint_eig(tf.constant(x))
      tf_eig_out = sess.run([tf_eig])[0]

    d, _ = x.shape
    self.assertEqual([d+1, d], tf_eig.get_shape().dims)
    self._testEigs(x, d, tf_eig_out, use_gpu)

  def _compareBatchSelfAdjointEigRank3(self, x, use_gpu=False):
    with self.test_session() as sess:
      tf_eig = tf.batch_self_adjoint_eig(tf.constant(x))
      tf_out = sess.run([tf_eig])[0]
    dlist = x.shape
    d = dlist[-2]

    self.assertEqual([d+1, d], tf_eig.get_shape().dims[-2:])
    # not testing the values.
    self.assertEqual(dlist[0], tf_eig.get_shape().dims[0])

    for i in xrange(dlist[0]):
      self._testEigs(x[i], d, tf_out[i])

  def _compareBatchSelfAdjointEigRank2(self, x, use_gpu=False):
    with self.test_session() as sess:
      tf_eig = tf.batch_self_adjoint_eig(tf.constant(x))
      tf_out = sess.run([tf_eig])[0]
    dlist = x.shape
    d = dlist[-2]

    self.assertEqual(len(tf_eig.get_shape()), 2)
    self.assertEqual([d+1, d], tf_eig.get_shape().dims[-2:])
    self._testEigs(x, d, tf_out)

  def testBasic(self):
    self._compareSelfAdjointEig(
        np.array([[3., 0., 1.], [0., 2., -2.], [1., -2., 3.]]))

  def testBatch(self):
    simple_array = np.array([[[1., 0.], [0., 5.]]])  # shape (1, 2, 2)
    simple_array_2d = simple_array[0]  # shape (2, 2)
    self._compareBatchSelfAdjointEigRank3(simple_array)
    self._compareBatchSelfAdjointEigRank3(
        np.vstack((simple_array, simple_array)))
    self._compareBatchSelfAdjointEigRank2(simple_array_2d)
    odd_sized_array = np.array([[[3., 0., 1.], [0., 2., -2.], [1., -2., 3.]]])
    self._compareBatchSelfAdjointEigRank3(
        np.vstack((odd_sized_array, odd_sized_array)))

    # Generate random positive-definite matrices.
    matrices = np.random.rand(10, 5, 5)
    for i in xrange(10):
      matrices[i] = np.dot(matrices[i].T, matrices[i])
    self._compareBatchSelfAdjointEigRank3(matrices)

  def testNonSquareMatrix(self):
    with self.assertRaises(ValueError):
      tf.self_adjoint_eig(tf.constant(np.array([[1., 2., 3.], [3., 4., 5.]])))

  def testWrongDimensions(self):
    tensor3 = tf.constant([1., 2.])
    with self.assertRaises(ValueError):
      tf.self_adjoint_eig(tensor3)


if __name__ == "__main__":
  tf.test.main()
