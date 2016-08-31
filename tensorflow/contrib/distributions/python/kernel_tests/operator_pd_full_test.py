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

from tensorflow.contrib.distributions.python.ops import operator_pd_full


class OperatorPDFullTest(tf.test.TestCase):
  # The only method needing checked (because it isn't part of the parent class)
  # is the check for symmetry.

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_positive_def_array(self, *shape):
    matrix = self._rng.rand(*shape)
    return tf.batch_matmul(matrix, matrix, adj_y=True).eval()

  def test_positive_definite_matrix_doesnt_raise(self):
    with self.test_session():
      matrix = self._random_positive_def_array(2, 3, 3)
      operator = operator_pd_full.OperatorPDFull(matrix, verify_pd=True)
      operator.to_dense().eval()  # Should not raise

  def test_negative_definite_matrix_raises(self):
    with self.test_session():
      matrix = -1 * self._random_positive_def_array(3, 2, 2)
      operator = operator_pd_full.OperatorPDFull(matrix, verify_pd=True)
      # Could fail inside Cholesky decomposition, or later when we test the
      # diag.
      with self.assertRaisesOpError("x > 0|LLT"):
        operator.to_dense().eval()

  def test_non_symmetric_matrix_raises(self):
    with self.test_session():
      matrix = self._random_positive_def_array(3, 2, 2)
      matrix[0, 0, 1] += 0.001
      operator = operator_pd_full.OperatorPDFull(matrix, verify_pd=True)
      with self.assertRaisesOpError("x == y"):
        operator.to_dense().eval()


if __name__ == "__main__":
  tf.test.main()
