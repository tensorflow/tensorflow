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

import abc
import numpy as np
import six
import tensorflow as tf

from tensorflow.contrib.distributions.python.ops import operator_pd_diag
from tensorflow.contrib.distributions.python.ops import operator_test_util


@six.add_metaclass(abc.ABCMeta)
class OperatorPDDiagBaseTest(object):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_pd_diag(self, diag_shape):
    return self._rng.rand(*diag_shape) + 0.1

  @abc.abstractmethod
  def _diag_to_matrix(self, diag):
    pass

  @abc.abstractproperty
  def operator_class(self):
    # Return the operator class that this tests.
    pass

  def _build_operator_and_mat(self, batch_shape, k, dtype=np.float64):
    # Create a diagonal matrix explicitly.
    # Create an OperatorPDSqrtDiag using the same diagonal.
    # The operator should have the same behavior.
    #
    batch_shape = list(batch_shape)
    diag_shape = batch_shape + [k]

    # The diag is the square root.
    diag = self._random_pd_diag(diag_shape).astype(dtype)
    mat = self._diag_to_matrix(diag).astype(dtype)
    operator = self.operator_class(diag)

    return operator, mat

  def test_non_positive_definite_matrix_raises(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      diag = [1.0, 0.0]
      operator = operator_pd_diag.OperatorPDSqrtDiag(diag)
      with self.assertRaisesOpError("assert_positive"):
        operator.to_dense().eval()

  def test_non_positive_definite_matrix_does_not_raise_if_not_verify_pd(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      diag = [1.0, 0.0]
      operator = operator_pd_diag.OperatorPDSqrtDiag(diag, verify_pd=False)
      operator.to_dense().eval()  # Should not raise


class OperatorPDDiagTest(
    OperatorPDDiagBaseTest, operator_test_util.OperatorPDDerivedClassTest):
  """Most tests done in the base classes."""

  def _diag_to_matrix(self, diag):
    return tf.matrix_diag(diag).eval()

  @property
  def operator_class(self):
    return operator_pd_diag.OperatorPDDiag


class OperatorPDSqrtDiagTest(
    OperatorPDDiagBaseTest, operator_test_util.OperatorPDDerivedClassTest):
  """Most tests done in the base classes."""

  def _diag_to_matrix(self, diag):
    return tf.matrix_diag(diag**2).eval()

  @property
  def operator_class(self):
    return operator_pd_diag.OperatorPDSqrtDiag


if __name__ == "__main__":
  tf.test.main()
