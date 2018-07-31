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
"""SmartMatrices definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.ops.linalg import linear_operator_util as lou


class LinearOperatorExtras(object):  # pylint: disable=missing-docstring

  def matmul(self, x, adjoint=False, adjoint_arg=False, name="matmul"):

    with self._name_scope(name, values=[x]):
      if isinstance(x, ops.IndexedSlices):
        return self._matmul_sparse(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

      x = ops.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)

      self_dim = -2 if adjoint else -1
      arg_dim = -1 if adjoint_arg else -2
      self.shape[self_dim].assert_is_compatible_with(x.get_shape()[arg_dim])

      return self._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def matmul_right(self, x, adjoint=False, adjoint_arg=False, name="matmul"):

    with self._name_scope(name, values=[x]):

      if isinstance(x, ops.IndexedSlices):
        return self._matmul_right_sparse(
            x, adjoint=adjoint, adjoint_arg=adjoint_arg)

      x = ops.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)

      self_dim = -1 if adjoint else -2
      arg_dim = -2 if adjoint_arg else -1
      self.shape[self_dim].assert_is_compatible_with(x.get_shape()[arg_dim])

      return self._matmul_right(x, adjoint=adjoint, adjoint_arg=adjoint_arg)


class LinearOperatorFullMatrix(LinearOperatorExtras,
                               linalg.LinearOperatorFullMatrix):

  # TODO(b/78117889) Remove this definition once core LinearOperator
  # has _matmul_right.
  def _matmul_right(self, x, adjoint=False, adjoint_arg=False):
    return lou.matmul_with_broadcast(
        x, self._matrix, adjoint_a=adjoint_arg, adjoint_b=adjoint)

  def _matmul_sparse(self, x, adjoint=False, adjoint_arg=False):
    raise NotImplementedError

  def _matmul_right_sparse(self, x, adjoint=False, adjoint_arg=False):
    assert not adjoint and not adjoint_arg
    return utils.matmul_sparse_dense(x, self._matrix)


class LinearOperatorDiag(LinearOperatorExtras,  # pylint: disable=missing-docstring
                         linalg.LinearOperatorDiag):

  def _matmul_right(self, x, adjoint=False, adjoint_arg=False):
    diag_mat = math_ops.conj(self._diag) if adjoint else self._diag
    x = linalg_impl.adjoint(x) if adjoint_arg else x
    return diag_mat * x

  def _matmul_sparse(self, x, adjoint=False, adjoint_arg=False):
    diag_mat = math_ops.conj(self._diag) if adjoint else self._diag
    assert not adjoint_arg
    return utils.matmul_diag_sparse(diag_mat, x)

  def _matmul_right_sparse(self, x, adjoint=False, adjoint_arg=False):
    raise NotImplementedError
