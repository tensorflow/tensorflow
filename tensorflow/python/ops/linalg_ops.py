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

"""Operations for linear algebra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_linalg_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_linalg_ops import *
# pylint: enable=wildcard-import


@ops.RegisterShape("Cholesky")
def _CholeskyShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  return [input_shape]


@ops.RegisterShape("BatchCholesky")
def _BatchCholeskyShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  return [input_shape]


@ops.RegisterShape("MatrixDeterminant")
def _MatrixDeterminantShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  if input_shape.ndims is not None:
    return [tensor_shape.scalar()]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("BatchMatrixDeterminant")
def _BatchMatrixDeterminantShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  if input_shape.ndims is not None:
    return [input_shape[:-2]]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("MatrixInverse")
def _MatrixInverseShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  return [input_shape]


@ops.RegisterShape("BatchMatrixInverse")
def _BatchMatrixInverseShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  return [input_shape]


@ops.RegisterShape("SelfAdjointEig")
def _SelfAdjointEigShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  d = input_shape.dims[0]
  out_shape = tensor_shape.TensorShape([d+1, d])
  return [out_shape]


@ops.RegisterShape("BatchSelfAdjointEig")
def _BatchSelfAdjointEigShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  dlist = input_shape.dims
  dlist[-2] += 1
  out_shape = tensor_shape.TensorShape(dlist)
  return [out_shape]
