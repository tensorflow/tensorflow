"""Operations for linear algebra."""

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
