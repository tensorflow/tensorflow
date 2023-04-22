# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test


class _LinearOperatorTriDiagBase(object):

  def build_operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False,
      diagonals_format='sequence'):
    shape = list(build_info.shape)

    # Ensure that diagonal has large enough values. If we generate a
    # self adjoint PD matrix, then the diagonal will be dominant guaranteeing
    # positive definitess.
    diag = linear_operator_test_util.random_sign_uniform(
        shape[:-1], minval=4., maxval=6., dtype=dtype)
    # We'll truncate these depending on the format
    subdiag = linear_operator_test_util.random_sign_uniform(
        shape[:-1], minval=1., maxval=2., dtype=dtype)
    if ensure_self_adjoint_and_pd:
      # Abs on complex64 will result in a float32, so we cast back up.
      diag = math_ops.cast(math_ops.abs(diag), dtype=dtype)
      # The first element of subdiag is ignored. We'll add a dummy element
      # to superdiag to pad it.
      superdiag = math_ops.conj(subdiag)
      superdiag = manip_ops.roll(superdiag, shift=-1, axis=-1)
    else:
      superdiag = linear_operator_test_util.random_sign_uniform(
          shape[:-1], minval=1., maxval=2., dtype=dtype)

    matrix_diagonals = array_ops.stack(
        [superdiag, diag, subdiag], axis=-2)
    matrix = gen_array_ops.matrix_diag_v3(
        matrix_diagonals,
        k=(-1, 1),
        num_rows=-1,
        num_cols=-1,
        align='LEFT_RIGHT',
        padding_value=0.)

    if diagonals_format == 'sequence':
      diagonals = [superdiag, diag, subdiag]
    elif diagonals_format == 'compact':
      diagonals = array_ops.stack([superdiag, diag, subdiag], axis=-2)
    elif diagonals_format == 'matrix':
      diagonals = matrix

    lin_op_diagonals = diagonals

    if use_placeholder:
      if diagonals_format == 'sequence':
        lin_op_diagonals = [array_ops.placeholder_with_default(
            d, shape=None) for d in lin_op_diagonals]
      else:
        lin_op_diagonals = array_ops.placeholder_with_default(
            lin_op_diagonals, shape=None)

    operator = linalg_lib.LinearOperatorTridiag(
        diagonals=lin_op_diagonals,
        diagonals_format=diagonals_format,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)
    return operator, matrix

  @staticmethod
  def operator_shapes_infos():
    shape_info = linear_operator_test_util.OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shape_info((3, 3)),
        shape_info((1, 6, 6)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 3, 3))
    ]


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagCompactTest(
    _LinearOperatorTriDiagBase,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    return self.build_operator_and_matrix(
        build_info, dtype, use_placeholder,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        diagonals_format='compact')

  def test_tape_safe(self):
    diag = variables_module.Variable([[3., 6., 2.], [2., 4., 2.], [5., 1., 2.]])
    operator = linalg_lib.LinearOperatorTridiag(
        diag, diagonals_format='compact')
    self.check_tape_safe(operator)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagSequenceTest(
    _LinearOperatorTriDiagBase,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    return self.build_operator_and_matrix(
        build_info, dtype, use_placeholder,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        diagonals_format='sequence')

  def test_tape_safe(self):
    diagonals = [
        variables_module.Variable([3., 6., 2.]),
        variables_module.Variable([2., 4., 2.]),
        variables_module.Variable([5., 1., 2.])]
    operator = linalg_lib.LinearOperatorTridiag(
        diagonals, diagonals_format='sequence')
    # Skip the diagonal part and trace since this only dependent on the
    # middle variable. We test this below.
    self.check_tape_safe(operator, skip_options=['diag_part', 'trace'])

    diagonals = [
        [3., 6., 2.],
        variables_module.Variable([2., 4., 2.]),
        [5., 1., 2.]
    ]
    operator = linalg_lib.LinearOperatorTridiag(
        diagonals, diagonals_format='sequence')


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagMatrixTest(
    _LinearOperatorTriDiagBase,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    return self.build_operator_and_matrix(
        build_info, dtype, use_placeholder,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        diagonals_format='matrix')

  def test_tape_safe(self):
    matrix = variables_module.Variable([[3., 2., 0.], [1., 6., 4.], [0., 2, 2]])
    operator = linalg_lib.LinearOperatorTridiag(
        matrix, diagonals_format='matrix')
    self.check_tape_safe(operator)


if __name__ == '__main__':
  if not test_util.is_xla_enabled():
    linear_operator_test_util.add_tests(LinearOperatorTriDiagCompactTest)
    linear_operator_test_util.add_tests(LinearOperatorTriDiagSequenceTest)
    linear_operator_test_util.add_tests(LinearOperatorTriDiagMatrixTest)
  test.main()
