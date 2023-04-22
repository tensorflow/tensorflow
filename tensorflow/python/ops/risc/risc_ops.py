# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""RISC Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.ops import gen_risc_ops


def risc_abs(x, name='RISC_ABS'):
  return gen_risc_ops.risc_abs(x, name=name)


def risc_add(
    input_lhs,
    input_rhs,
    name='RISC_ADD'):
  return gen_risc_ops.risc_add(input_lhs, input_rhs, name=name)


def risc_binary_arithmetic(x, y, op_type, name='RISC_BinaryArithmetic'):
  return gen_risc_ops.risc_binary_arithmetic(x, y, op_type=op_type, name=name)


def risc_binary_comparison(x, y, op_type, name='RISC_BinaryComparison'):
  return gen_risc_ops.risc_binary_comparison(x, y, op_type=op_type, name=name)


def risc_bitcast(x, dtype, name='RISC_BITCAST'):
  return gen_risc_ops.risc_bitcast(x, dtype, name=name)


def risc_broadcast(x, shape, name='RISC_BROADCAST'):
  return gen_risc_ops.risc_broadcast(x, shape, name=name)


def risc_cast(x, dtype, name='RISC_CAST'):
  return gen_risc_ops.risc_cast(x, dtype, name=name)


def risc_ceil(x, name='RISC_CEIL'):
  return gen_risc_ops.risc_ceil(x, name=name)


def risc_cos(x, name='RISC_COS'):
  return gen_risc_ops.risc_cos(x, name=name)


def risc_cholesky(x, name='RISC_CHOLESKY'):
  return gen_risc_ops.risc_cholesky(x, name=name)


def risc_concat(x, axis, name='RISC_CONCAT'):
  return gen_risc_ops.risc_concat(x, axis, name=name)


def risc_condition(pred,
                   input_true,
                   input_false,
                   func_true,
                   func_false,
                   name='RISC_CONDITION'):
  return gen_risc_ops.risc_condition(
      pred,
      input_true,
      input_false,
      func_true=func_true,
      func_false=func_false,
      name=name)


def risc_conv(x,
              kernel,
              strides,
              data_format='NHWC',
              dilations=None,
              name='RISC_CONV'):
  return gen_risc_ops.risc_conv(
      x,
      kernel,
      strides,
      data_format=data_format,
      dilations=dilations,
      name=name)


def risc_div(input_lhs, input_rhs, name='RISC_DIV'):
  return gen_risc_ops.risc_div(input_lhs, input_rhs, name=name)


def risc_dot(input_lhs,
             input_rhs,
             transpose_a=False,
             transpose_b=False,
             name='RISC_DOT'):
  return gen_risc_ops.risc_dot(
      input_lhs,
      input_rhs,
      transpose_a=transpose_a,
      transpose_b=transpose_b,
      name=name)


def risc_exp(x, name='RISC_EXP'):
  return gen_risc_ops.risc_exp(x, name=name)


def risc_fft(x, name='RISC_FFT'):
  return gen_risc_ops.risc_fft(x, name=name)


def risc_floor(x, name='RISC_FLOOR'):
  return gen_risc_ops.risc_floor(x, name=name)


def risc_gather(params,
                indices,
                validate_indices=None,
                axis=None,
                batch_dims=0,
                name='RISC_GATHER'):
  return gen_risc_ops.risc_gather(
      params,
      indices,
      validate_indices=validate_indices,
      name=name,
      axis=axis,
      batch_dims=batch_dims)


def risc_imag(x, name='RISC_IMAG'):
  return gen_risc_ops.risc_imag(x, name=name)


def risc_is_finite(x, name='RISC_IS_FINITE'):
  return gen_risc_ops.risc_is_finite(x, name=name)


def risc_log(x, name='RISC_LOG'):
  return gen_risc_ops.risc_log(x, name=name)


def risc_logical_and(a, b, name='RISC_LOGICAL_AND'):
  return gen_risc_ops.risc_logical_and(a, b, name=name)


def risc_logical_not(a, b, name='RISC_LOGICAL_NOT'):
  return gen_risc_ops.risc_logical_not(a, b, name=name)


def risc_logical_or(a, b, name='RISC_LOGICAL_OR'):
  return gen_risc_ops.risc_logical_or(a, b, name=name)


def risc_max(input_lhs, input_rhs, name='RISC_MAX'):
  return gen_risc_ops.risc_max(input_lhs, input_rhs, name=name)


def risc_min(input_lhs, input_rhs, name='RISC_MIN'):
  return gen_risc_ops.risc_min(input_lhs, input_rhs, name=name)


def risc_mul(input_lhs, input_rhs, name='RISC_MUL'):
  return gen_risc_ops.risc_mul(input_lhs, input_rhs, name=name)


def risc_neg(x, name='RISC_NEG'):
  return gen_risc_ops.risc_neg(x, name=name)


def risc_pad(x, padding, constant_values, name='RISC_PAD'):
  return gen_risc_ops.risc_pad(x, padding, constant_values, name=name)


def risc_pool(x, ksize, strides, pooling_type='MAX', name='RISC_POOL'):
  return gen_risc_ops.risc_pool(
      x, ksize, strides, pooling_type=pooling_type, name=name)


def risc_pow(input_lhs, input_rhs, name='RISC_POW'):
  return gen_risc_ops.risc_pow(input_lhs, input_rhs, name=name)


def risc_random_uniform(shape, seed, name='RISC_RANDOM_UNIFORM'):
  return gen_risc_ops.risc_random_uniform(shape, seed, name=name)


def risc_real(x, name='RISC_REAL'):
  return gen_risc_ops.risc_real(x, name=name)


def risc_reduce(x, axis, reduce_type, name='RISC_REDUCE'):
  return gen_risc_ops.risc_reduce(x, axis, reduce_type=reduce_type, name=name)


def risc_rem(x, name='RISC_REM'):
  return gen_risc_ops.risc_rem(x, name=name)


def risc_reshape(x, shape, name='RISC_RESHAPE'):
  return gen_risc_ops.risc_reshape(x, shape, name=name)


def risc_reverse(x, axis, name='RISC_REVERSE'):
  return gen_risc_ops.risc_reverse(x, axis, name=name)


def risc_scatter(indices, updates, shape, name='RISC_SCATTER'):
  return gen_risc_ops.risc_scatter(indices, updates, shape, name=name)


def risc_shape(x, name='RISC_SHAPE'):
  return gen_risc_ops.risc_shape(x, name=name)


def risc_sign(x, name='RISC_SIGN'):
  return gen_risc_ops.risc_sign(x, name=name)


def risc_slice(x, begin, size, name='RISC_SLICE'):
  return gen_risc_ops.risc_slice(x, begin, size, name=name)


def risc_sub(input_lhs, input_rhs, name='RISC_SUB'):
  return gen_risc_ops.risc_sub(input_lhs, input_rhs, name=name)


def risc_sort(x, axis, direction='ASCENDING', name='RISC_SORT'):
  return gen_risc_ops.risc_sort(x, axis, direction=direction, name=name)


def risc_squeeze(x, axis=None, name='RISC_SQUEEZE'):
  return gen_risc_ops.risc_squeeze(x, axis, name=name)


def risc_transpose(x, perm=None, name='RISC_TRANSPOSE'):
  return gen_risc_ops.risc_transpose(x, perm, name=name)


def risc_triangular_solve(matrix,
                          rhs,
                          lower=True,
                          adjoint=False,
                          name='RISC_TRIANGULAR_SOLVE'):
  return gen_risc_ops.risc_triangular_solve(
      matrix, rhs, lower=lower, adjoint=adjoint, name=name)


def risc_unary(x, op_type='ABL', name='RISC_UNARY'):
  return gen_risc_ops.risc_unary(x, op_type=op_type, name=name)


def risc_while(cond,
               body,
               loop_vars,
               shape_invariants=None,
               parallel_iterations=10,
               back_prop=True,
               swap_memory=False,
               maximum_iterations=None,
               name='RISC_WHILE'):
  return gen_risc_ops.risc_while(
      cond=cond,
      body=body,
      loop_vars=loop_vars,
      shape_invariants=shape_invariants,
      parallel_iterations=parallel_iterations,
      back_prop=back_prop,
      swap_memory=swap_memory,
      name=name,
      maximum_iterations=maximum_iterations,
      return_same_structure=True)
