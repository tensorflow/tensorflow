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
"""Tests for vectorization of math kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class MathTest(PForTestCase, parameterized.TestCase):

  def _test_unary_cwise_ops(self, ops, is_complex):
    for op in ops:
      with backprop.GradientTape(persistent=True) as g:
        x = random_ops.random_uniform([3, 5])
        g.watch(x)
        if is_complex:
          y = random_ops.random_uniform([3, 5])
          g.watch(y)
          x = math_ops.complex(x, y)

      # pylint: disable=cell-var-from-loop

      def loop_fn(i):
        with g:
          y = op(x)
          x_i = array_ops.gather(x, i)
          y_i = op(x_i)
          outputs = [y_i]
          # Build cross product of loop variant/invariant outputs and gradients.
          for out in (y, y_i):
            if out.dtype == dtypes.float32:
              for output_gradients in (None, out * math_ops.cast(i, out.dtype)):
                grad = g.gradient(out, x_i, output_gradients=output_gradients)
                if grad is not None:
                  outputs.append(grad)
        return outputs

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_unary_cwise_complex_ops(self):
    complex_ops = [
        math_ops.angle,
        math_ops.imag,
        math_ops.complex_abs,
        math_ops.real,
        math_ops.conj,
    ]
    self._test_unary_cwise_ops(complex_ops, True)

  def test_unary_cwise_real_ops_1(self):
    real_ops = [
        lambda x: math_ops.acosh(1 + math_ops.square(x)),
        math_ops.abs,
        math_ops.acos,
        math_ops.asin,
        math_ops.asinh,
        math_ops.atan,
        math_ops.atanh,
        math_ops.bessel_i0e,
        math_ops.bessel_i1e,
        math_ops.cos,
        math_ops.cosh,
        math_ops.digamma,
        math_ops.erf,
        math_ops.erfc,
        math_ops.erfinv,
        math_ops.exp,
        math_ops.expm1,
        math_ops.inv,
        math_ops.is_finite,
        math_ops.is_inf,
        math_ops.lgamma,
        math_ops.log,
        math_ops.log1p,
        math_ops.ndtri,
    ]
    self._test_unary_cwise_ops(real_ops, False)

  def test_unary_cwise_real_ops_2(self):
    real_ops = [
        math_ops.neg,
        math_ops.negative,
        math_ops.reciprocal,
        math_ops.rint,
        math_ops.round,
        math_ops.rsqrt,
        math_ops.sigmoid,
        math_ops.sign,
        math_ops.sin,
        math_ops.sinh,
        math_ops.sqrt,
        math_ops.square,
        math_ops.tan,
        math_ops.tanh,
        nn.elu,
        nn.relu,
        nn.relu6,
        lambda t: nn.leaky_relu(t, alpha=0.1),
        nn.selu,
        nn.softplus,
        nn.softsign,
    ]
    self._test_unary_cwise_ops(real_ops, False)

  def test_unary_cwise_no_grad(self):
    for op in [math_ops.ceil, math_ops.floor, math_ops.logical_not]:
      x = random_ops.random_uniform([3, 5])
      if op == math_ops.logical_not:
        x = x > 0

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        return op(array_ops.gather(x, i))

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_binary_cwise_ops(self):
    # Enable tensor equality to test `equal` and `not_equal` ops below.
    default_equality = framework_ops.Tensor._USE_EQUALITY
    framework_ops.enable_tensor_equality()
    try:
      logical_ops = [
          math_ops.logical_and, math_ops.logical_or, math_ops.logical_xor
      ]

      # Wrapper functions restricting the range of inputs of zeta and polygamma.
      def safe_polygamma(x, y):
        return math_ops.polygamma(
            math_ops.round(clip_ops.clip_by_value(y, 1, 10)), x * x + 1)

      def safe_zeta(x, y):
        return math_ops.zeta(x * x + 1, y * y)

      float_ops = [
          math_ops.add,
          math_ops.add_v2,
          math_ops.atan2,
          math_ops.complex,
          math_ops.div,
          math_ops.divide,
          math_ops.div_no_nan,
          math_ops.equal,
          lambda x, y: framework_ops.convert_to_tensor(x == y),
          lambda x, y: framework_ops.convert_to_tensor(x != y),
          math_ops.floor_mod,
          math_ops.greater,
          math_ops.greater_equal,
          math_ops.igamma,
          math_ops.igammac,
          math_ops.igamma_grad_a,
          math_ops.less,
          math_ops.less_equal,
          math_ops.maximum,
          math_ops.minimum,
          math_ops.mod,
          math_ops.multiply,
          math_ops.not_equal,
          math_ops.pow,
          math_ops.squared_difference,
          math_ops.subtract,
          math_ops.truncate_mod,
          safe_polygamma,
          safe_zeta,
      ]
      # FloorDiv fails on XLA due floor's discontinuities exacerbating small
      # division differences.
      if not test_util.is_xla_enabled():
        float_ops += [math_ops.floor_div]
      for op in logical_ops + float_ops:
        x = random_ops.random_uniform([7, 3, 5])
        y = random_ops.random_uniform([3, 5])
        if op in logical_ops:
          x = x > 0
          y = y > 0

        output_dtypes = []

        # pylint: disable=cell-var-from-loop
        def loop_fn(i):
          x1 = array_ops.gather(x, i)
          y1 = array_ops.gather(y, i)
          outputs = [op(x, y), op(x1, y), op(x, y1), op(x1, y1), op(x1, x1)]
          del output_dtypes[:]
          output_dtypes.extend(t.dtype for t in outputs)
          return outputs

        # pylint: enable=cell-var-from-loop

        self._test_loop_fn(loop_fn, 3)
    finally:
      if not default_equality:
        framework_ops.disable_tensor_equality()

  def test_approximate_equal(self):
    x = random_ops.random_uniform([3, 5])
    y = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      y1 = array_ops.gather(y, i)
      return math_ops.approximate_equal(x1, y1)

    self._test_loop_fn(loop_fn, 3)

  def test_addn(self):
    x = random_ops.random_uniform([2, 3, 5])
    y = random_ops.random_uniform([3, 5])
    z = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return math_ops.add_n([x1, y, z])

    self._test_loop_fn(loop_fn, 2)

  def test_cross(self):
    x = random_ops.random_uniform([4, 2, 3])
    y = random_ops.random_uniform([4, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      x_0 = array_ops.gather(x, 0)
      return math_ops.cross(x_i, y_i), math_ops.cross(x_0, y_i)

    self._test_loop_fn(loop_fn, 4)

  def test_matmul(self):
    for tr_a in (True, False):
      for tr_b in (True, False):
        for stack_a in (True, False):
          for stack_b in (True, False):
            shape_a = (5, 3) if tr_a else (3, 5)
            if stack_a:
              shape_a = (2,) + shape_a
            shape_b = (7, 5) if tr_b else (5, 7)
            if stack_b:
              shape_b = (2,) + shape_b

            x = random_ops.random_uniform(shape_a)
            y = random_ops.random_uniform(shape_b)

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
              a = array_ops.gather(x, i) if stack_a else x
              b = array_ops.gather(y, i) if stack_b else y
              return math_ops.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)

            # pylint: enable=cell-var-from-loop

            self._test_loop_fn(loop_fn, 2)

  def test_batch_matmul(self):
    for tr_a in (True, False):
      for tr_b in (True, False):
        for stack_a in (True, False):
          for stack_b in (True, False):
            shape_a = (4, 5, 3) if tr_a else (4, 3, 5)
            if stack_a:
              shape_a = (2,) + shape_a
            shape_b = (4, 7, 5) if tr_b else (4, 5, 7)
            if stack_b:
              shape_b = (2,) + shape_b

            x = random_ops.random_uniform(shape_a)
            y = random_ops.random_uniform(shape_b)

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
              a = array_ops.gather(x, i) if stack_a else x
              b = array_ops.gather(y, i) if stack_b else y
              return math_ops.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)

            # pylint: enable=cell-var-from-loop

            self._test_loop_fn(loop_fn, 2)

  def test_batch_matmul_broadcast(self):
    for broadcast_a in (True, False):
      for broadcast_b in (True, False):
        for stack_a in (True, False):
          for stack_b in (True, False):
            shape_a = (2, 3, 5) if broadcast_a else (4, 2, 3, 5)
            shape_b = (2, 5, 7) if broadcast_b else (4, 2, 5, 7)
            shape_a = (2,) + shape_a if stack_a else shape_a
            shape_b = (2,) + shape_b if stack_b else shape_b
            x = random_ops.random_uniform(shape_a)
            y = random_ops.random_uniform(shape_b)

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
              a = array_ops.gather(x, i) if stack_a else x
              b = array_ops.gather(y, i) if stack_b else y
              return math_ops.matmul(a, b)

            # pylint: enable=cell-var-from-loop
            self._test_loop_fn(loop_fn, 2)

  def test_reduction(self):
    x = random_ops.random_uniform([2, 3, 4, 5])
    for op in [
        math_ops.reduce_sum,
        math_ops.reduce_prod,
        math_ops.reduce_max,
        math_ops.reduce_min,
        math_ops.reduce_mean,
    ]:
      for axis in ([1], None, [0, 2]):
        for keepdims in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            a = array_ops.gather(x, i)
            return op(a, axis=axis, keepdims=keepdims)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_boolean_reduction(self):
    x = random_ops.random_uniform([2, 3, 4, 5]) > 0.5
    for op in [math_ops.reduce_any, math_ops.reduce_all]:
      for axis in ([1], None, [0, 2]):
        for keepdims in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            a = array_ops.gather(x, i)
            return op(a, axis=axis, keepdims=keepdims)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_argmin_argmax(self):
    x = random_ops.random_uniform([2, 3, 4, 5])
    for op in [math_ops.argmin, math_ops.argmax]:
      for axis in (1, None, -1):
        for output_dtype in (dtypes.int32, dtypes.int64, None):
          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            a = array_ops.gather(x, i)
            return op(a, axis=axis, output_type=output_dtype)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_bucketize(self):
    x = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      a = array_ops.gather(x, i)
      return math_ops.bucketize(a, [-1, 0.5, 1])

    self._test_loop_fn(loop_fn, 2)

  def test_clip_by_value(self):
    x = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      a = array_ops.gather(x, i)
      return clip_ops.clip_by_value(a, 0.5, 1.0)

    self._test_loop_fn(loop_fn, 2)

  def test_cum_sum(self):
    x = random_ops.random_uniform([2, 3, 4, 5])
    for axis in (1, -2):
      for exclusive in (True, False):
        for reverse in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            a = array_ops.gather(x, i)
            return math_ops.cumsum(
                a, axis=axis, exclusive=exclusive, reverse=reverse)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_cum_prod(self):
    x = random_ops.random_uniform([2, 3, 4, 5])
    for axis in (1, -2):
      for exclusive in (True, False):
        for reverse in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            a = array_ops.gather(x, i)
            return math_ops.cumprod(
                a, axis=axis, exclusive=exclusive, reverse=reverse)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_bias_add(self):
    for data_format in ("NCHW", "NHWC"):
      for stacked_value in (True, False):
        x_shape = [3, 4, 5, 6]
        if stacked_value:
          x_shape = [2] + x_shape
        x = random_ops.random_uniform(x_shape)
        for stacked_bias in (True, False):
          if not (stacked_value or stacked_bias):
            continue
          with backprop.GradientTape(persistent=True) as g:
            bias_dim = -1
            if data_format == "NCHW":
              bias_dim = 2 if stacked_value else 1
            bias_shape = [x_shape[bias_dim]]
            if stacked_bias:
              bias_shape = [2] + bias_shape
            bias = random_ops.random_uniform(bias_shape)
            g.watch(bias)

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            with g:
              a = array_ops.gather(x, i) if stacked_value else x
              b = array_ops.gather(bias, i) if stacked_bias else bias
              y = nn.bias_add(a, b, data_format=data_format)
              loss = math_ops.reduce_sum(y * y)
            grad = g.gradient(loss, bias)
            if stacked_bias:
              # If we gather over bias in loop_fn, the gradient will be an
              # instance of `IndexedSlices` with attrs `values` and `indices`.
              return y, grad.values, grad.indices
            else:
              return y, grad

          # pylint: enable=cell-var-from-loop

          out_dtypes = [dtypes.float32, dtypes.float32]
          if stacked_bias:
            out_dtypes = out_dtypes + [dtypes.int32]
          self._test_loop_fn(loop_fn, 2)

  @parameterized.parameters(
      (math_ops.unsorted_segment_sum,), (math_ops.unsorted_segment_min,),
      (math_ops.unsorted_segment_max,), (math_ops.unsorted_segment_prod,))
  def test_unsorted_segment_reduction(self, reduction_op):
    t = random_ops.random_uniform([3, 3, 2])
    for segment_ids_dtype in (dtypes.int32, dtypes.int64):
      for num_segments_dtype in (dtypes.int32, dtypes.int64):
        segment_ids = constant_op.constant([[0, 0, 2], [0, 1, 2], [2, 2, 2]],
                                           dtype=segment_ids_dtype)
        num_segments = constant_op.constant(3, dtype=num_segments_dtype)

        # pylint: disable=cell-var-from-loop
        def loop_fn(i):
          data = array_ops.gather(t, i)
          data_0 = array_ops.gather(t, 0)
          seg_ids = array_ops.gather(segment_ids, i)
          seg_ids_0 = array_ops.gather(segment_ids, 0)
          return (reduction_op(data, seg_ids, num_segments),
                  reduction_op(data_0, seg_ids, num_segments),
                  reduction_op(data, seg_ids_0, num_segments))

        # pylint: enable=cell-var-from-loop

        self._test_loop_fn(loop_fn, 3)

  @parameterized.parameters((math_ops.sparse_segment_sum_v2, True),
                            (math_ops.sparse_segment_mean_v2, True),
                            (math_ops.sparse_segment_sqrt_n_v2, True),
                            (math_ops.sparse_segment_sum_v2, False),
                            (math_ops.sparse_segment_mean_v2, False),
                            (math_ops.sparse_segment_sqrt_n_v2, False))
  def test_sparse_segment(self, op_func, with_num_segments):
    data = random_ops.random_uniform([3, 4, 2])
    indices = constant_op.constant([[1, 2, 3], [0, 1, 2], [0, 2, 3]])
    seg_ids = constant_op.constant([[0, 0, 2], [1, 1, 1], [0, 1, 1]])
    if with_num_segments:
      num_segments = 3
    else:
      num_segments = None

    def loop_fn(i):
      data_i = array_ops.gather(data, i)
      data_0 = array_ops.gather(data, 0)
      indices_i = array_ops.gather(indices, i)
      indices_0 = array_ops.gather(indices, 0)
      seg_ids_i = array_ops.gather(seg_ids, i)
      seg_ids_0 = array_ops.gather(seg_ids, 0)
      outputs = [
          op_func(data_0, indices_i, seg_ids_0, num_segments=num_segments),
          op_func(data_i, indices_i, seg_ids_0, num_segments=num_segments),
          op_func(data_0, indices_0, seg_ids_0, num_segments=num_segments),
          op_func(data_i, indices_0, seg_ids_0, num_segments=num_segments)
      ]
      if with_num_segments:
        # For this case, we support loop variant segment_ids as well.
        outputs += [
            op_func(data_0, indices_i, seg_ids_i, num_segments=num_segments),
            op_func(data_i, indices_i, seg_ids_i, num_segments=num_segments),
            op_func(data_0, indices_0, seg_ids_i, num_segments=num_segments),
            op_func(data_i, indices_0, seg_ids_i, num_segments=num_segments)
        ]
      return outputs

    self._test_loop_fn(loop_fn, 3)

  @parameterized.parameters(math_ops.sparse_segment_mean_grad,
                            math_ops.sparse_segment_sqrt_n_grad)
  def test_sparse_segment_grad(self, op_func):
    grad = random_ops.random_uniform([3, 3, 2])
    indices = constant_op.constant([1, 2, 3])
    seg_ids = constant_op.constant([0, 0, 2])
    dim0 = 4

    def loop_fn(i):
      grad_i = array_ops.gather(grad, i)
      return op_func(grad_i, indices, seg_ids, dim0)

    self._test_loop_fn(loop_fn, 3)

  def test_cast(self):
    x = constant_op.constant([[1], [2]])
    y = constant_op.constant([[1.0], [2.0]])

    def loop_fn(i):
      return (math_ops.cast(array_ops.gather(x, i), dtypes.float32),
              math_ops.cast(array_ops.gather(y, i), dtypes.int32))

    self._test_loop_fn(loop_fn, 2)

  def test_tanh_axpy(self):
    a = constant_op.constant(3.)
    x = random_ops.random_uniform([4, 5])
    y = random_ops.random_uniform([6, 5])
    n = x.shape[0]

    def loop_fn(i):
      return math_ops.tanh(a * array_ops.gather(x, i) + array_ops.gather(y, i))

    self._test_loop_fn(loop_fn, n)

  def test_select(self):
    a = random_ops.random_uniform([2, 3, 5])
    b = random_ops.random_uniform([2, 3, 5])
    for cond_shape in [2], [2, 3], [2, 3, 5]:
      cond = random_ops.random_uniform(cond_shape) > 0.5

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        a_i = array_ops.gather(a, i)
        b_i = array_ops.gather(b, i)
        cond_i = array_ops.gather(cond, i)
        return array_ops.where(cond_i, a_i, b_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)

  def test_selectv2_cond_needs_broadcast(self):
    a = random_ops.random_uniform([2, 3, 5])
    b = random_ops.random_uniform([2, 3, 5])
    # wherev2 assumes all shapes are broadcastable with each other.
    # This means that we can only specify conditions that are
    # broadcastable with [3, 5].
    for cond_shape in [2], [2, 1], [2, 5], [2, 3, 1], [2, 3, 5]:
      cond = random_ops.random_uniform(cond_shape) > 0.5

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        a_i = array_ops.gather(a, i)
        b_i = array_ops.gather(b, i)
        cond_i = array_ops.gather(cond, i)
        return array_ops.where_v2(cond_i, a_i, b_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)

  def test_selectv2_args_need_broadcast(self):
    a = random_ops.random_uniform([2, 5])
    b = random_ops.random_uniform([2, 3, 5])
    # wherev2 assumes all shapes are broadcastable with each other.
    # This means that we can only specify conditions that are
    # broadcastable with [3, 5].
    for cond_shape in [2], [2, 1], [2, 5], [2, 3, 1], [2, 3, 5]:
      cond = random_ops.random_uniform(cond_shape) > 0.5

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        a_i = array_ops.gather(a, i)
        b_i = array_ops.gather(b, i)
        cond_i = array_ops.gather(cond, i)
        return array_ops.where_v2(cond_i, a_i, b_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)

  def test_selectv2_cond_fixed(self):
    cond = random_ops.random_uniform([3, 5]) > 0.5
    b = random_ops.random_uniform([2, 3, 5])
    # wherev2 assumes all shapes are broadcastable with each other.
    # This means that we can only specify conditions that are
    # broadcastable with [3, 5].
    for a_shape in [2], [2, 1], [2, 5], [2, 3, 1], [2, 3, 5]:
      a = random_ops.random_uniform(a_shape)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        a_i = array_ops.gather(a, i)
        b_i = array_ops.gather(b, i)
        return array_ops.where_v2(cond, a_i, b_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)


@test_util.run_all_in_graph_and_eager_modes
class LinalgTest(PForTestCase):

  def test_cholesky(self):
    z = random_ops.random_normal([2, 3, 3])
    x = (
        math_ops.matmul(z, array_ops.matrix_transpose(z))  # Ensure pos. def.
        + linalg_ops.eye(3))  # Ensure well-conditioned.

    def loop_fn(i):
      return linalg_ops.cholesky(array_ops.gather(x, i))

    self._test_loop_fn(loop_fn, 2)

  def test_log_matrix_determinant(self):
    for x_shape in ([3, 4, 2, 2], [3, 2, 2]):
      x = random_ops.random_normal(x_shape)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        return linalg_ops.log_matrix_determinant(array_ops.gather(x, i))

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_matrix_inverse(self):
    x = (random_ops.random_uniform([3, 4, 2, 2]) +
         10 * linalg_ops.eye(2))  # Ensure well-conditioned.

    for adjoint in (True, False):

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        return linalg_ops.matrix_inverse(array_ops.gather(x, i),
                                         adjoint=adjoint)

      # pylint: enable=cell-var-from-loop
      self._test_loop_fn(loop_fn, 2)

  def test_matrix_solve(self):
    for adjoint in (True, False):
      for stack_a in (True, False):
        for stack_b in (True, False):
          shape_a = (2, 4, 3, 3) if stack_a else (4, 3, 3)
          shape_b = (2, 4, 3, 5) if stack_b else (4, 3, 5)
          x = (random_ops.random_uniform(shape_a) +
               10 * linalg_ops.eye(3))  # Ensure well-conditioned.
          y = random_ops.random_uniform(shape_b)

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            a = array_ops.gather(x, i) if stack_a else x
            b = array_ops.gather(y, i) if stack_b else y
            return linalg_ops.matrix_solve(a, b, adjoint=adjoint)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_matrix_triangular_solve(self):
    for lower in (True, False):
      for adjoint in (True, False):
        for stack_a in (True, False):
          for stack_b in (True, False):
            shape_a = (2, 4, 3, 3) if stack_a else (4, 3, 3)
            shape_b = (2, 4, 3, 5) if stack_b else (4, 3, 5)
            x = array_ops.matrix_band_part(
                random_ops.random_uniform(shape_a) +
                linalg_ops.eye(3),  # Ensure well-conditioned.
                *((-1, 0) if lower else (0, -1)))  # Ensure triangular.
            y = random_ops.random_uniform(shape_b)

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
              a = array_ops.gather(x, i) if stack_a else x
              b = array_ops.gather(y, i) if stack_b else y
              return linalg_ops.matrix_triangular_solve(
                  a, b, lower=lower, adjoint=adjoint)

            # pylint: enable=cell-var-from-loop

            self._test_loop_fn(loop_fn, 2)

  def test_self_adjoint_eig(self):
    z = random_ops.random_normal([2, 3, 3])
    x = z + array_ops.matrix_transpose(z)  # Ensure self-adjoint.

    def loop_fn(i):
      return (linalg_ops.self_adjoint_eig(array_ops.gather(x, i)),
              linalg_ops.self_adjoint_eigvals(array_ops.gather(x, i)))

    self._test_loop_fn(loop_fn, 2)

  def test_einsum(self):
    b = 10
    x_series = random_ops.random_uniform([b, 9, 9])
    y_series = random_ops.random_uniform([b, 9, 1])

    def loop_fn(i):
      x = array_ops.gather(x_series, 0)  # invariant.
      y = array_ops.gather(y_series, 0)  # invariant.
      x_i = array_ops.gather(x_series, i)
      y_i = array_ops.gather(y_series, i)
      z1 = special_math_ops.einsum("ab,bc->ac", x_i, y)
      z2 = special_math_ops.einsum("ab,bc->ac", x, y_i)
      z3 = special_math_ops.einsum("ab,bc->ac", x, y)
      z4 = special_math_ops.einsum("ab,bc->ac", x_i, y_i)
      z5 = special_math_ops.einsum("cd,ce->de", y_i, x_i)  # Includes transpose.
      outputs = [z1, z2, z3, z4, z5]
      return outputs

    self._test_loop_fn(loop_fn, b)


if __name__ == "__main__":
  test.main()
