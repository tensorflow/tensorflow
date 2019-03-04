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

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class MathTest(PForTestCase):

  def test_unary_cwise_ops(self):
    complex_ops = [
        math_ops.angle,
        math_ops.imag,
        math_ops.complex_abs,
        math_ops.real,
        math_ops.conj,
    ]
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
        math_ops.exp,
        math_ops.expm1,
        math_ops.inv,
        math_ops.is_finite,
        math_ops.is_inf,
        math_ops.lgamma,
        math_ops.log,
        math_ops.log1p,
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
        math_ops.tanh,
        nn.elu,
        nn.relu,
        nn.relu6,
        nn.selu,
        nn.softplus,
        nn.softsign,
    ]
    for op in complex_ops + real_ops:
      with backprop.GradientTape(persistent=True) as g:
        x = random_ops.random_uniform([3, 5])
        g.watch(x)
        if op in complex_ops:
          y = random_ops.random_uniform([3, 5])
          g.watch(y)
          x = math_ops.complex(x, y)

      # pylint: disable=cell-var-from-loop
      output_dtypes = []
      def loop_fn(i):
        with g:
          x1 = array_ops.gather(x, i)
          y1 = op(x1)
          outputs = [op(x), y1]
          if y1.dtype == dtypes.float32:
            loss = math_ops.reduce_sum(y1 * y1)
          else:
            loss = None
        if loss is not None:
          grad = g.gradient(loss, x1)
          if grad is not None:
            outputs.append(grad)
        del output_dtypes[:]
        output_dtypes.extend([t.dtype for t in outputs])
        return outputs

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=output_dtypes)

  def test_unary_cwise_no_grad(self):
    for op in [math_ops.ceil,
               math_ops.floor,
               math_ops.logical_not]:
      x = random_ops.random_uniform([3, 5])
      if op == math_ops.logical_not:
        x = x > 0

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        return op(array_ops.gather(x, i))

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=x.dtype)

  def test_binary_cwise_ops(self):
    logical_ops = [
        math_ops.logical_and,
        math_ops.logical_or,
        math_ops.logical_xor
    ]

    # Wrapper functions restricting the range of inputs of zeta and polygamma.
    def safe_polygamma(x, y):
      return math_ops.polygamma(
          math_ops.round(clip_ops.clip_by_value(y, 1, 10)),
          x * x + 1)

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
        output_dtypes.extend([t.dtype for t in outputs])
        return outputs
      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=output_dtypes)

  def test_approximate_equal(self):
    x = random_ops.random_uniform([3, 5])
    y = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      y1 = array_ops.gather(y, i)
      return math_ops.approximate_equal(x1, y1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.bool])

  def test_addn(self):
    x = random_ops.random_uniform([2, 3, 5])
    y = random_ops.random_uniform([3, 5])
    z = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return math_ops.add_n([x1, y, z])

    self._test_loop_fn(loop_fn, 2)

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

  def test_reduction(self):
    x = random_ops.random_uniform([2, 3, 4, 5])
    for op in [
        math_ops.reduce_sum, math_ops.reduce_prod, math_ops.reduce_max,
        math_ops.reduce_min, math_ops.reduce_mean,
    ]:
      for axis in ([1], None, [0, 2]):
        for keepdims in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            a = array_ops.gather(x, i)
            return op(a, axis=axis, keepdims=keepdims)

          # pylint: enable=cell-var-from-loop

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
          self._test_loop_fn(
              loop_fn, 2, loop_fn_dtypes=out_dtypes)

  def test_unsorted_segment_sum(self):
    t = random_ops.random_uniform([3, 3, 2])
    segment_ids = constant_op.constant([[0, 0, 2], [0, 1, 2], [2, 2, 2]])
    num_segments = 3

    def loop_fn(i):
      data = array_ops.gather(t, i)
      data_0 = array_ops.gather(t, 0)
      seg_ids = array_ops.gather(segment_ids, i)
      return (math_ops.unsorted_segment_sum(data, seg_ids, num_segments),
              math_ops.unsorted_segment_sum(data_0, seg_ids, num_segments))

    self._test_loop_fn(loop_fn, 3, [dtypes.float32] * 2)

  def test_cast(self):
    x = constant_op.constant([[1], [2]])
    y = constant_op.constant([[1.0], [2.0]])

    def loop_fn(i):
      return (math_ops.cast(array_ops.gather(x, i), dtypes.float32),
              math_ops.cast(array_ops.gather(y, i), dtypes.int32))

    self._test_loop_fn(
        loop_fn, 2, loop_fn_dtypes=[dtypes.float32, dtypes.int32])

  def test_tanh_axpy(self):
    a = constant_op.constant(3.)
    x = random_ops.random_uniform([4, 5])
    y = random_ops.random_uniform([6, 5])
    n = x.shape[0]

    def loop_fn(i):
      return math_ops.tanh(a * array_ops.gather(x, i) + array_ops.gather(y, i))

    self._test_loop_fn(loop_fn, n)

  def test_select(self):
    cond = constant_op.constant([True, False])
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


if __name__ == "__main__":
  test.main()
