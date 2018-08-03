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
"""Tests for pfor and for_loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import flags
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class PForTest(test.TestCase):

  def _run_targets(self, targets1, targets2=None, run_init=True):
    targets1 = nest.flatten(targets1)
    targets2 = ([] if targets2 is None else nest.flatten(targets2))
    assert len(targets1) == len(targets2) or not targets2
    if run_init:
      init = variables.global_variables_initializer()
      self.evaluate(init)
    return self.evaluate(targets1 + targets2)

  def run_and_assert_equal(self, targets1, targets2):
    outputs = self._run_targets(targets1, targets2)
    outputs = nest.flatten(outputs)  # flatten SparseTensorValues
    n = len(outputs) // 2
    for i in range(n):
      if outputs[i + n].dtype != np.object:
        self.assertAllClose(outputs[i + n], outputs[i], rtol=1e-4, atol=1e-5)
      else:
        self.assertAllEqual(outputs[i + n], outputs[i])

  def _test_loop_fn(self, loop_fn, iters, loop_fn_dtypes=dtypes.float32):
    t1 = pfor_control_flow_ops.pfor(loop_fn, iters=iters)
    t2 = pfor_control_flow_ops.for_loop(loop_fn, loop_fn_dtypes, iters=iters)
    self.run_and_assert_equal(t1, t2)

  def test_op_conversion_fallback_to_while_loop(self):
    # Note that we used top_k op for this test. If a converter gets defined for
    # it, we will need to find another op for which a converter has not been
    # defined.
    x = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return nn.top_k(x_i)

    with self.assertRaisesRegexp(ValueError, "No converter defined"):
      self._test_loop_fn(
          loop_fn, 3, loop_fn_dtypes=[dtypes.float32, dtypes.int32])
    flags.FLAGS.op_conversion_fallback_to_while_loop = True
    self._test_loop_fn(
        loop_fn, 3, loop_fn_dtypes=[dtypes.float32, dtypes.int32])
    flags.FLAGS.op_conversion_fallback_to_while_loop = False


class ArrayTest(PForTest):

  def test_gather(self):
    x = random_ops.random_uniform([3, 3, 3])

    def loop_fn(i):
      outputs = []
      x_i = array_ops.gather(x, i)
      for y in [x, x_i]:
        axes = [0, 2, -1] if y == x else [0]
        for axis in axes:
          outputs.append(array_ops.gather(y, 2, axis=axis))
          outputs.append(array_ops.gather(y, i, axis=axis))
          outputs.append(array_ops.gather(y, [i], axis=axis))
          outputs.append(array_ops.gather(y, [i, 2], axis=axis))
          outputs.append(array_ops.gather(y, [[2, i], [i, 1]], axis=axis))
      return outputs

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 20)

  def test_shape(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.shape(x_i), array_ops.shape(x_i, out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32, dtypes.int64])

  def test_size(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.size(x_i), array_ops.size(x_i, out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32, dtypes.int64])

  def test_rank(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.rank(x_i)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32])

  def test_shape_n(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      return array_ops.shape_n([x_i, x, y, y_i]), array_ops.shape_n(
          [x_i, x, y, y_i], out_type=dtypes.int64)

    self._test_loop_fn(
        loop_fn, 3, loop_fn_dtypes=[dtypes.int32] * 4 + [dtypes.int64] * 4)

  def test_reshape(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.reshape(x1, [-1]), array_ops.reshape(x1, [1, 3, 1, -1])

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_expand_dims(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.expand_dims(
          x1, axis=-1), array_ops.expand_dims(
              x1, axis=1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_slice(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.slice(x1, begin=(0, 1), size=(2, 1))

    self._test_loop_fn(loop_fn, 3)

  def test_tile(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.tile(x1, [2, 1])

    self._test_loop_fn(loop_fn, 3)

  def test_tile_loop_dependent(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.tile(x1, [i, 1])

    with self.assertRaisesRegexp(ValueError, "expected to be loop invariant"):
      pfor_control_flow_ops.pfor(loop_fn, 2)

  def test_pack(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.stack([x1, y], axis=-1)

    self._test_loop_fn(loop_fn, 1)

  def test_unpack(self):
    x = random_ops.random_uniform([3, 2, 3, 4])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.unstack(
          x_i, 4, axis=-1), array_ops.unstack(
              x_i, 3, axis=1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 7)

  def test_pad(self):
    x = random_ops.random_uniform([3, 2, 3])
    padding = constant_op.constant([[1, 2], [3, 4]])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.pad(x1, padding, mode="CONSTANT")

    self._test_loop_fn(loop_fn, 3)

  def test_split(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.split(x1, 2, axis=0), array_ops.split(x1, 3, axis=-1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 5)

  def test_transpose(self):
    x = random_ops.random_uniform([3, 2, 3, 4])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.transpose(x1, [2, 1, 0])

    self._test_loop_fn(loop_fn, 3)

  def test_zeros_like(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      z = array_ops.zeros_like(x1),
      return z, z + x1

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_concat_v2(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.concat(
          [x1, x1, y], axis=0), array_ops.concat(
              [x1, x1, y], axis=-1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_unary_cwise_ops(self):
    for op in [array_ops.identity, array_ops.stop_gradient]:
      x = random_ops.random_uniform([3, 5])

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        x1 = array_ops.gather(x, i)
        y = op(x1) + x1
        loss = nn.l2_loss(y)
        return op(x), y, gradient_ops.gradients(loss, x1)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 3)

  def test_strided_slice(self):
    x = random_ops.random_uniform([3, 3, 4, 4, 2, 2, 2])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      y = x_i[:2, ::2, 1::3, ..., array_ops.newaxis, 1]
      loss = nn.l2_loss(y)
      return y, gradient_ops.gradients(loss, x_i)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)


class MathTest(PForTest):

  def test_unary_cwise_ops(self):
    for op in [
        math_ops.tanh, nn.relu, math_ops.sigmoid, math_ops.negative,
        math_ops.square
    ]:
      x = random_ops.random_uniform([3, 5])

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        x1 = array_ops.gather(x, i)
        y = op(x1)
        loss = math_ops.reduce_sum(y * y)
        return op(x), y, gradient_ops.gradients(loss, x1)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 3)

  def test_unary_cwise_no_grad(self):
    for op in [math_ops.ceil, math_ops.floor, math_ops.logical_not]:
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
        math_ops.logical_and, math_ops.logical_or, math_ops.logical_xor
    ]
    bool_ops = [
        math_ops.less, math_ops.less_equal, math_ops.greater,
        math_ops.greater_equal, math_ops.equal, math_ops.not_equal
    ]
    float_ops = [
        math_ops.add, math_ops.subtract, math_ops.multiply, math_ops.divide,
        math_ops.maximum, math_ops.minimum
    ]
    for op in logical_ops + bool_ops + float_ops:
      x = random_ops.random_uniform([7, 3, 5])
      y = random_ops.random_uniform([3, 5])
      if op in logical_ops:
        x = x > 0
        y = y > 0

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        x1 = array_ops.gather(x, i)
        y1 = array_ops.gather(y, i)
        return op(x, y), op(x1, y), op(x, y1), op(x1, y1), op(x1, x1)

      # pylint: enable=cell-var-from-loop

      dtype = dtypes.float32 if op in float_ops else dtypes.bool
      self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtype] * 5)

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
        math_ops.reduce_min
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
    x_shape = [2, 3, 4, 5, 6]
    x = random_ops.random_uniform(x_shape)
    for data_format in ("NCHW", "NHWC"):
      bias_dim = 2 if data_format == "NCHW" else -1
      bias_shape = x_shape[bias_dim]
      bias = random_ops.random_uniform([bias_shape])

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        a = array_ops.gather(x, i)
        y = nn.bias_add(a, bias, data_format=data_format)
        loss = math_ops.reduce_sum(y * y)
        return y, gradient_ops.gradients(loss, bias)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(
          loop_fn, 2, loop_fn_dtypes=[dtypes.float32, dtypes.float32])

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


class NNTest(PForTest):

  def test_conv2d(self):
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    filt = random_ops.random_uniform([3, 3, 3, 7])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return nn.conv2d(
          x1, filt, strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_conv2d_backprop_input(self):
    x_shape = [2, 12, 12, 3]
    filt = random_ops.random_uniform([3, 3, 3, 7])
    grad = random_ops.random_uniform([3, 2, 5, 5, 7])

    def loop_fn(i):
      grad1 = array_ops.gather(grad, i)
      return nn.conv2d_backprop_input(
          x_shape,
          filt,
          grad1,
          strides=[1, 2, 2, 1],
          padding="VALID",
          data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_conv2d_backprop_filter(self):
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    x_0 = array_ops.gather(x, 0)
    filter_sizes = [3, 3, 3, 7]
    grad = random_ops.random_uniform([3, 2, 5, 5, 7])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      grad_i = array_ops.gather(grad, i)
      return [
          nn.conv2d_backprop_filter(
              inp,
              filter_sizes,
              grad_i,
              strides=[1, 2, 2, 1],
              padding="VALID",
              data_format="NHWC") for inp in [x_i, x_0]
      ]

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_avg_pool(self):
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    ksize = [1, 3, 3, 1]

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      output = nn.avg_pool(
          x1, ksize, strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")
      loss = nn.l2_loss(output)
      return output, gradient_ops.gradients(loss, x1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_max_pool(self):
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    ksize = [1, 3, 3, 1]

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      output = nn.max_pool(
          x1, ksize, strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")
      loss = nn.l2_loss(output)
      return output, gradient_ops.gradients(loss, x1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_fused_batch_norm(self):
    data_formats = ["NHWC"]
    if test.is_gpu_available():
      data_formats.append("NCHW")
    for is_training in (True, False):
      for data_format in data_formats:
        if data_format == "NCHW":
          x = random_ops.random_uniform([3, 1, 2, 5, 5])
        else:
          x = random_ops.random_uniform([3, 1, 5, 5, 2])
        scale = random_ops.random_uniform([2])
        offset = random_ops.random_uniform([2])
        mean = None if is_training else random_ops.random_uniform([2])
        variance = None if is_training else random_ops.random_uniform([2])

        # pylint: disable=cell-var-from-loop
        def loop_fn(i):
          x1 = array_ops.gather(x, i)
          outputs = nn.fused_batch_norm(
              x1,
              scale,
              offset,
              mean=mean,
              variance=variance,
              epsilon=0.01,
              data_format=data_format,
              is_training=is_training)
          outputs = list(outputs)
          # We only test the first value of outputs when is_training is False.
          # It looks like CPU and GPU have different outputs for batch_mean and
          # batch_variance for this case.
          if not is_training:
            outputs[1] = constant_op.constant(0.)
            outputs[2] = constant_op.constant(0.)
          loss = nn.l2_loss(outputs[0])
          gradients = gradient_ops.gradients(loss, [x1, scale, offset])
          return outputs + gradients

        # pylint: enable=cell-var-from-loop

        self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 6)

  def test_softmax_cross_entropy_with_logits(self):
    logits = random_ops.random_uniform([3, 2, 4])
    labels = random_ops.random_uniform([3, 2, 4])
    labels /= math_ops.reduce_sum(labels, axis=[2], keepdims=True)

    def loop_fn(i):
      logits_i = array_ops.gather(logits, i)
      labels_i = array_ops.gather(labels, i)
      loss = nn.softmax_cross_entropy_with_logits(
          labels=labels_i, logits=logits_i)
      return loss, gradient_ops.gradients(math_ops.reduce_sum(loss), logits_i)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)


class RandomTest(PForTest):

  # The random values generated in the two implementations are not guaranteed to
  # match. So we only check the returned shapes.
  def run_and_assert_equal(self, targets1, targets2):
    outputs = self._run_targets(targets1, targets2)
    n = len(outputs) // 2
    for i in range(n):
      self.assertAllEqual(outputs[i].shape, outputs[i + n].shape)

  def test_random_uniform(self):

    def loop_fn(_):
      return random_ops.random_uniform([3])

    self._test_loop_fn(loop_fn, 5)

  def test_random_uniform_int(self):

    def loop_fn(_):
      return random_ops.random_uniform([3], maxval=1, dtype=dtypes.int32)

    self._test_loop_fn(loop_fn, 5, loop_fn_dtypes=dtypes.int32)

  def test_random_standard_normal(self):

    def loop_fn(_):
      return random_ops.random_normal([3])

    self._test_loop_fn(loop_fn, 5)

  def test_truncated_normal(self):

    def loop_fn(_):
      return random_ops.truncated_normal([3])

    self._test_loop_fn(loop_fn, 5)

  def test_random_gamma(self):

    def loop_fn(_):
      return random_ops.random_gamma([3], alpha=[0.5])

    self._test_loop_fn(loop_fn, 5)

  def test_random_poisson_v2(self):

    def loop_fn(_):
      return random_ops.random_poisson(lam=[1.3], shape=[3])

    self._test_loop_fn(loop_fn, 5)


class LoggingTest(PForTest):

  def test_print(self):
    x = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return logging_ops.Print(
          x1, [x1, "x1", array_ops.shape(x1)], summarize=10)

    self._test_loop_fn(loop_fn, 3)

  def test_assert(self):

    def loop_fn(i):
      return control_flow_ops.Assert(i < 10, [i, [10], [i + 1]])

    # TODO(agarwal): make this work with for_loop.
    with session.Session() as sess:
      sess.run(pfor_control_flow_ops.pfor(loop_fn, 3))


class TensorArrayTest(PForTest):

  def test_create_outside_and_read(self):

    ta = tensor_array_ops.TensorArray(
        dtypes.int32, 2, clear_after_read=False).write(0, 0).write(1, 1)

    def loop_fn(i):
      return ta.read(i), ta.read(0)

    self._test_loop_fn(loop_fn, 2, [dtypes.int32] * 2)

  def test_create_outside_and_gather(self):

    ta = tensor_array_ops.TensorArray(
        dtypes.int32, 2, clear_after_read=False).write(0, 0).write(1, 1)

    def loop_fn(i):
      return ta.gather([i]), ta.gather([0, 1])

    self._test_loop_fn(loop_fn, 2, [dtypes.int32] * 2)

  def test_create_outside_and_write_and_scatter(self):

    t = tensor_array_ops.TensorArray(dtypes.int32, 10, clear_after_read=False)
    handle = t.handle

    def loop_fn(i):
      ta = t.write(i + 2, 2 * i).write(i, 5)
      ta = ta.scatter([4 + i], [4]).scatter([6 + i, 8 + i], [6 + i, 8 + i])
      return ta.flow

    t1 = pfor_control_flow_ops.pfor(loop_fn, iters=2)
    out1 = tensor_array_ops.TensorArray(
        dtypes.int32, handle=handle, flow=t1[-1]).stack()
    output1 = self._run_targets(out1)

    t2 = pfor_control_flow_ops.for_loop(loop_fn, dtypes.float32, iters=2)
    out2 = tensor_array_ops.TensorArray(
        dtypes.int32, handle=handle, flow=t2[-1]).stack()
    output2 = self._run_targets(out2)
    self.assertAllClose(output2, output1)

  def test_create_inside_and_write(self):

    def loop_fn(i):
      # TODO(agarwal): switching the order of writes to ta1 does not work.
      ta1 = tensor_array_ops.TensorArray(dtypes.int32, 2).write(0, i).write(
          1, 1)
      ta2 = tensor_array_ops.TensorArray(dtypes.int32, 1).write(0, 1)
      return ta1.stack(), ta2.stack()

    self._test_loop_fn(loop_fn, 3, [dtypes.int32] * 2)

  def test_create_inside_and_scatter(self):

    def loop_fn(i):
      # TODO(agarwal): switching the order of scatter to ta1 does not work.
      ta1 = tensor_array_ops.TensorArray(dtypes.int32, 2).scatter(
          [0], [[i, 2]]).scatter([1], [[1, 2]])
      ta2 = tensor_array_ops.TensorArray(dtypes.int32,
                                         2).scatter([0], [3]).scatter([1], [4])
      return ta1.stack(), ta2.stack()

    self._test_loop_fn(loop_fn, 3, [dtypes.int32] * 2)

  def test_create_inside_and_read(self):

    def loop_fn(i):
      ta1 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, i).write(1, 1)
      ta2 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, 1).write(1, 2)
      # TODO(agarwal): ta1.read(i) currently is not supported.
      return ta1.read(0), ta2.read(0), ta2.read(i)

    self._test_loop_fn(loop_fn, 2, [dtypes.int32] * 3)

  def test_create_inside_and_gather(self):

    def loop_fn(i):
      ta1 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, i).write(1, 1)
      ta2 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, 1).write(1, 2)
      # TODO(agarwal): ta1.read(i) currently is not supported.
      return ta1.gather([0, 1]), ta2.gather([0, 1]), ta2.gather([i])

    self._test_loop_fn(loop_fn, 2, [dtypes.int32] * 3)

  def test_grad(self):
    x = random_ops.random_uniform([3, 2])
    ta = tensor_array_ops.TensorArray(
        dtypes.float32, 3, clear_after_read=False).unstack(x)
    y = math_ops.square(ta.stack())

    def loop_fn(i):
      y_i = array_ops.gather(y, i)
      grad = gradient_ops.gradients(y_i, x)[0]
      return array_ops.gather(grad, i)

    t1 = pfor_control_flow_ops.pfor(loop_fn, iters=3)
    # y = x * x. Hence dy/dx = 2 * x.
    actual_grad = 2.0 * x
    with session.Session() as sess:
      actual_grad, computed_grad = sess.run([t1, actual_grad])
      self.assertAllClose(actual_grad, computed_grad)


class StackTest(PForTest):

  def test_stack_inside_loop_invariant(self):

    def loop_fn(_):
      s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)
      op1 = data_flow_ops.stack_push_v2(s, 1)
      with ops.control_dependencies([op1]):
        op2 = data_flow_ops.stack_push_v2(s, 2)
      with ops.control_dependencies([op2]):
        e2 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      with ops.control_dependencies([e2]):
        e1 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      return e1, e2

    self._test_loop_fn(loop_fn, 2, [dtypes.int32] * 2)

  def test_stack_inside_push_loop_dependent(self):

    def loop_fn(i):
      s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)
      op1 = data_flow_ops.stack_push_v2(s, i)
      with ops.control_dependencies([op1]):
        op2 = data_flow_ops.stack_push_v2(s, 2)
      with ops.control_dependencies([op2]):
        e2 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      with ops.control_dependencies([e2]):
        e1 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      return e1, e2

    self._test_loop_fn(loop_fn, 2, [dtypes.int32] * 2)

  def test_stack_outside_pop(self):
    s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)
    op = data_flow_ops.stack_push_v2(s, 5)
    with ops.control_dependencies([op]):
      op = data_flow_ops.stack_push_v2(s, 6)
    with ops.control_dependencies([op]):
      op = data_flow_ops.stack_push_v2(s, 7)

    def loop_fn(_):
      e1 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      with ops.control_dependencies([e1]):
        e2 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      return e1, e2

    with ops.control_dependencies([op]):
      e1, e2 = pfor_control_flow_ops.pfor(loop_fn, iters=2)
    with ops.control_dependencies([e1, e2]):
      e3 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
    v1, v2, v3 = self._run_targets([e1, e2, e3], run_init=False)
    self.assertAllEqual([7, 7], v1)
    self.assertAllEqual([6, 6], v2)
    self.assertAllEqual(5, v3)

  def test_stack_outside_push(self):
    s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)

    def loop_fn(_):
      return data_flow_ops.stack_push_v2(s, 7)

    with self.assertRaisesRegexp(ValueError, "StackPushV2 not allowed.*"):
      pfor_control_flow_ops.pfor(loop_fn, iters=2)


# TODO(agarwal): test nested while_loops. This currently requires converting a
# tf.cond.
class ControlFlowTest(PForTest):

  def test_while_outside_loop(self):

    x = control_flow_ops.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    def loop_fn(i):
      return x + i

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32])

  def test_invariant_while(self):

    def loop_fn(_):
      return control_flow_ops.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32])

  def test_invariant_while_with_control_dependency(self):

    def loop_fn(i):
      with ops.control_dependencies([i]):
        return control_flow_ops.while_loop(lambda j: j < 4, lambda j: j + 1,
                                           [0])

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32])

  def test_while_with_stateful_ops(self):

    def loop_fn(_):
      return control_flow_ops.while_loop(
          lambda j, x: j < 4,
          lambda j, x: (j + 1, x + random_ops.random_uniform([])), [0, 0.])[0]

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32])

  def test_while_unstacked_condition(self):

    def loop_fn(i):
      return control_flow_ops.while_loop(lambda j, x: j < 4,
                                         lambda j, x: (j + 1, x + i), [0, 0])

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32, dtypes.int32])

  def test_while(self):
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      _, total = control_flow_ops.while_loop(
          lambda j, _: j < lengths_i,
          lambda j, t: (j + 1, t + array_ops.gather(x_i, j)), [0, 0.])
      return total

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32])

  def test_while_jacobian(self):
    x = random_ops.random_uniform([1, 3])
    y = random_ops.random_uniform([3, 3])

    # out = x @ y @ y @ y @ y, where @ is matmul operator.
    _, out = control_flow_ops.while_loop(
        lambda i, _: i < 4, lambda i, out: (i + 1, math_ops.matmul(out, y)),
        [0, x])

    def loop_fn(i):
      out_i = array_ops.gather(out, i, axis=1)
      return array_ops.reshape(gradient_ops.gradients(out_i, x)[0], [-1])

    out = pfor_control_flow_ops.pfor(loop_fn, iters=3)

    # The above code does not work with tf.while_loop instead of pfor. So we
    # manually compute the expected output here.
    # Note that gradient of output w.r.t is (y @ y @ y @ y)^T.
    expected_output = y
    for _ in range(3):
      expected_output = math_ops.matmul(expected_output, y)
    expected_output = array_ops.transpose(expected_output, [1, 0])

    with session.Session() as sess:
      out, expected = sess.run([out, expected_output])
      self.assertAllClose(expected, out)

  def test_tensor_array_as_loop_variable(self):

    def loop_fn(i):

      def body(j, ta):
        ta = ta.write(j, i + j * j)
        return j + 1, ta

      _, ta = control_flow_ops.while_loop(
          lambda j, _: j < 4, body,
          (0, tensor_array_ops.TensorArray(dtypes.int32, size=4)))
      return ta.stack()

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32])

  def test_read_tensor_array_partitioned_indices(self):
    # Note that tensor array values are pfor loop dependent, and the while loop
    # termination condition is also dependent on pfor iteration.
    def loop_fn(i):
      ta = tensor_array_ops.TensorArray(dtypes.int32, size=6)
      ta = ta.unstack(i + list(range(5)))

      def body(j, s):
        return j + 1, s + ta.read(j)

      _, s = control_flow_ops.while_loop(lambda j, _: j < i,
                                         body,
                                         (0, 0))
      return s

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32])

  def test_external_while_loop_grad(self):
    # Here we test that external while_loops that are extended from inside pfor
    # (due to gradient calls) are not actually converted. If the below was
    # converted all pfor iterations would write to the same tensor array
    # indices.
    x = constant_op.constant(1.)

    def body(j, ta):
      ta = ta.write(j, x)
      return j + 1, ta

    _, ta = control_flow_ops.while_loop(
        lambda j, _: j < 4, body,
        (0, tensor_array_ops.TensorArray(dtypes.float32, size=4)))
    out = ta.stack()

    def loop_fn(i):
      out_i = array_ops.gather(out, i)
      return gradient_ops.gradients(out_i, x)[0]

    with session.Session() as sess:
      # out is [x, x, x]. Hence the gradients should be [1, 1, 1].
      self.assertAllEqual([1, 1, 1],
                          sess.run(pfor_control_flow_ops.pfor(loop_fn, 3)))

  def test_tensor_array_grad(self):
    inp = constant_op.constant(np.random.rand(3, 4, 2), dtype=dtypes.float32)
    ta = tensor_array_ops.TensorArray(dtypes.float32, size=3)
    ta = ta.unstack(inp)

    def loop_fn(i):

      def body(j, x):
        value = ta.gather([j])
        value = array_ops.gather(array_ops.reshape(value, [4, 2]), i)
        return j + 1, x + value

      _, out = control_flow_ops.while_loop(lambda j, _: j < 3, body,
                                           (0, array_ops.zeros([2])))
      out = math_ops.reduce_prod(out)
      return out, gradient_ops.gradients(out, inp)[0]

    pfor_out, pfor_out_grad = pfor_control_flow_ops.pfor(loop_fn, 4)
    # Note that tf.while_loop does not work in the setup above. So we manually
    # construct the equivalent computation of the above loops here.
    real_out = math_ops.reduce_sum(inp, reduction_indices=[0])
    real_out = math_ops.reduce_prod(real_out, reduction_indices=[1])
    # Note that gradients of real_out will accumulate the gradients across the
    # output value. Hence we do the same aggregation on pfor_out_grad.
    real_out_grad = gradient_ops.gradients(real_out, inp)[0]
    sum_pfor_out_grad = math_ops.reduce_sum(
        pfor_out_grad, reduction_indices=[0])

    with session.Session() as sess:
      v1, v2, v1_grad, v2_grad = sess.run(
          [pfor_out, real_out, sum_pfor_out_grad, real_out_grad])
      self.assertAllClose(v1, v2)
      self.assertAllClose(v1_grad, v2_grad)


def dynamic_lstm_input_fn(batch_size, state_size, max_steps):
  # We make inputs and sequence_length constant so that multiple session.run
  # calls produce the same result.
  inputs = constant_op.constant(
      np.random.rand(batch_size, max_steps, state_size), dtype=dtypes.float32)
  sequence_length = np.random.randint(0, size=[batch_size], high=max_steps + 1)
  sequence_length = constant_op.constant(sequence_length, dtype=dtypes.int32)
  return inputs, sequence_length


def create_dynamic_lstm(cell_fn, batch_size, state_size, max_steps):
  cell = cell_fn(state_size)
  inputs, sequence_length = dynamic_lstm_input_fn(batch_size,
                                                  state_size,
                                                  max_steps)
  inputs_ta = tensor_array_ops.TensorArray(
      dtypes.float32, size=max_steps, element_shape=[batch_size, state_size])
  inputs_time_major = array_ops.transpose(inputs, [1, 0, 2])
  inputs_ta = inputs_ta.unstack(inputs_time_major)
  zeros = array_ops.zeros([state_size])

  def loop_fn(i):
    sequence_length_i = array_ops.gather(sequence_length, i)

    def body_fn(t, state, ta):
      inputs_t = array_ops.expand_dims(
          array_ops.gather(inputs_ta.read(t), i), 0)
      output, new_state = cell(inputs_t, state)
      output = array_ops.reshape(output, [-1])
      # TODO(agarwal): one optimization that dynamic_rnn uses is to avoid the
      # array_ops.where when t < min(sequence_length). Doing that requires
      # supporting tf.cond pfor conversion.
      done = t >= sequence_length_i
      output = array_ops.where(done, zeros, output)
      ta = ta.write(t, output)
      new_state = [array_ops.where(done, s, ns) for s, ns in
                   zip(nest.flatten(state), nest.flatten(new_state))]
      new_state = nest.pack_sequence_as(state, new_state)
      return t + 1, new_state, ta

    def condition_fn(t, _, unused):
      del unused
      return t < max_steps

    initial_state = cell.zero_state(1, dtypes.float32)
    _, state, ta = control_flow_ops.while_loop(condition_fn, body_fn, [
        0, initial_state,
        tensor_array_ops.TensorArray(dtypes.float32, max_steps)
    ])

    new_state = [array_ops.reshape(x, [-1]) for x in nest.flatten(state)]
    new_state = nest.pack_sequence_as(initial_state, new_state)
    return ta.stack(), new_state

  pfor_output = pfor_control_flow_ops.pfor(loop_fn, batch_size)
  tf_output = rnn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=sequence_length,
      initial_state=cell.zero_state(batch_size, dtypes.float32))
  return pfor_output, tf_output


class RNNTest(PForTest):

  def test_dynamic_rnn(self):
    pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicRNNCell,
                                                   3, 5, 7)
    self.run_and_assert_equal(pfor_outputs, tf_outputs)

  def test_dynamic_lstm(self):
    pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicLSTMCell,
                                                   3, 5, 7)
    self.run_and_assert_equal(pfor_outputs, tf_outputs)


# TODO(agarwal): benchmark numbers on GPU for graphs based on while_loop
# conversion don't look good. Some of it seems like lot of copies between host
# and device. Optimize that.
class Benchmarks(test.Benchmark):

  def _run(self, targets, iters, name=None):

    def _done(t):
      # Note that we don't use tf.control_dependencies since that will not make
      # sure that the computation on GPU has actually finished. So we fetch the
      # first element of the output, and assume that this will not be called on
      # empty tensors.
      return array_ops.gather(array_ops.reshape(t, [-1]), 0)

    targets = [_done(x) for x in nest.flatten(targets)]
    sess = session.Session()
    with sess:
      init = variables.global_variables_initializer()
      sess.run(init)
      sess.run(targets)
      begin = time.time()
      for _ in range(iters):
        sess.run(targets)
      end = time.time()
    avg_time_ms = 1000 * (end - begin) / iters
    self.report_benchmark(iters=iters, wall_time=avg_time_ms, name=name)
    return avg_time_ms

  def benchmark_basic_while(self):
    with ops.Graph().as_default():

      def loop_fn(i):
        _, s = control_flow_ops.while_loop(
            lambda t, x: t < i,
            lambda t, x: (t + 1, x + i),
            [0, 0])
        return s

      iters = 50
      pfor_output = pfor_control_flow_ops.pfor(loop_fn, iters)
      for_loop_output = pfor_control_flow_ops.for_loop(loop_fn, dtypes.int32,
                                                       iters)
      self._run(pfor_output, 100, name="pfor_basic")
      self._run(for_loop_output, 100, name="for_loop_basic")

  def benchmark_dynamic_rnn(self):
    with ops.Graph().as_default():
      pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicRNNCell,
                                                     128, 512, 16)
      self._run(pfor_outputs, 100, name="pfor_rnn")
      self._run(tf_outputs, 100, name="tf_rnn")

  def benchmark_dynamic_lstm(self):
    with ops.Graph().as_default():
      pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicLSTMCell,
                                                     128, 512, 16)
      self._run(pfor_outputs, 100, name="pfor_lstm")
      self._run(tf_outputs, 100, name="tf_lstm")


class SparseTest(PForTest):

  def test_var_loop_len(self):
    num_iters = array_ops.placeholder(dtypes.int32)

    def loop_fn(_):
      return sparse_tensor.SparseTensor([[0], [1], [2]], [4, 5, 6],
                                        [3])  # [0, 2, 0]

    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    with self.test_session() as sess:
      sess.run(pfor, feed_dict={num_iters: 3})

  def test_sparse_result_none_stacked(self):
    num_iters = 10

    def loop_fn(_):
      return sparse_tensor.SparseTensor([[0], [1], [2]], [4, 5, 6],
                                        [3])  # [0, 2, 0]

    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)

    indices = [[i, j] for i in range(num_iters) for j in range(3)]
    values = [4, 5, 6] * num_iters
    dense_shapes = [num_iters, 3]
    # Expected result: [[4, 5, 6], [4, 5, 6], [4, 5, 6], ...]
    manual = sparse_tensor.SparseTensor(indices, values, dense_shapes)
    self.run_and_assert_equal(pfor, manual)

  def test_sparse_result_all_stacked(self):
    num_iters = 10

    def loop_fn(i):
      i = array_ops.expand_dims(math_ops.cast(i, dtypes.int64), 0)
      indices = array_ops.expand_dims(i, 0)
      return sparse_tensor.SparseTensor(indices, i, i + 1)  # [0, ..., 0, i]

    # Expected result: [[0], [0, 1], [0, 0, 2], [0, 0, 0, 3], ...]
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, i] for i in range(num_iters)],
                                        list(range(num_iters)),
                                        (num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)

  def test_sparse_result_indices_stacked(self):
    num_iters = 10

    def loop_fn(i):
      i = array_ops.expand_dims(math_ops.cast(i, dtypes.int64), 0)
      indices = array_ops.expand_dims(i, 0)
      return sparse_tensor.SparseTensor(indices, [1], [num_iters])

    # Expected result: identity matrix size num_iters * num_iters
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, i] for i in range(num_iters)],
                                        [1] * num_iters, (num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)

  def test_sparse_result_values_stacked(self):
    num_iters = 10

    def loop_fn(i):
      i = array_ops.expand_dims(math_ops.cast(i, dtypes.int64), 0)
      return sparse_tensor.SparseTensor([[0]], i, [num_iters])  # [i, 0, ..., 0]

    # Expected result: [[1, 0, ...], [2, 0, ...], [3, 0, ...], ...]
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, 0] for i in range(num_iters)],
                                        list(range(num_iters)),
                                        (num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)

  def test_sparse_result_shapes_stacked(self):
    num_iters = 10

    def loop_fn(i):
      i = array_ops.expand_dims(math_ops.cast(i, dtypes.int64), 0)
      return sparse_tensor.SparseTensor([[0]], [1], i + 1)  # [1, 0, ..., 0]

    # Expected result: [[1, 0, 0, ...], [1, 0, 0, ...], ...]
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, 0] for i in range(num_iters)],
                                        [1] * num_iters, (num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)

  def test_sparse_result_shapes_stacked_2D(self):
    num_iters = 10

    def loop_fn(i):
      i = array_ops.expand_dims(math_ops.cast(i + 1, dtypes.int64), 0)
      shape = array_ops.concat([i, i], 0)
      return sparse_tensor.SparseTensor([[0, 0]], [1], shape)  # [1, 0, ..., 0]

    # Expected result: [[[1, 0, ...], [0, ..., 0], [0, ..., 0], ...], ...]
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, 0, 0] for i in range(num_iters)],
                                        [1] * num_iters,
                                        (num_iters, num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)


class ParsingTest(PForTest):

  def test_decode_csv(self):
    csv_tensor = constant_op.constant([["1:2:3"], ["::"], ["7:8:9"]])
    kwargs = {"record_defaults": [[10], [20], [30]], "field_delim": ":"}

    def loop_fn(i):
      line = array_ops.gather(csv_tensor, i)
      return parsing_ops.decode_csv(line, **kwargs)

    self._test_loop_fn(loop_fn, iters=3, loop_fn_dtypes=[dtypes.int32] * 3)

  def test_parse_single_example(self):

    def _int64_feature(*values):
      return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=values))

    def _bytes_feature(*values):
      return feature_pb2.Feature(
          bytes_list=feature_pb2.BytesList(
              value=[v.encode("utf-8") for v in values]))

    examples = constant_op.constant([
        example_pb2.Example(
            features=feature_pb2.Features(
                feature={
                    "dense_int": _int64_feature(i),
                    "dense_str": _bytes_feature(str(i)),
                    "sparse_int": _int64_feature(i, i * 2, i * 4, i * 8),
                    "sparse_str": _bytes_feature(*["abc"] * i)
                })).SerializeToString() for i in range(10)
    ])

    features = {
        "dense_int": parsing_ops.FixedLenFeature((), dtypes.int64, 0),
        "dense_str": parsing_ops.FixedLenFeature((), dtypes.string, ""),
        "sparse_int": parsing_ops.VarLenFeature(dtypes.int64),
        "sparse_str": parsing_ops.VarLenFeature(dtypes.string),
    }

    def loop_fn(i):
      example_proto = array_ops.gather(examples, i)
      f = parsing_ops.parse_single_example(example_proto, features)
      return f

    pfor = pfor_control_flow_ops.pfor(loop_fn, iters=10)
    manual = parsing_ops.parse_example(examples, features)
    self.run_and_assert_equal(pfor, manual)


if __name__ == "__main__":
  test.main()
