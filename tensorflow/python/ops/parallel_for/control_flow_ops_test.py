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
# pylint: disable=g-direct-tensorflow-import

import functools
import sys
import time

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class PForTest(PForTestCase):

  def test_op_conversion_fallback_to_while_loop(self):
    # Note that we used top_k op for this test. If a converter gets defined for
    # it, we will need to find another op for which a converter has not been
    # defined.
    x = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return nn.top_k(x_i)

    self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=True)

  def test_parallel_iterations(self):
    for parallel_iterations in [2, 3, 8, 10]:
      x = random_ops.random_uniform([8, 3])

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        return array_ops.gather(x, i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 8, parallel_iterations=parallel_iterations)
      self._test_loop_fn(
          loop_fn,
          4 * constant_op.constant(2),
          parallel_iterations=parallel_iterations)

  def test_parallel_iterations_preserves_static_shape(self):
    for parallel_iterations in [2, 3, 8, 10]:
      x = pfor_control_flow_ops.pfor(
          lambda _: random_ops.random_uniform([2, 3]),
          8,
          parallel_iterations=parallel_iterations)
      self.assertAllEqual(x.shape, [8, 2, 3])

  def test_parallel_iterations_zero(self):
    with self.assertRaisesRegex(ValueError, "positive integer"):
      pfor_control_flow_ops.pfor(lambda i: 1, 8, parallel_iterations=0)
    with self.assertRaisesRegex(TypeError, "positive integer"):
      pfor_control_flow_ops.for_loop(
          lambda i: 1, dtypes.int32, 8, parallel_iterations=0)

  def test_parallel_iterations_one(self):
    with self.assertRaisesRegex(ValueError, "Use `for_loop` instead"):
      pfor_control_flow_ops.pfor(lambda i: 1, 8, parallel_iterations=1)

  def test_zero_loop_iters_basic(self):
    self._test_loop_fn(lambda i: 1, 0)

  def test_zero_loop_iters_tensor(self):
    self._test_loop_fn(
        lambda i: array_ops.zeros([10, 3], dtype=dtypes.int32), 0
    )

  def test_vectorized_map(self):

    def compute(x):
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    result = pfor_control_flow_ops.vectorized_map(compute,
                                                  array_ops.ones((10, 5, 3)))
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))

  def test_vectorized_map_with_dynamic_shape(self):

    def compute(x):
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    x = array_ops.placeholder_with_default(
        array_ops.ones((10, 5, 3)), shape=None)
    result = pfor_control_flow_ops.vectorized_map(compute, x)
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))

  def test_where_shape(self):
    @def_function.function
    def f():
      a = constant_op.constant([[1.], [1.]])
      b = constant_op.constant([1.])
      result = pfor_control_flow_ops.vectorized_map(
          lambda x: array_ops.where(x > 0, x, b), a)
      return result.shape

    self.assertAllEqual([2, 1], f())

  def test_vectorized_map_broadcasts_unit_dimensions(self):
    convert_with_static_shape = ops.convert_to_tensor
    convert_with_dynamic_shape = (
        lambda x: array_ops.placeholder_with_default(x, shape=None))

    for convert in (convert_with_static_shape, convert_with_dynamic_shape):
      a = convert([3.1])
      b = convert([-2., 6., 9.])

      # One elem with leading unit dimension.
      a_plus_1 = pfor_control_flow_ops.vectorized_map(lambda a: a + 1, a)
      self.assertAllEqual(*self.evaluate((a_plus_1, a + 1)))

      # Two elems, both with leading unit dimension.
      a_plus_a = pfor_control_flow_ops.vectorized_map(sum, (a, a))
      self.assertAllEqual(*self.evaluate((a_plus_a, a + a)))

      # Elem w/ unit dimension broadcast against elem with batch dim.
      a_plus_b = pfor_control_flow_ops.vectorized_map(sum, (a, b))
      self.assertAllEqual(*self.evaluate((a_plus_b, a + b)))

  def test_vectorized_map_example_1(self):

    def outer_product(a):
      return math_ops.tensordot(a, a, 0)

    batch_size = 100
    a = array_ops.ones((batch_size, 32, 32))
    c = pfor_control_flow_ops.vectorized_map(outer_product, a)
    self.assertAllEqual((batch_size, 32, 32, 32, 32), c.shape)

  def test_disable_tf_function(self):
    def_function.run_functions_eagerly(True)
    # vectorized_map should ignore disabling tf.functions
    self.assertTrue(def_function.functions_run_eagerly())
    self.assertAllEqual([0, 1, 4, 9],
                        pfor_control_flow_ops.vectorized_map(
                            lambda x: x * x, math_ops.range(4)))
    self.assertTrue(def_function.functions_run_eagerly())
    def_function.run_functions_eagerly(False)


@test_util.run_all_in_graph_and_eager_modes
class IndexedSlicesTest(PForTestCase):

  def test_indexed_slices(self):

    def loop_fn(i):
      return indexed_slices.IndexedSlices(
          indices=i, values=array_ops.reshape(i, [1]), dense_shape=[3, 1])

    self._test_loop_fn(loop_fn, 2)

  def test_indexed_slices_components(self):

    def loop_fn(i):
      slices = indexed_slices.IndexedSlices(
          indices=i, values=array_ops.reshape(i, [1]), dense_shape=[3, 1])
      # Note that returning the components inside the slice avoids
      # densification, which may be more efficient.
      return slices.values, slices.indices

    self._test_loop_fn(loop_fn, 2)


@test_util.run_all_in_graph_and_eager_modes
class ReductionTest(PForTestCase):

  def test_reduce(self):

    def reduce_fn(p, q):
      return math_ops.reduce_mean(p + q, axis=0)

    x = random_ops.random_uniform([4, 3, 2])
    y = random_ops.random_uniform([4, 3, 2])

    def loop_fn(i, pfor_config):
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      reduced = pfor_config.reduce(reduce_fn, x_i, y_i)
      return reduced + x_i

    output = pfor_control_flow_ops.pfor(loop_fn, 4)
    ans = reduce_fn(x, y) + x
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_concat(self):
    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      x_i = array_ops.gather(x, i)
      vectorized_value = pfor_config.reduce_concat(x_i)
      mean_value = math_ops.reduce_mean(vectorized_value, axis=0)
      return x_i - mean_value

    output = pfor_control_flow_ops.pfor(loop_fn, 8)
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_mean(self):
    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      x_i = array_ops.gather(x, i)
      return x_i - pfor_config.reduce_mean(x_i)

    output = pfor_control_flow_ops.pfor(loop_fn, 8)
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_sum(self):
    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      x_i = array_ops.gather(x, i)
      return x_i - pfor_config.reduce_sum(x_i)

    output = pfor_control_flow_ops.pfor(loop_fn, 8)
    ans = x - math_ops.reduce_sum(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_class(self):
    x = random_ops.random_uniform([8, 3])

    class LoopFn:

      def __init__(self):
        pass

      def __call__(self, i, pfor_config):
        x_i = array_ops.gather(x, i)
        return x_i - pfor_config.reduce_mean(x_i)

    output = pfor_control_flow_ops.pfor(LoopFn(), 8)
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_functools_partial(self):
    x = random_ops.random_uniform([8, 3])

    def fn(i, pfor_config, dummy=None):
      del dummy
      x_i = array_ops.gather(x, i)
      return x_i - pfor_config.reduce_mean(x_i)

    loop_fn = functools.partial(fn, dummy=1)
    output = pfor_control_flow_ops.pfor(loop_fn, 8)
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_parallel_iterations(self):
    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      x_i = array_ops.gather(x, i)
      return pfor_config.reduce_sum(x_i)

    with self.assertRaisesRegex(ValueError,
                                "`parallel_iterations` currently unsupported"):
      pfor_control_flow_ops.pfor(loop_fn, 8, parallel_iterations=2)

  def test_var_loop_len(self):
    if context.executing_eagerly():
      self.skipTest("Variable length not possible under eager execution.")

    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      return pfor_config.reduce_sum(array_ops.gather(x, i))

    num_iters = array_ops.placeholder(dtypes.int32)
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    with self.cached_session() as sess:
      sess.run(pfor, feed_dict={num_iters: 8})


@test_util.run_all_in_graph_and_eager_modes
class BitwiseTest(PForTestCase):

  def test_unary_cwise(self):
    for op in [bitwise_ops.invert]:
      x = random_ops.random_uniform([7, 3, 5], maxval=10, dtype=dtypes.int32)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        x1 = array_ops.gather(x, i)
        return op(x1)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_binary_cwise(self):
    binary_ops = [
        bitwise_ops.bitwise_and,
        bitwise_ops.bitwise_or,
        bitwise_ops.bitwise_xor,
        bitwise_ops.left_shift,
        bitwise_ops.right_shift,
    ]
    for op in binary_ops:
      x = random_ops.random_uniform([7, 3, 5], maxval=10, dtype=dtypes.int32)
      y = random_ops.random_uniform([3, 5], maxval=10, dtype=dtypes.int32)

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


@test_util.run_all_in_graph_and_eager_modes
class ImageTest(PForTestCase):

  def test_adjust_contrast(self):
    images = random_ops.random_uniform([3, 2, 4, 4, 3])

    def loop_fn(i):
      image = array_ops.gather(images, i)
      return image_ops.adjust_contrast(image, 2.0)

    self._test_loop_fn(loop_fn, 3)

  def test_adjust_hue(self):
    images = random_ops.random_uniform([3, 2, 4, 4, 3])

    def loop_fn(i):
      image = array_ops.gather(images, i)
      return image_ops.adjust_hue(image, .25)

    self._test_loop_fn(loop_fn, 3)

  def test_adjust_saturation(self):
    images = random_ops.random_uniform([3, 2, 4, 4, 3])

    def loop_fn(i):
      image = array_ops.gather(images, i)
      return image_ops.adjust_saturation(image, 0.1)

    self._test_loop_fn(loop_fn, 3)


@test_util.run_all_in_graph_and_eager_modes
@test_util.run_all_without_tensor_float_32("Uses matmul")
class NNTest(PForTestCase):

  def test_conv2d(self):
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    filt = random_ops.random_uniform([3, 3, 3, 7])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return nn.conv2d(
          x1, filt, strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_conv2d_backprop_input(self):
    self.skipTest("b/262851489: Fix nightly build for GPU.")
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

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native(self):
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    filt = random_ops.random_uniform([3, 3, 3, 3, 2])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      filt1 = array_ops.gather(filt, i)
      return nn.depthwise_conv2d_native(
          x1, filt1, strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_backprop_input(self):
    x_shape = [2, 12, 12, 3]
    filt = random_ops.random_uniform([3, 3, 3, 3, 2])
    grad = random_ops.random_uniform([3, 2, 5, 5, 6])

    def loop_fn(i):
      grad1 = array_ops.gather(grad, i)
      filt1 = array_ops.gather(filt, i)
      return nn.depthwise_conv2d_native_backprop_input(
          x_shape,
          filt1,
          grad1,
          strides=[1, 2, 2, 1],
          padding="VALID",
          data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_backprop_filter(self):
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    filter_sizes = [3, 3, 3, 2]
    grad = random_ops.random_uniform([3, 2, 5, 5, 6])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      grad_i = array_ops.gather(grad, i)
      return nn.depthwise_conv2d_native_backprop_filter(
          x_i,
          filter_sizes,
          grad_i,
          strides=[1, 2, 2, 1],
          padding="VALID",
          data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_nchw(self):
    if not test_util.is_gpu_available():
      self.skipTest("NCHW only works on GPU")
    x = random_ops.random_uniform([3, 2, 3, 12, 12])
    filt = random_ops.random_uniform([3, 3, 3, 3, 2])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      filt1 = array_ops.gather(filt, i)
      return nn.depthwise_conv2d_native(
          x1, filt1, strides=[1, 1, 2, 2], padding="VALID", data_format="NCHW")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_backprop_input_nchw(self):
    if not test_util.is_gpu_available():
      self.skipTest("NCHW only works on GPU")
    x_shape = [2, 3, 12, 12]
    filt = random_ops.random_uniform([3, 3, 3, 3, 2])
    grad = random_ops.random_uniform([3, 2, 6, 5, 5])

    def loop_fn(i):
      grad1 = array_ops.gather(grad, i)
      filt1 = array_ops.gather(filt, i)
      return nn.depthwise_conv2d_native_backprop_input(
          x_shape,
          filt1,
          grad1,
          strides=[1, 1, 2, 2],
          padding="VALID",
          data_format="NCHW")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_backprop_filter_nchw(self):
    if not test_util.is_gpu_available():
      self.skipTest("NCHW only works on GPU")
    x = random_ops.random_uniform([3, 2, 3, 12, 12])
    filter_sizes = [3, 3, 3, 2]
    grad = random_ops.random_uniform([3, 2, 6, 5, 5])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      grad_i = array_ops.gather(grad, i)
      return nn.depthwise_conv2d_native_backprop_filter(
          x_i,
          filter_sizes,
          grad_i,
          strides=[1, 1, 2, 2],
          padding="VALID",
          data_format="NCHW")

    self._test_loop_fn(loop_fn, 3)

  def test_roll(self):
    x = random_ops.random_uniform([3, 6, 7])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return manip_ops.roll(x_i, 3, axis=1)

    self._test_loop_fn(loop_fn, 3)

  def test_ensure_shape(self):
    x = random_ops.random_uniform([3, 6, 7])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.ensure_shape(x_i, [6, 7])

    self._test_loop_fn(loop_fn, 3)

  def test_loop_variant_roll_shift(self):
    x = random_ops.random_uniform([3, 5, 6, 7])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return manip_ops.roll(x_i, [i - 2, -1, i], axis=[1, 2, 2])

    self._test_loop_fn(loop_fn, 3)

  def test_loop_variant_roll_scalar_shift(self):
    x = random_ops.random_uniform([5, 5, 6])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return manip_ops.roll(x_i, i, axis=0)

    self._test_loop_fn(loop_fn, 5)

  def test_avg_pool(self):
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 2, 12, 12, 3])
      g.watch(x)
      ksize = [1, 3, 3, 1]

    def loop_fn(i):
      with g:
        x1 = array_ops.gather(x, i)
        output = nn.avg_pool(
            x1,
            ksize,
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC")
        loss = nn.l2_loss(output)
      return output, g.gradient(loss, x1)

    self._test_loop_fn(loop_fn, 3)

  def test_avg_pool3d(self):
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([5, 3, 7, 6, 6, 5])
      g.watch(x)
      ksize = [1, 2, 2, 2, 1]
      strides = [1, 2, 2, 2, 1]

    def loop_fn(i):
      with g:
        x1 = array_ops.gather(x, i)
        output = nn.avg_pool3d(
            x1, ksize, strides=strides, padding="VALID", data_format="NDHWC")
        loss = nn.l2_loss(output)
      return output, g.gradient(loss, x1)

    self._test_loop_fn(loop_fn, 3)

  def test_max_pool(self):
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 2, 12, 12, 3])
      g.watch(x)
      ksize = [1, 3, 3, 1]
      strides = [1, 2, 2, 1]

    def loop_fn(i):
      with g:
        x1 = array_ops.gather(x, i)
        output = nn.max_pool(
            x1, ksize, strides=strides, padding="VALID", data_format="NHWC")
        loss = nn.l2_loss(output)
        ones = array_ops.ones_like(output)
        g.watch(ones)
        grad = g.gradient(loss, x1, output_gradients=ones)
      grad_grad = g.gradient(grad, ones)
      return output, grad, grad_grad

    self._test_loop_fn(loop_fn, 3)

  def test_max_pool_v2(self):
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 2, 12, 12, 3])
      g.watch(x)
      ksize = [1, 3, 3, 1]
      strides = [1, 2, 2, 1]

    def loop_fn(i):
      with g:
        x1 = array_ops.gather(x, i)
        output = gen_nn_ops.max_pool_v2(
            x1, ksize, strides=strides, padding="VALID", data_format="NHWC")
        loss = nn.l2_loss(output)
        ones = array_ops.ones_like(output)
        g.watch(ones)
        grad = g.gradient(loss, x1, output_gradients=ones)
      grad_grad = g.gradient(grad, ones)
      return output, grad, grad_grad

    self._test_loop_fn(loop_fn, 3)

  def test_max_pool3d(self):
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 3, 2, 12, 12, 3])
      g.watch(x)
      ksize = [1, 1, 3, 3, 1]
      strides = [1, 1, 2, 2, 1]

    def loop_fn(i):
      with g:
        x1 = array_ops.gather(x, i)
        output = nn.max_pool3d(
            x1, ksize, strides=strides, padding="VALID", data_format="NDHWC")
        loss = nn.l2_loss(output)
        ones = array_ops.ones_like(output)
        g.watch(ones)
        grad = g.gradient(loss, x1, output_gradients=ones)
      grad_grad = g.gradient(grad, ones)
      return output, grad, grad_grad

    self._test_loop_fn(loop_fn, 3)

  def test_fused_batch_norm(self):
    data_formats = ["NHWC"]
    if test.is_gpu_available():
      data_formats.append("NCHW")
    for is_training in (True, False):
      for data_format in data_formats:
        with backprop.GradientTape(persistent=True) as g:
          if data_format == "NCHW":
            x = random_ops.random_uniform([3, 1, 2, 5, 5])
          else:
            x = random_ops.random_uniform([3, 1, 5, 5, 2])
          g.watch(x)
          scale = random_ops.random_uniform([2])
          g.watch(scale)
          offset = random_ops.random_uniform([2])
          g.watch(offset)
          mean = None if is_training else random_ops.random_uniform([2])
          variance = None if is_training else random_ops.random_uniform([2])

        # pylint: disable=cell-var-from-loop
        def loop_fn(i):
          with g:
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
            # We only test the first value of outputs when is_training is
            # False. It looks like CPU and GPU have different outputs for
            # batch_mean and batch_variance for this case.
            if not is_training:
              outputs[1] = constant_op.constant(0.)
              outputs[2] = constant_op.constant(0.)
            loss = nn.l2_loss(outputs[0])
          if is_training:
            gradients = g.gradient(loss, [x1, scale, offset])
          else:
            gradients = [constant_op.constant(0.)] * 3
          return outputs + gradients

        # pylint: enable=cell-var-from-loop

        self._test_loop_fn(loop_fn, 3)

  def test_log_softmax(self):
    logits = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      logits_i = array_ops.gather(logits, i)
      return (nn.log_softmax(logits_i), nn.log_softmax(logits_i, axis=0),
              nn.log_softmax(logits_i, axis=-1))

    self._test_loop_fn(loop_fn, 3)

  def test_softmax(self):
    logits = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      logits_i = array_ops.gather(logits, i)
      return (nn.softmax(logits_i), nn.softmax(logits_i, axis=0),
              nn.softmax(logits_i, axis=-1))

    self._test_loop_fn(loop_fn, 3)

  def test_softmax_cross_entropy_with_logits(self):
    with backprop.GradientTape(persistent=True) as g:
      logits = random_ops.random_uniform([3, 2, 4])
      g.watch(logits)
      labels = random_ops.random_uniform([3, 2, 4])
      labels /= math_ops.reduce_sum(labels, axis=[2], keepdims=True)

    def loop_fn(i):
      with g:
        logits_i = array_ops.gather(logits, i)
        labels_i = array_ops.gather(labels, i)
        loss = nn.softmax_cross_entropy_with_logits(
            labels=labels_i, logits=logits_i)
        total_loss = math_ops.reduce_sum(loss)
      return loss, g.gradient(total_loss, logits_i)

    self._test_loop_fn(loop_fn, 3)

  def test_sparse_softmax_cross_entropy_with_logits(self):
    logits = random_ops.random_uniform([3, 2, 4])
    labels = random_ops.random_uniform(
        shape=[3, 2], maxval=4, dtype=dtypes.int32)

    def loop_fn(i):
      logits_i = array_ops.gather(logits, i)
      labels_i = array_ops.gather(labels, i)
      loss = nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels_i, logits=logits_i)
      return loss

    self._test_loop_fn(loop_fn, 3)


class RandomTest(PForTestCase):

  # The random values generated in the two implementations are not guaranteed to
  # match. So we only check the returned shapes.
  def run_and_assert_equal(self, targets1, targets2, rtol=1e-4, atol=1e-5):
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

    self._test_loop_fn(loop_fn, 5)

  def test_random_standard_normal(self):

    def loop_fn(_):
      return random_ops.random_normal([3])

    self._test_loop_fn(loop_fn, 5)

  def test_truncated_normal(self):

    def loop_fn(_):
      return random_ops.truncated_normal([3])

    self._test_loop_fn(loop_fn, 5)

  def test_random_gamma_invariant_alpha(self):

    def loop_fn(_):
      return random_ops.random_gamma([3], alpha=[0.5])

    self._test_loop_fn(loop_fn, 5)

  def test_random_gamma_varying_alpha(self):
    alphas = math_ops.exp(random_ops.random_normal([5, 3, 2]))

    def loop_fn(i):
      alphas_i = array_ops.gather(alphas, i)
      # Test both scalar and non-scalar params and shapes.
      return (random_ops.random_gamma(alpha=alphas_i[0, 0], shape=[]),
              random_ops.random_gamma(alpha=alphas_i, shape=[]),
              random_ops.random_gamma(alpha=alphas_i[0, 0], shape=[3]),
              random_ops.random_gamma(alpha=alphas_i, shape=[3]))

    self._test_loop_fn(loop_fn, 5)

  def test_random_poisson_v2_invariant_rate(self):

    def loop_fn(_):
      return random_ops.random_poisson(lam=[1.3], shape=[3])

    self._test_loop_fn(loop_fn, 5)

  def test_random_poisson_v2_varying_rate(self):
    rates = math_ops.exp(random_ops.random_normal([5, 3, 2]))

    def loop_fn(i):
      rates_i = array_ops.gather(rates, i)
      # Test both scalar and non-scalar params and shapes.
      return (random_ops.random_poisson(lam=rates_i[0, 0], shape=[]),
              random_ops.random_poisson(lam=rates_i, shape=[]),
              random_ops.random_poisson(lam=rates_i[0, 0], shape=[3]),
              random_ops.random_poisson(lam=rates_i, shape=[3]))

    self._test_loop_fn(loop_fn, 5)

  def test_random_multinomial_invariant_logits(self):

    def loop_fn(_):
      return random_ops.categorical(logits=[[1., -1.]], num_samples=3)

    self._test_loop_fn(loop_fn, 5)

  def test_random_multinomial_varying_logits(self):
    logits = random_ops.random_normal([5, 3, 2])

    def loop_fn(i):
      logits_i = array_ops.gather(logits, i)
      return random_ops.categorical(logits_i, num_samples=3)

    self._test_loop_fn(loop_fn, 5)


class StatelessRandomTest(PForTestCase):

  # This test currently only tests that the vectorized and non-vectorized
  # outputs have same shapes. This is needed since under XLA compilation,
  # stateless random numbers can generate different random numbers.
  # TODO(agarwal): switch to checking for actual values matching once
  # b/149402339 is resolved.
  def run_and_assert_equal(self, targets1, targets2, rtol=1e-4, atol=1e-5):
    outputs = self._run_targets(targets1, targets2)
    n = len(outputs) // 2
    for i in range(n):
      self.assertAllEqual(outputs[i].shape, outputs[i + n].shape)

  # TODO(agarwal): add tests for other random functions
  def test_multinomial(self):
    seeds = [[1, 2], [3, 4]]
    logits = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      logits_0 = array_ops.gather(logits, 0)
      logits_i = array_ops.gather(logits, i)
      seeds_0 = array_ops.gather(seeds, 0)
      seeds_i = array_ops.gather(seeds, i)
      return (stateless_random_ops.stateless_categorical(
          logits=logits_i, num_samples=3, seed=seeds_i),
              stateless_random_ops.stateless_categorical(
                  logits=logits_i, num_samples=3, seed=seeds_0),
              stateless_random_ops.stateless_categorical(
                  logits=logits_0, num_samples=3, seed=seeds_i),
              stateless_random_ops.stateless_categorical(
                  logits=logits_0, num_samples=3, seed=seeds_0))

    self._test_loop_fn(loop_fn, 2)


class LoggingTest(PForTestCase):

  def test_print(self):
    x = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return logging_ops.Print(
          x1, [x1, "x1", array_ops.shape(x1)], summarize=10)

    self._test_loop_fn(loop_fn, 3)

  def test_print_v2(self):
    x = constant_op.constant([1, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      with ops.control_dependencies([
          logging_ops.print_v2(
              x1, "x1", array_ops.shape(x1), summarize=10)]):
        return array_ops.identity(x1)

    self._test_loop_fn(loop_fn, 3)

    with self.captureWritesToStream(sys.stderr) as printed:
      self.evaluate(pfor_control_flow_ops.pfor(loop_fn, 3))
    self.assertIn("[1 2 3] x1 []", printed.contents())

  def test_assert(self):

    def loop_fn(i):
      return control_flow_assert.Assert(i < 10, [i, [10], [i + 1]])

    # TODO(agarwal): make this work with for_loop.
    with session.Session() as sess:
      sess.run(pfor_control_flow_ops.pfor(loop_fn, 3))
      sess.run(pfor_control_flow_ops.pfor(
          lambda i, pfor_config: loop_fn(i), 3))


class TensorArrayTest(PForTestCase):

  def setUp(self):
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.disable_control_flow_v2()
    super(TensorArrayTest, self).setUp()

  def tearDown(self):
    if self._enabled:
      control_flow_v2_toggles.enable_control_flow_v2()
    super(TensorArrayTest, self).tearDown()

  @test_util.run_v1_only("b/122612051")
  def test_create_outside_and_read(self):

    ta = tensor_array_ops.TensorArray(
        dtypes.int32, 2, clear_after_read=False).write(0, 0).write(1, 1)

    def loop_fn(i):
      return ta.read(i), ta.read(0)

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
  def test_create_outside_and_gather(self):

    ta = tensor_array_ops.TensorArray(
        dtypes.int32, 2, clear_after_read=False).write(0, 0).write(1, 1)

    def loop_fn(i):
      return ta.gather([i]), ta.gather([0, 1])

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
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

  @test_util.run_v1_only("b/122612051")
  def test_create_inside_and_write(self):

    def loop_fn(i):
      # TODO(agarwal): switching the order of writes to ta1 does not work.
      ta1 = tensor_array_ops.TensorArray(dtypes.int32, 2).write(0,
                                                                i).write(1, 1)
      ta2 = tensor_array_ops.TensorArray(dtypes.int32, 1).write(0, 1)
      return ta1.stack(), ta2.stack()

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_create_inside_and_scatter(self):

    def loop_fn(i):
      # TODO(agarwal): switching the order of scatter to ta1 does not work.
      ta1 = tensor_array_ops.TensorArray(dtypes.int32,
                                         2).scatter([0],
                                                    [[i, 2]]).scatter([1],
                                                                      [[1, 2]])
      ta2 = tensor_array_ops.TensorArray(dtypes.int32,
                                         2).scatter([0], [3]).scatter([1], [4])
      return ta1.stack(), ta2.stack()

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_create_inside_and_read(self):

    def loop_fn(i):
      ta1 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, i).write(1, 1)
      ta2 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, 1).write(1, 2)
      # TODO(agarwal): ta1.read(i) currently is not supported.
      return ta1.read(0), ta2.read(0), ta2.read(i)

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
  def test_create_inside_and_gather(self):

    def loop_fn(i):
      ta1 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, i).write(1, 1)
      ta2 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, 1).write(1, 2)
      # TODO(agarwal): ta1.read(i) currently is not supported.
      return ta1.gather([0, 1]), ta2.gather([0, 1]), ta2.gather([i])

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
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


@test_util.run_all_in_graph_and_eager_modes
class TensorListTest(PForTestCase):

  def test_create_outside_and_write(self):
    handle1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    handle2 = list_ops.tensor_list_reserve([], 2, dtypes.int32)

    def loop_fn(i):
      h1 = list_ops.tensor_list_set_item(handle1, 0, i)
      h1 = list_ops.tensor_list_set_item(h1, 1, 1)
      h2 = list_ops.tensor_list_set_item(handle2, 0, 1)
      return (list_ops.tensor_list_stack(h1, dtypes.int32),
              list_ops.tensor_list_stack(h2, dtypes.int32))

    self._test_loop_fn(loop_fn, 3)

  def _make_graph_def(self, text):
    ret = graph_pb2.GraphDef()
    text_format.Parse(text, ret)
    return ret

  def test_no_fallback_with_internal_stacking(self):

    # Create an op (really a function) that pfor definitely does not have a
    # converter for. Assumes pfor does not start looking up function definitions
    # for op-type-is-function-name calls.
    @def_function.function
    def opaque_list_fetch(x):
      array_ops.identity(x)
      return list_ops.tensor_list_get_item(x, 0, dtypes.int32)

    external_handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    opaque_list_fetch_concrete = opaque_list_fetch.get_concrete_function(
        external_handle)
    opaque_list_fetch_name = opaque_list_fetch_concrete.name

    def loop_fn(i):
      h1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      h1 = list_ops.tensor_list_set_item(h1, 0, i)
      opaque_list_fetch_concrete.add_to_graph()
      graph_def = self._make_graph_def("""
         node { name: 'x' op: 'Placeholder'
                attr { key: 'dtype' value { type: DT_FLOAT } }}
         node { name: 'fn' op: '""" + opaque_list_fetch_name.decode()
                                       + """' input: 'x:0' }""")
      return importer.import_graph_def(
          graph_def,
          input_map={"x:0": h1},
          return_elements=["fn"],
          name="import")[0].outputs[0]

    with self.assertRaisesRegex(ValueError, "No pfor vectorization"):
      self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=False)
    with self.assertRaisesRegex(ValueError, "No pfor vectorization"):
      self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=True)

  def test_create_inside_and_write(self):

    def loop_fn(i):
      h1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      h1 = list_ops.tensor_list_set_item(h1, 0, i)
      h1 = list_ops.tensor_list_set_item(h1, 1, 1)
      h2 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      h2 = list_ops.tensor_list_set_item(h2, 0, 1)
      return (list_ops.tensor_list_stack(h1, dtypes.int32),
              list_ops.tensor_list_stack(h2, dtypes.int32))

    self._test_loop_fn(loop_fn, 3)

  def test_create_outside_and_read(self):
    handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    handle = list_ops.tensor_list_set_item(handle, 0, 0)
    handle = list_ops.tensor_list_set_item(handle, 1, 1)

    def loop_fn(i):
      return (list_ops.tensor_list_get_item(handle, i, dtypes.int32),
              list_ops.tensor_list_get_item(handle, 0, dtypes.int32),
              list_ops.tensor_list_length(handle),
              list_ops.tensor_list_element_shape(handle, dtypes.int32),
              list_ops.tensor_list_element_shape(handle, dtypes.int64))

    self._test_loop_fn(loop_fn, 2)

  def test_create_outside_and_read_zero_loop_iters(self):
    handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    handle = list_ops.tensor_list_set_item(handle, 0, 0)
    handle = list_ops.tensor_list_set_item(handle, 1, 1)

    def loop_fn(i):
      return (
          list_ops.tensor_list_get_item(handle, i, dtypes.int32),
          list_ops.tensor_list_get_item(handle, 0, dtypes.int32),
          list_ops.tensor_list_length(handle),
          list_ops.tensor_list_element_shape(handle, dtypes.int32),
          list_ops.tensor_list_element_shape(handle, dtypes.int64),
      )

    self._test_loop_fn(loop_fn, 0)

  def test_create_inside_and_read(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      handle = list_ops.tensor_list_set_item(handle, 0, i)
      handle = list_ops.tensor_list_set_item(handle, 1, 1)
      return (list_ops.tensor_list_get_item(handle, 0, dtypes.int32),
              list_ops.tensor_list_get_item(handle, i, dtypes.int32),
              list_ops.tensor_list_length(handle),
              list_ops.tensor_list_element_shape(handle, dtypes.int32),
              list_ops.tensor_list_element_shape(handle, dtypes.int64))

    self._test_loop_fn(loop_fn, 2)

  def test_create_outside_and_push_back(self):
    h = list_ops.tensor_list_reserve([2], 2, dtypes.int32)

    def loop_fn(i):
      handle = list_ops.tensor_list_push_back(h, [i, 2])
      handle = list_ops.tensor_list_push_back(handle, [1, 2])
      handle = list_ops.tensor_list_push_back(handle, [1, 2])
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_create_inside_and_push_back(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_push_back(handle, [i, 2])
      handle = list_ops.tensor_list_push_back(handle, [1, 2])
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_pop_back_no_shape(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_push_back(handle, [1, 2])
      handle = list_ops.tensor_list_push_back(handle, [i, 2])
      handle, tensor = list_ops.tensor_list_pop_back(handle, dtypes.int32)
      return tensor, list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_pop_back_no_shape_capture(self):
    h = list_ops.tensor_list_reserve([2], 1, dtypes.int32)
    h = list_ops.tensor_list_push_back(h, [1, 2])

    def loop_fn(i):
      handle, tensor = list_ops.tensor_list_pop_back(h, dtypes.int32)
      handle = list_ops.tensor_list_push_back(handle, [1, i])
      return tensor, list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_pop_back_with_shape(self):

    @def_function.function
    def loop_fn(i):
      with backprop.GradientTape() as tape:
        handle = list_ops.tensor_list_reserve(None, 1, dtypes.float32)
        x = math_ops.cast(i, dtypes.float32)[None]
        tape.watch(x)
        handle = list_ops.tensor_list_push_back(handle, x)
        stacked = list_ops.tensor_list_stack(handle, dtypes.float32)
      list_grad = tape.gradient(stacked, x, x)
      self.assertEqual("TensorListPopBack", list_grad.op.type)
      return list_grad, stacked, list_grad.op.inputs[1]

    self._test_loop_fn(loop_fn, 3)

  def test_create_outside_and_scatter(self):
    h = list_ops.tensor_list_reserve([2], 2, dtypes.int32)

    def loop_fn(i):
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=h)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_create_inside_and_scatter(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=handle)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_loop_variant_scatter_indices(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([2], 10, dtypes.int32)
      handle = list_ops.tensor_list_scatter(
          [[1, i], [i + 1, 2]],
          [i, i + 5], input_handle=handle)
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_duplicate_indices(self):
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([2], 10, dtypes.int32)
      handle = list_ops.tensor_list_scatter(
          [[1, i], [1, i + 1], [i + 2, 3]],
          [i, i, i + 2], input_handle=handle)
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  def test_create_outside_and_gather(self):
    handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
    handle = list_ops.tensor_list_scatter([[2, 3]], [0], input_handle=handle)
    handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)

    def loop_fn(i):
      return (list_ops.tensor_list_gather(handle, [0, 1], dtypes.int32),
              list_ops.tensor_list_gather(handle, [i], dtypes.int32))

    self._test_loop_fn(loop_fn, 2)

  def test_create_inside_and_gather(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=handle)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return (list_ops.tensor_list_gather(handle, [0, 1], dtypes.int32),
              list_ops.tensor_list_gather(handle, [i], dtypes.int32))

    self._test_loop_fn(loop_fn, 2)

  def test_create_inside_and_concat(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=handle)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return gen_list_ops.tensor_list_concat_v2(
          handle,
          element_dtype=dtypes.int32,
          element_shape=[2],
          leading_dims=[])

    output = pfor_control_flow_ops.pfor(loop_fn, 2)
    self.assertAllClose([[0, 2, 1, 2], [1, 2, 1, 2]], output[0])
    self.assertAllClose([[2, 2], [2, 2]], output[1])

  def test_create_outside_and_concat(self):
    h = list_ops.tensor_list_reserve([2], 2, dtypes.int32)

    def loop_fn(i):
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=h)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return gen_list_ops.tensor_list_concat_v2(
          handle,
          element_dtype=dtypes.int32,
          element_shape=[2],
          leading_dims=[])

    output = pfor_control_flow_ops.pfor(loop_fn, 2)
    self.assertAllClose([[0, 2, 1, 2], [1, 2, 1, 2]], output[0])
    self.assertAllClose([[2, 2], [2, 2]], output[1])

  def test_tensor_list_from_tensor(self):
    t = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      handle = list_ops.tensor_list_from_tensor(array_ops.gather(t, i), [4])
      return list_ops.tensor_list_stack(handle, t.dtype)

    self._test_loop_fn(loop_fn, 2)

  @test_util.enable_control_flow_v2
  def test_tensor_list_reserve_while_loop(self):
    # Here a loop invariant TensorList is captured by a while_loop, which then
    # performs loop dependent operations on it, resulting in a loop variant
    # output. This forces stacking of the variant handle captured by the
    # while_loop.
    # We handle this particular case by forcing vectorization of
    # TensorListReserve operation.

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      _, out_handle = while_loop.while_loop(
          lambda j, _: j < 2, lambda j, h:
          (j + 1, list_ops.tensor_list_set_item(h, j, i)), (0, handle))
      return list_ops.tensor_list_stack(out_handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 2)

  @test_util.enable_control_flow_v2
  def test_tensor_list_while_loop_stacked_cond_stacked_list(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_from_tensor([20, 21, 22, 23, i], [])
      _, out_handle = while_loop.while_loop(
          lambda j, _: j < i,
          lambda j, h: (j + 1, list_ops.tensor_list_set_item(h, j, i)),
          (0, handle))
      return list_ops.tensor_list_stack(out_handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  @test_util.enable_control_flow_v2
  def test_tensor_list_while_loop_stacked_cond_stacked_list_unknown_shape(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_reserve(None, 5, dtypes.int32)
      _, handle = while_loop.while_loop(
          lambda j, _: j < 5,
          lambda j, h: (j + 1, list_ops.tensor_list_set_item(h, j, 0)),
          (0, handle))
      _, out_handle = while_loop.while_loop(
          lambda j, _: j < i,
          lambda j, h: (j + 1, list_ops.tensor_list_set_item(h, j, i)),
          (0, handle))
      return list_ops.tensor_list_stack(out_handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  @test_util.enable_control_flow_v2
  def test_tensor_list_while_loop_stacked_cond_unstacked_list(self):

    def loop_fn(i):
      handle = list_ops.tensor_list_from_tensor([20, 21, 22, 23, 24], [])
      _, out_handle = while_loop.while_loop(
          lambda j, _: j < i, lambda j, h:
          (j + 1, list_ops.tensor_list_set_item(h, j, i)), (0, handle))
      return list_ops.tensor_list_stack(out_handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  def test_tensor_list_addn_already_stacked(self):

    def loop_fn(i):
      l1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      l1 = list_ops.tensor_list_set_item(l1, 0, i)
      l2 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      l2 = list_ops.tensor_list_set_item(l2, 1, i)
      return list_ops.tensor_list_stack(math_ops.add_n([l1, l2]), dtypes.int32)

    self._test_loop_fn(loop_fn, 2)

  def test_tensor_list_addn_stacking_required(self):
    l1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    l1 = list_ops.tensor_list_set_item(l1, 1, 1)

    def loop_fn(i):
      l2 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      l2 = list_ops.tensor_list_set_item(l2, 1, i)
      return list_ops.tensor_list_stack(
          math_ops.add_n([l1, l2]), dtypes.int32)

    self._test_loop_fn(loop_fn, 2)


@test_util.run_all_in_graph_and_eager_modes
class TensorTest(PForTestCase):

  def test_loop_variant_scatter_update_no_shape(self):
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
    ])
    def shapeless_func(tensor, indices, updates):
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    def loop_fn(i):
      tensor = [0, 0, 0, 0, 0, 0, 0, 0]
      indices = [[i], [i + 1], [i + 3], [i + 2]]
      updates = [i, i - 10, i + 11, 12]
      return shapeless_func(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_update_singles(self):
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      tensor = [0, 0, 0, 0, 0, 0, 0, 0]
      indices = [[i], [i+1], [i+3], [i+2]]
      updates = [i, i-10, i+11, 12]
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_update_slices(self):
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      tensor = array_ops.zeros([10, 3], dtype=dtypes.int32)
      indices = [[i+2], [4]]
      updates = [[1, i*2, 3], [i+4, i-5, 6]]
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_update_multi_dim_index(self):
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      tensor = array_ops.zeros([10, 3], dtype=dtypes.int32)
      indices = [[i+2, 1], [4, 2]]
      updates = [i, 5]
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_update_folded_indices(self):
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      tensor = array_ops.zeros([5, 5])
      indices = [
          [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
          [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
      ]
      updates = [
          [1, i, 1, 1, 1],
          [1, 1, i+2, 1, i-5],
      ]
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)


class OptionalTest(PForTestCase):

  def test_optional_from_value(self):

    def loop_fn(i):
      o = gen_optional_ops.optional_from_value(
          [i, i + 1, constant_op.constant(3)]
      )
      gen_optional_ops.optional_none()
      return gen_optional_ops.optional_get_value(
          o, [dtypes.int32, dtypes.int32, dtypes.int32], [[], [], []]
      )

    self._test_loop_fn(loop_fn, 2)


class StackTest(PForTestCase):

  @test_util.run_v1_only("b/122612051")
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

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
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

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
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

  @test_util.run_v1_only("b/122612051")
  def test_stack_outside_push(self):
    s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)

    def loop_fn(_):
      return data_flow_ops.stack_push_v2(s, 7)

    with self.assertRaisesRegex(ValueError, "StackPushV2 not allowed.*"):
      pfor_control_flow_ops.pfor(loop_fn, iters=2)


# TODO(agarwal): test nested while_loops. This currently requires converting a
# tf.cond.
class WhileV1Test(PForTestCase):

  def setUp(self):
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.disable_control_flow_v2()
    super(WhileV1Test, self).setUp()

  def tearDown(self):
    if self._enabled:
      control_flow_v2_toggles.enable_control_flow_v2()
    super(WhileV1Test, self).tearDown()

  def test_while_outside_loop(self):

    x = while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    def loop_fn(i):
      return x + i

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_invariant_while(self):

    def loop_fn(_):
      return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_invariant_while_with_control_dependency(self):

    def loop_fn(i):
      with ops.control_dependencies([i]):
        return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_while_with_stateful_ops(self):

    def loop_fn(_):
      return while_loop.while_loop(
          lambda j, x: j < 4, lambda j, x:
          (j + 1, x + random_ops.random_uniform([])), [0, 0.])[0]

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_while_unstacked_condition(self):

    def loop_fn(i):
      return while_loop.while_loop(lambda j, x: j < 4, lambda j, x:
                                   (j + 1, x + i), [0, 0])

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_while(self):
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      _, total = while_loop.while_loop(
          lambda j, _: j < lengths_i, lambda j, t:
          (j + 1, t + array_ops.gather(x_i, j)), [0, 0.])
      return total

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_while_jacobian(self):
    x = random_ops.random_uniform([1, 3])
    y = random_ops.random_uniform([3, 3])

    # out = x @ y @ y @ y @ y, where @ is matmul operator.
    _, out = while_loop.while_loop(
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

  @test_util.run_v1_only("b/122612051")
  def test_tensor_array_as_loop_variable(self):

    def loop_fn(i):

      def body(j, ta):
        ta = ta.write(j, i + j * j)
        return j + 1, ta

      _, ta = while_loop.while_loop(
          lambda j, _: j < 4, body,
          (0, tensor_array_ops.TensorArray(dtypes.int32, size=4)))
      return ta.stack()

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_read_tensor_array_partitioned_indices(self):
    # Note that tensor array values are pfor loop dependent, and the while loop
    # termination condition is also dependent on pfor iteration.
    def loop_fn(i):
      ta = tensor_array_ops.TensorArray(dtypes.int32, size=6)
      ta = ta.unstack(i + list(range(5)))

      def body(j, s):
        return j + 1, s + ta.read(j)

      _, s = while_loop.while_loop(lambda j, _: j < i, body, (0, 0))
      return s

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_external_while_loop_grad(self):
    # Here we test that external while_loops that are extended from inside pfor
    # (due to gradient calls) are not actually converted. If the below was
    # converted all pfor iterations would write to the same tensor array
    # indices.
    x = constant_op.constant(1.)

    def body(j, ta):
      ta = ta.write(j, x)
      return j + 1, ta

    _, ta = while_loop.while_loop(
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

  @test_util.run_v1_only("b/122612051")
  def test_tensor_array_grad(self):
    inp = constant_op.constant(np.random.rand(3, 4, 2), dtype=dtypes.float32)
    ta = tensor_array_ops.TensorArray(dtypes.float32, size=3)
    ta = ta.unstack(inp)

    def loop_fn(i):

      def body(j, x):
        value = ta.gather([j])
        value = array_ops.gather(array_ops.reshape(value, [4, 2]), i)
        return j + 1, x + value

      _, out = while_loop.while_loop(lambda j, _: j < 3, body,
                                     (0, array_ops.zeros([2])))
      out = math_ops.reduce_prod(out)
      return out, gradient_ops.gradients(out, inp)[0]

    pfor_out, pfor_out_grad = pfor_control_flow_ops.pfor(loop_fn, 4)
    # Note that tf.while_loop does not work in the setup above. So we manually
    # construct the equivalent computation of the above loops here.
    real_out = math_ops.reduce_sum(inp, axis=[0])
    real_out = math_ops.reduce_prod(real_out, axis=[1])
    # Note that gradients of real_out will accumulate the gradients across the
    # output value. Hence we do the same aggregation on pfor_out_grad.
    real_out_grad = gradient_ops.gradients(real_out, inp)[0]
    sum_pfor_out_grad = math_ops.reduce_sum(pfor_out_grad, axis=[0])

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
  inputs, sequence_length = dynamic_lstm_input_fn(batch_size, state_size,
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
      new_state = [
          array_ops.where(done, s, ns)
          for s, ns in zip(nest.flatten(state), nest.flatten(new_state))
      ]
      new_state = nest.pack_sequence_as(state, new_state)
      return t + 1, new_state, ta

    def condition_fn(t, _, unused):
      del unused
      return t < max_steps

    initial_state = cell.zero_state(1, dtypes.float32)
    _, state, ta = while_loop.while_loop(condition_fn, body_fn, [
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


@test_util.run_all_in_graph_and_eager_modes
class WhileV2Test(PForTestCase):

  def setUp(self):
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.enable_control_flow_v2()
    super(WhileV2Test, self).setUp()

  def tearDown(self):
    if not self._enabled:
      control_flow_v2_toggles.disable_control_flow_v2()
    super(WhileV2Test, self).tearDown()

  def test_while_outside_loop(self):

    def _f():
      return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    def loop_fn(i):
      return _f() + i

    self._test_loop_fn(loop_fn, 3)

  def test_invariant_while(self):

    def loop_fn(_):
      return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3)

  def test_invariant_while_with_control_dependency(self):

    def loop_fn(i):
      with ops.control_dependencies([i]):
        return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3)

  def test_while_with_stateful_ops(self):

    def loop_fn(_):
      j, _ = while_loop.while_loop(
          lambda j, x: j < 4, lambda j, x:
          (j + 1, x + random_ops.random_uniform([])), [0, 0.])
      return j

    self._test_loop_fn(loop_fn, 3)

  def test_while_with_variable(self):
    v = resource_variable_ops.ResourceVariable(5.)

    def loop_fn(_):
      _, output = while_loop.while_loop(lambda j, x: j < 4, lambda j, x:
                                        (j + 1, x + v), [0, 0.])
      return output

    self._test_loop_fn(loop_fn, 3)

  def test_while_unstacked_condition(self):

    def loop_fn(i):
      return while_loop.while_loop(lambda j, x: j < 4, lambda j, x:
                                   (j + 1, x + i), [0, 0])

    self._test_loop_fn(loop_fn, 3)

  def test_while(self):
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      return while_loop.while_loop(
          lambda j, _: j < lengths_i, lambda j, t:
          (j + 1, t + array_ops.gather(x_i, j)), [0, 0.])

    self._test_loop_fn(loop_fn, 3)

  def test_while_change_input_invariance(self):
    # This tests cases where a loop invariant input to while has loop dependent
    # operations applied to it inside the while body.
    # It also test inputs that are passed through.
    def loop_fn(i):
      return while_loop.while_loop(
          lambda j, *_: j < i, lambda j, x, y, z, w:
          (j + 1, x + i, y + x, z, w), [
              0,
              constant_op.constant(0),
              constant_op.constant(1), i,
              constant_op.constant(2)
          ])

    self._test_loop_fn(loop_fn, 3)

  def test_while_shape_invariants(self):

    def loop_fn(i):
      return while_loop.while_loop(
          lambda j, *_: j < 4,
          lambda j, x, y: (j + 1, x + i, y + 1),
          [0, constant_op.constant([0, 1]),
           constant_op.constant([2, 3])],
          shape_invariants=[
              None,
              tensor_shape.TensorShape([2]),
              tensor_shape.TensorShape([2])
          ])

    self._test_loop_fn(loop_fn, 3)

  def test_while_jacobian(self):
    # Note that we wrap the code below in a tf.function since we don't want the
    # while_loop call to be evaluated eagerly using a python loop.
    @def_function.function
    def _f(x, y, use_pfor):
      # out = x @ y @ y @ y @ y, where @ is matmul operator.
      _, out = while_loop.while_loop(
          lambda i, _: i < 4, lambda i, out: (i + 1, math_ops.matmul(out, y)),
          [0, x])

      def loop_fn(i):
        out_i = array_ops.gather(out, i, axis=1)
        grad = gradient_ops.gradients(out_i, x)
        return array_ops.reshape(grad[0], [-1])

      if use_pfor:
        return pfor_control_flow_ops.pfor(loop_fn, iters=3)
      else:
        return pfor_control_flow_ops.for_loop(
            loop_fn, iters=3, loop_fn_dtypes=out.dtype)

    x = constant_op.constant(np.random.uniform(size=(1, 3)))
    y = constant_op.constant(np.random.uniform(size=(3, 3)))
    self.assertAllClose(_f(x, y, True), _f(x, y, False))

  def test_scan(self):
    np.random.seed(seed=42)
    data = np.random.randn(3).astype(np.float32)

    def log_prob(x):
      return math_ops.reduce_sum(functional_ops.scan_v2(
          lambda _, yi: (x - yi)**2,
          elems=data,
          initializer=constant_op.constant(0.)))

    x = variables.Variable(array_ops.ones([2]))
    self.evaluate(x.initializer)
    v_log_prob = lambda x: pfor_control_flow_ops.vectorized_map(log_prob, x)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        v_log_prob, (x,), delta=1e-3)
    self.assertAllClose(theoretical, numerical, rtol=1e-2)

  def test_scan_captured_variable(self):
    if not context.executing_eagerly():
      self.skipTest("Test only written for 2.x")
    v = variables.Variable(math_ops.range(10, dtype=dtypes.float32))

    def loop_fn(idx):
      del idx
      return functional_ops.scan_v2(lambda _, i: array_ops.gather(v, i),
                                    elems=math_ops.range(v.shape[0]),
                                    initializer=0.0)
    with backprop.GradientTape() as tape:
      result = pfor_control_flow_ops.pfor(loop_fn, 2)
    self.assertAllClose([2.] * 10, tape.gradient(result, v))


@test_util.run_all_in_graph_and_eager_modes
class NestedControlFlowTest(PForTestCase):

  def setUp(self):
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.enable_control_flow_v2()
    super(NestedControlFlowTest, self).setUp()

  def tearDown(self):
    if not self._enabled:
      control_flow_v2_toggles.disable_control_flow_v2()
    super(NestedControlFlowTest, self).tearDown()

  def _cond(self, f=None, split=0):
    if f is None:
      f = lambda x, y: (x, y)

    def _f(x, y):
      return cond.cond(y > split, lambda: f(x, y), lambda:
                       (x + 1., y))

    return _f

  def _while(self, f=None):
    if f is None:
      f = lambda x, y: (x, y)

    def _f(x, y):
      return while_loop.while_loop(
          lambda j, _: j < y, lambda j, t:
          (j + 1, t + array_ops.gather(f(x, y)[0], j)), [0, x])[1], y

    return _f

  def _test_helper(self, f):
    x = random_ops.random_uniform([5, 5])
    y = constant_op.constant([4, -1, 2, -2, 2])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      return f(x_i, y_i)

    self._test_loop_fn(loop_fn, 5)

  def test_cond_while(self):
    self._test_helper(self._cond(self._while()))

  def test_while_cond(self):
    self._test_helper(self._while(self._cond()))

  def test_while_while(self):
    self._test_helper(self._while(self._while()))

  def test_cond_cond(self):
    self._test_helper(self._cond(self._cond()))


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class StatelessIfTest(PForTestCase):

  def test_loop_variant_cond(self):
    x = [1, 2, 3, 4, 5.]
    y = 2.5

    @def_function.function
    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      # Note that the output has a combination of then and else branches being
      # loop variant / invariant.
      return cond_v2.cond_v2(x_i < y, lambda: (y - x_i, y, 1., 2.), lambda:
                             (x_i - y, 0., y, 3.))

    self._test_loop_fn(loop_fn, iters=5)

  def test_loop_invariant_cond(self):
    x = [1, 2, 3, 4, 5.]
    y = 0.5
    z = random_ops.random_uniform([])

    @def_function.function
    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      # Note that the output has a combination of then and else branches being
      # loop variant / invariant.
      return cond_v2.cond_v2(z < y, lambda: (y - x_i, y, 1., 2.), lambda:
                             (x_i - y, 0., y, 3.))

    self._test_loop_fn(loop_fn, iters=5)

  def test_empty_branch(self):
    x = [1, 2, 3, 4, 5.]
    y = 6.

    @def_function.function
    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return cond_v2.cond_v2(
          x_i < y,  # Note that else branch is empty.
          lambda: (y - x_i, y, 1., 2.),
          lambda: (x_i - y, 0., y, 3.))

    self._test_loop_fn(loop_fn, iters=5)


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class IfTest(PForTestCase):

  def test_read_var(self):
    self.skipTest("b/156438918")  # Flaky

    x = [1, 2, 3, 4, 5.]
    y = 2.5
    z = resource_variable_ops.ResourceVariable(5.)

    @def_function.function
    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return cond_v2.cond_v2(x_i < y, lambda: z - x_i, lambda: z + x_i)

    self._test_loop_fn(loop_fn, iters=5)


class RNNTest(PForTestCase):

  @test_util.run_v1_only("b/122612051")
  def test_dynamic_rnn(self):
    pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicRNNCell, 3, 5,
                                                   7)
    self.run_and_assert_equal(pfor_outputs, tf_outputs)

  @test_util.run_v1_only("b/122612051")
  def test_dynamic_lstm(self):
    pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicLSTMCell, 3, 5,
                                                   7)
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
      run_fn = sess.make_callable(targets)
      run_fn()  # Warm up
      begin = time.time()
      for _ in range(iters):
        run_fn()
      end = time.time()
    avg_time_ms = 1000 * (end - begin) / iters
    self.report_benchmark(iters=iters, wall_time=avg_time_ms, name=name)
    return avg_time_ms

  def benchmark_sess_run_overhead(self):
    with ops.Graph().as_default():
      x = constant_op.constant(1.0)
      self._run(x, 10000, name="session_run_overhead")

  def benchmark_add(self):
    with ops.Graph().as_default():
      n = 256
      params = 1000
      x = random_ops.random_normal([n, params])
      y = random_ops.random_normal([n, params])

      def loop_fn(i):
        x_i = array_ops.gather(x, i)
        y_i = array_ops.gather(y, i)
        return x_i + y_i

      pfor_outputs = pfor_control_flow_ops.pfor(loop_fn, n)
      while_outputs = pfor_control_flow_ops.for_loop(loop_fn, dtypes.float32, n)
      manual = x + y

      self._run(manual, 1000, name="manual_add")
      self._run(pfor_outputs, 1000, name="pfor_add")
      self._run(while_outputs, 100, name="while_add")

  def benchmark_matmul(self):
    with ops.Graph().as_default():
      n = 1024
      params = 1000
      x = random_ops.random_normal([n, params])
      y = random_ops.random_normal([params, params])

      def loop_fn(i):
        x_i = array_ops.expand_dims(array_ops.gather(x, i), 0)
        return math_ops.matmul(x_i, y)

      pfor_outputs = pfor_control_flow_ops.pfor(loop_fn, n)
      while_outputs = pfor_control_flow_ops.for_loop(loop_fn, dtypes.float32, n)
      manual = math_ops.matmul(x, y)

      self._run(manual, 1000, name="manual_matmul")
      self._run(pfor_outputs, 1000, name="pfor_matmul")
      self._run(while_outputs, 100, name="while_matmul")

  def benchmark_map_fn(self):
    with ops.Graph().as_default():
      b = 256
      params = 1000
      inp = random_ops.random_normal((b, params))
      fn = lambda x: x * x

      def pfor_map_fn(f, x):
        return pfor_control_flow_ops.pfor(lambda i: f(array_ops.gather(x, i)),
                                          array_ops.shape(x)[0])

      map_output = map_fn.map_fn(fn, inp)
      pfor_output = pfor_map_fn(fn, inp)

      self._run(map_output, 100, name="tf_map_fn")
      self._run(pfor_output, 100, name="pfor_map_fn")

  def benchmark_basic_while(self):
    with ops.Graph().as_default():

      def loop_fn(i):
        _, s = while_loop.while_loop(lambda t, x: t < i, lambda t, x:
                                     (t + 1, x + i), [0, 0])
        return s

      iters = 50
      pfor_output = pfor_control_flow_ops.pfor(loop_fn, iters)
      for_loop_output = pfor_control_flow_ops.for_loop(loop_fn, dtypes.int32,
                                                       iters)
      self._run(pfor_output, 100, name="pfor_basic")
      self._run(for_loop_output, 100, name="for_loop_basic")

  def benchmark_dynamic_rnn(self):
    with ops.Graph().as_default():
      pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicRNNCell, 128,
                                                     512, 16)
      self._run(pfor_outputs, 100, name="pfor_rnn")
      self._run(tf_outputs, 100, name="tf_rnn")

  def benchmark_reduction(self):
    n = 1024
    with ops.Graph().as_default():
      x = random_ops.random_uniform([n, n])
      w = random_ops.random_uniform([n, n])

      def loop_fn(i, pfor_config):
        x_i = array_ops.gather(x, i)
        return math_ops.reduce_sum(
            math_ops.matmul(pfor_config.reduce_concat(x_i), w))

      # Note that output_reduction will be tiled, so there may be some minor
      # overheads compared to output_no_reduction.
      output_reduction = pfor_control_flow_ops.pfor(loop_fn, n)
      output_no_reduction = math_ops.reduce_sum(math_ops.matmul(x, w))
      # Benchmark to test that reduction does not add overhead and its output is
      # treated as loop invariant.
      self._run(output_reduction, 30, name="matmul_reduction")
      self._run(output_no_reduction, 30, name="matmul_no_reduction")


class SparseTest(PForTestCase):

  @test_util.run_v1_only("b/122612051")
  def test_var_loop_len(self):
    num_iters = array_ops.placeholder(dtypes.int32)

    def loop_fn(_):
      return sparse_tensor.SparseTensor([[0], [1], [2]], [4, 5, 6],
                                        [3])  # [0, 2, 0]

    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    with self.cached_session() as sess:
      sess.run(pfor, feed_dict={num_iters: 3})

  @test_util.run_v1_only("b/122612051")
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

  @test_util.run_v1_only("b/122612051")
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

  @test_util.run_v1_only("b/122612051")
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

  @test_util.run_v1_only("b/122612051")
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

  @test_util.run_v1_only("b/122612051")
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

  @test_util.run_v1_only("b/122612051")
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


# Dummy CompositeTensor to test CompositeTensor support.
class Particle(composite_tensor.CompositeTensor):
  """A (batch of) particles each defined by a mass and a scalar velocity."""

  def __init__(self, mass, velocity):
    mass = ops.convert_to_tensor(mass)
    velocity = ops.convert_to_tensor(velocity)
    self.shape = array_ops.broadcast_static_shape(mass.shape, velocity.shape)
    self.mass = mass
    self.velocity = velocity

  @property
  def _type_spec(self):
    return ParticleSpec(
        type_spec.type_spec_from_value(self.mass),
        type_spec.type_spec_from_value(self.velocity))


class ParticleSpec(type_spec.BatchableTypeSpec):

  def __init__(self, mass, velocity):
    self.shape = array_ops.broadcast_static_shape(
        mass.shape, velocity.shape)
    self.mass = mass
    self.velocity = velocity

  def _serialize(self):
    return (self.mass, self.velocity)

  @property
  def value_type(self):
    return Particle

  @property
  def _component_specs(self):
    return (self.mass, self.velocity)

  def _to_components(self, value):
    return (value.mass, value.velocity)

  def _from_components(self, components):
    return Particle(*components)

  def _pad_shape_to_full_rank(self, s):
    """Pad component shapes with 1's so all components have the same rank."""
    return tensor_shape.TensorShape(
        [1] * (self.shape.ndims - s.ndims)).concatenate(s)

  def _batch(self, batch_size):
    return ParticleSpec(
        mass=tensor_spec.TensorSpec(
            dtype=self.mass.dtype,
            shape=tensor_shape.TensorShape([batch_size]).concatenate(
                self._pad_shape_to_full_rank(self.mass.shape))),
        velocity=tensor_spec.TensorSpec(
            dtype=self.velocity.dtype,
            shape=tensor_shape.TensorShape([batch_size]).concatenate(
                self._pad_shape_to_full_rank(self.velocity.shape))))

  def _unbatch(self):
    return ParticleSpec(
                tensor_spec.TensorSpec(dtype=self.mass.dtype,
                                       shape=self.mass.shape[1:]),
                tensor_spec.TensorSpec(dtype=self.velocity.dtype,
                                       shape=self.velocity.shape[1:]))

  def _to_tensor_list(self, value):
    return [array_ops.reshape(
                value.mass,
                self._pad_shape_to_full_rank(value.mass.shape)),
            array_ops.reshape(
                value.velocity,
                self._pad_shape_to_full_rank(value.velocity.shape))]


class CompositeTensorTest(PForTestCase, parameterized.TestCase):

  @parameterized.parameters((None,), (3,))
  def test_create_composite_inside_loop(self, parallel_iterations):
    num_particles = 10
    velocities = random_ops.random_uniform([num_particles])
    particles = pfor_control_flow_ops.pfor(
        # Build a batch of particles all with the same mass.
        lambda i: Particle(mass=4., velocity=array_ops.gather(velocities, i)),
        num_particles,
        parallel_iterations=parallel_iterations)
    particles_mass, particles_velocity, velocities = self.evaluate(
        (particles.mass, particles.velocity, velocities))
    self.assertAllEqual(particles_mass, 4. * np.ones([num_particles]))
    self.assertAllEqual(particles_velocity, velocities)

  @parameterized.parameters((None,), (3,))
  def test_composite_is_converted_to_batched_tensor(
      self, parallel_iterations):
    particles = pfor_control_flow_ops.pfor(
        lambda _: Particle(mass=random_ops.random_uniform([3]),  # pylint: disable=g-long-lambda
                           velocity=random_ops.random_uniform([5, 3])),
        4,
        parallel_iterations=parallel_iterations)
    # Naively batching the component shapes would give `[4, 3]` and `[4, 5, 3]`
    # which have no consistent broadcast shape.
    self.assertEqual(particles.mass.shape, [4, 1, 3])
    self.assertAllEqual(particles.velocity.shape, [4, 5, 3])

  def test_vectorized_map_gathers_composite_tensors(self):
    particles = Particle(mass=[1., 2., 3., 4., 5.],
                         velocity=[1., 2., 3., 4., 5.])
    self.assertAllEqual(
        pfor_control_flow_ops.vectorized_map(
            lambda x: x.mass * x.velocity, particles),
        particles.mass * particles.velocity)

  def test_vectorized_map_of_ragged_tensors(self):
    # Vmap should be able to handle ragged Tensors as long as they're not
    # *actually* ragged.
    ragged = ragged_tensor.RaggedTensor.from_uniform_row_length(
        ragged_tensor.RaggedTensor.from_row_lengths(
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            row_lengths=[3, 3, 3, 3]),
        uniform_row_length=2)  # Overall shape [2, 2, 3].
    self.assertAllEqual(
        pfor_control_flow_ops.vectorized_map(
            lambda x: x.to_tensor(shape=[2, 3]), ragged),
        ragged.to_tensor(shape=[2, 2, 3]))


class ParsingTest(PForTestCase):

  def test_decode_csv(self):
    csv_tensor = constant_op.constant([["1:2:3"], ["::"], ["7:8:9"]])
    kwargs = {"record_defaults": [[10], [20], [30]], "field_delim": ":"}

    def loop_fn(i):
      line = array_ops.gather(csv_tensor, i)
      return parsing_ops.decode_csv(line, **kwargs)

    self._test_loop_fn(loop_fn, iters=3)

  @test_util.run_v1_only("b/122612051")
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


class PartitionedCallTest(PForTestCase):

  def test_simple(self):

    @def_function.function
    def f(x):
      return math_ops.square(x) + 1

    z = random_ops.random_uniform([4])

    def loop_fn(i):
      return f(array_ops.gather(z, i))

    self._test_loop_fn(loop_fn, 4)

  def test_nested_calls(self):

    @def_function.function
    def inner(x):
      return math_ops.square(x)

    @def_function.function
    def outer(y):
      return math_ops.reduce_sum(inner(y)) + 2

    z = random_ops.random_uniform([4, 2])

    def loop_fn(i):
      return outer(array_ops.gather(z, i))

    self._test_loop_fn(loop_fn, 4)

  def test_nested_calls_loop_fn_autograph(self):
    #TODO (@bhack) Do we need to extend the coverage?

    def loop_fn(x):
      for y in range(array_ops.constant(3)):
        pass
      return math_ops.square(x)

    @def_function.function
    def loop_fn_caller():
      self._test_loop_fn(loop_fn, 4)

    loop_fn_caller()


  def test_nested_definition(self):

    @def_function.function
    def outer(y):

      @def_function.function
      def inner(x):
        return math_ops.square(x) + 1

      return math_ops.reduce_sum(inner(y)) + 2

    z = random_ops.random_uniform([4, 2])

    def loop_fn(i):
      return outer(array_ops.gather(z, i))

    self._test_loop_fn(loop_fn, 4)

  def test_gradients(self):

    @def_function.function
    def f(x):
      return math_ops.square(x) + 1

    z = random_ops.random_uniform([4, 2])

    def loop_fn(i):
      z_i = array_ops.gather(z, i)
      with backprop.GradientTape() as g:
        g.watch(z_i)
        out = f(z_i)
      return out, g.gradient(out, z_i)

    self._test_loop_fn(loop_fn, 4)

  def test_stateful_with_gradients(self):

    z = random_ops.random_uniform([4, 2])
    v = variables.Variable(z[0])

    @def_function.function
    def f(x):
      return math_ops.square(x) + v + 1

    def loop_fn(i):
      z_i = array_ops.gather(z, i)
      with backprop.GradientTape() as g:
        g.watch(z_i)
        out = f(z_i)
      return out, g.gradient(out, z_i)

    self._test_loop_fn(loop_fn, 4)


class SpectralTest(PForTestCase, parameterized.TestCase):

  @parameterized.parameters(
      (fft_ops.fft,),
      (fft_ops.fft2d,),
      (fft_ops.fft3d,),
      (fft_ops.ifft,),
      (fft_ops.ifft2d,),
      (fft_ops.ifft3d,),
  )
  def test_fft(self, op_func):
    shape = [2, 3, 4, 3, 4]
    x = np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return op_func(x_i)

    self._test_loop_fn(loop_fn, 2)

  @parameterized.parameters(
      (fft_ops.rfft,),
      (fft_ops.rfft2d,),
      (fft_ops.rfft3d,),
  )
  def test_rfft(self, op_func):
    for dtype in (dtypes.float32, dtypes.float64):
      x = random_ops.random_uniform([2, 3, 4, 3, 4], dtype=dtype)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        x_i = array_ops.gather(x, i)
        return op_func(x_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)

  @parameterized.parameters(
      (fft_ops.irfft,),
      (fft_ops.irfft2d,),
      (fft_ops.irfft3d,),
  )
  def test_irfft(self, op_func):
    if config.list_physical_devices("GPU"):
      # TODO(b/149957923): The test is flaky
      self.skipTest("b/149957923: irfft vectorization flaky")
    for dtype in (dtypes.complex64, dtypes.complex128):
      shape = [2, 3, 4, 3, 4]
      x = np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)
      x = math_ops.cast(x, dtype=dtype)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        x_i = array_ops.gather(x, i)
        return op_func(x_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)


class VariableTest(PForTestCase):

  def test_create_variable_once(self):
    x = array_ops.ones(shape=(3, 2, 2), dtype=dtypes.float32)
    y = array_ops.ones(shape=(2, 3), dtype=dtypes.float32)
    a_var = []

    def f(z):
      if not a_var:
        a_var.append(variables.Variable(lambda: y, name="a"))
      return math_ops.matmul(z, a_var[0] / 16)

    pfor_control_flow_ops.vectorized_map(f, x)

  @test_util.run_v2_only
  def test_create_variable_repeated(self):
    x = array_ops.ones(shape=(3, 2, 2), dtype=dtypes.float32)
    y = array_ops.ones(shape=(2, 3), dtype=dtypes.float32)

    def f(z):
      a_var = variables.Variable(lambda: y, name="a") / 4
      return math_ops.matmul(z, a_var / 16)

    # Note that this error is only raised under v2 behavior.
    with self.assertRaisesRegex(
        ValueError, "singleton tf.Variable.*on the first call"):
      pfor_control_flow_ops.vectorized_map(f, x)

  @test_util.run_all_in_graph_and_eager_modes
  def test_variable_shape(self):
    v = resource_variable_ops.ResourceVariable([1, 2])

    def loop_fn(_):
      return resource_variable_ops.variable_shape(v.handle)

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_all_in_graph_and_eager_modes
  def test_variable_input(self):
    v = resource_variable_ops.ResourceVariable([1, 2])
    self.evaluate(v.initializer)

    def loop_fn(x):
      return x + 1

    result = pfor_control_flow_ops.vectorized_map(loop_fn, v)
    expected_result = [2, 3]
    self.assertAllEqual(result, expected_result)

  @test_util.run_all_in_graph_and_eager_modes
  def testStatelessCase(self):

    def branch1(x):
      return x

    def branch2(x):
      return x + 1

    def branch3(x):
      return x + 2

    x = constant_op.constant(10)
    elems = constant_op.constant([1, 0, 0, 0, 2, 1, 0, 2, 0, 1])
    def loop_fn(z_i):
      return cond_v2.indexed_case(
          z_i, [lambda: branch1(x), lambda: branch2(x), lambda: branch3(x)])

    result = pfor_control_flow_ops.vectorized_map(
        loop_fn, elems, fallback_to_while_loop=False)

    expected_result = [11, 10, 10, 10, 12, 11, 10, 12, 10, 11]
    self.assertAllEqual(result, expected_result)

  @test_util.run_all_in_graph_and_eager_modes
  def testStatelessCaseUnstacked(self):

    def branch1(x):
      return x + 1

    def branch2(x):
      return x + 2

    # Unstacked case input
    case_input = constant_op.constant(1)
    @def_function.function
    def function(z_i):
      return cond_v2.indexed_case(case_input,
                                  [lambda: branch1(z_i), lambda: branch2(z_i)])

    inputs = constant_op.constant([0, 1, 1, 0, 1, 0, 1, 0, 0])

    result = pfor_control_flow_ops.vectorized_map(
        function, inputs, fallback_to_while_loop=False)
    expected_result = [2, 3, 3, 2, 3, 2, 3, 2, 2]
    self.assertAllEqual(result, expected_result)

if __name__ == "__main__":
  test.main()
