# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.math_ops.matmul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test as test_lib

_MAXDIM = 5


def _add_test(test, test_name, fn):
  test_name = "_".join(["test", test_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class TensordotTest(test_lib.TestCase):

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def test_invalid_shape(self):
    a = [[1, 2], [3, 4]]
    b = [[1, 2], [3, 4], [5, 6]]
    a_axes = [1]
    b_axes = [0]
    # Invalid static shapes.
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      math_ops.tensordot(a, b, (a_axes, b_axes))

    # Invalid dynamic shapes.
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      with self.assertRaisesOpError(
          r"In\[0\] mismatch In\[1\] shape: 2 vs\. 3: \[2,2\] \[3,2\]"):
        a_ph = array_ops.placeholder(dtypes.float32)
        b_ph = array_ops.placeholder(dtypes.float32)
        axes_ph = array_ops.placeholder(dtypes.int32)
        output = math_ops.tensordot(a_ph, b_ph, axes_ph)
        _ = sess.run(
            [output], feed_dict={
                a_ph: a,
                b_ph: b,
                axes_ph: (a_axes, b_axes)
            })

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def test_invalid_axes(self):
    a = [[1, 2], [3, 4]]
    b = [[1, 2], [3, 4]]
    # Invalid static axes.
    for axes_value in -1, 3, [1], [[1]], [[1], [0, 1]]:
      with self.assertRaises(ValueError):
        math_ops.tensordot(a, b, axes_value)

    with self.assertRaises(IndexError):
      math_ops.tensordot(a, b, [[0], [7]])

    if context.executing_eagerly():
      return
    # Invalid dynamic axes.
    a_ph = array_ops.placeholder(dtypes.float32)
    b_ph = array_ops.placeholder(dtypes.float32)
    axes_ph = array_ops.placeholder(dtypes.int32)
    output = math_ops.tensordot(a_ph, b_ph, axes_ph)
    # Note: We don't support scalar Tensor values for axes.
    for axes_value in 1, [1], [0, 1], [[1]], [[0, 1]], [[0], [7]]:
      with self.cached_session() as sess:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          _ = sess.run(
              [output], feed_dict={
                  a_ph: a,
                  b_ph: b,
                  axes_ph: axes_value
              })

  # Test case for https://github.com/tensorflow/tensorflow/issues/11950
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def test_valid_axis(self):
    for axes_value in [1, 2], [[1], [2]], [[], []], 0:
      np_a = np.ones((3, 3))
      np_b = np.array([2, 3, 1])[None, None]
      np_ans = np.tensordot(np_a, np_b, axes_value)

      tf_a = array_ops.ones((3, 3), dtype=dtypes.float32)
      tf_b = constant_op.constant([2, 3, 1], dtype=dtypes.float32)[None, None]
      tf_ans = math_ops.tensordot(tf_a, tf_b, axes_value)

      self.assertAllEqual(tf_ans.shape, np_ans.shape)
      self.assertAllEqual(self.evaluate(tf_ans), np_ans)

  @test_util.run_v1_only("Shape inference test")
  def test_partial_shape_inference(self):
    for axes in ([1], [0]), 1:
      a = array_ops.placeholder(dtypes.float32)
      b = array_ops.placeholder(dtypes.float32)
      output = math_ops.tensordot(a, b, axes)
      self.assertEqual(output.get_shape().ndims, None)
      a.set_shape([None, 2])
      b.set_shape([2, 3])
      output = math_ops.tensordot(a, b, axes)
      output_shape = output.get_shape()
      self.assertEqual(output_shape.ndims, 2)
      output_shape = output_shape.as_list()
      self.assertEqual(output_shape[0], None)
      self.assertEqual(output_shape[1], 3)
      a = array_ops.placeholder(dtypes.float32)
      b = array_ops.placeholder(dtypes.float32)
      a.set_shape([2, 2])
      b.set_shape([2, None])
      output = math_ops.tensordot(a, b, axes)
      output_shape = output.get_shape()
      self.assertEqual(output_shape.ndims, 2)
      output_shape = output_shape.as_list()
      self.assertEqual(output_shape[0], 2)
      self.assertEqual(output_shape[1], None)


def _get_tensordot_tests(dtype_, rank_a_, rank_b_, num_dims_, dynamic_shape_):

  # Select a random subset of size m from [0, 1, ..., n-1].
  def _random_subset(m, n):
    assert m <= n
    return (np.random.permutation(n)[:m]).astype(np.int32)

  def _generate_random_tensors_and_dims():
    a_shape = np.random.random_integers(1, _MAXDIM, rank_a_)
    b_shape = np.random.random_integers(1, _MAXDIM, rank_b_)
    shared_shape = np.random.random_integers(1, _MAXDIM, num_dims_)
    a_dims = _random_subset(num_dims_, rank_a_)
    b_dims = _random_subset(num_dims_, rank_b_)
    for i in range(num_dims_):
      a_shape[a_dims[i]] = shared_shape[i]
      b_shape[b_dims[i]] = shared_shape[i]
    a = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(a_shape)).reshape(a_shape).astype(dtype_)
    b = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(b_shape)).reshape(b_shape).astype(dtype_)
    return a, b, a_dims, b_dims

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32("Tests tensordot, which calls matmul")
  def test_tensordot(self):
    if dynamic_shape_ and context.executing_eagerly():
      self.skipTest("Placeholders not support in eager mode")
    num_trials = min(30, num_dims_ * num_dims_)
    if dtype_ == np.float16:
      tol = 0.05
    elif dtype_ == np.float32 or dtype_ == np.complex64:
      tol = 1e-5
    else:
      tol = 1e-12
    for _ in range(num_trials):
      a_np, b_np, a_dims_np, b_dims_np = _generate_random_tensors_and_dims()
      np_ans = np.tensordot(a_np, b_np, axes=(a_dims_np, b_dims_np))
      with self.cached_session() as sess:
        if dynamic_shape_:
          a = array_ops.placeholder(dtype_)
          b = array_ops.placeholder(dtype_)
          axes = array_ops.placeholder(dtypes.int32)
          c = math_ops.tensordot(a, b, axes)
          tf_ans = sess.run(
              c, feed_dict={
                  a: a_np,
                  b: b_np,
                  axes: (a_dims_np, b_dims_np)
              })
        else:
          tf_ans = math_ops.tensordot(a_np, b_np, (a_dims_np, b_dims_np))
      self.assertAllClose(tf_ans, np_ans, rtol=tol, atol=tol)
      self.assertAllEqual(tf_ans.shape, np_ans.shape)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32("Tests tensordot, which calls matmul")
  def test_tensordot_scalar_axes(self):
    if dynamic_shape_ and context.executing_eagerly():
      self.skipTest("Placeholders not support in eager mode")
    if num_dims_ < 1:
      self.skipTest("Not a test")
    if dtype_ == np.float16:
      tol = 0.05
    elif dtype_ == np.float32 or dtype_ == np.complex64:
      tol = 1e-5
    else:
      tol = 1e-12
    shape = [5] * num_dims_
    a_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype_)
    b_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype_)
    all_axes = [0, 1]
    if a_np.ndim > 2:
      all_axes.append(a_np.ndim - 1)
    for axes in all_axes:
      np_ans = np.tensordot(a_np, b_np, axes=axes)
      with self.cached_session() as sess:
        if dynamic_shape_:
          a = array_ops.placeholder(dtype_)
          b = array_ops.placeholder(dtype_)
          c = math_ops.tensordot(a, b, axes=axes)
          tf_ans = sess.run(c, feed_dict={a: a_np, b: b_np})
        else:
          tf_ans = math_ops.tensordot(a_np, b_np, axes=axes)
      self.assertAllClose(tf_ans, np_ans, rtol=tol, atol=tol)
      self.assertAllEqual(tf_ans.shape, np_ans.shape)

  return [test_tensordot, test_tensordot_scalar_axes]


if __name__ == "__main__":
  dtypes_to_test = [
      np.float16, np.float32, np.float64, np.complex64, np.complex128
  ]
  for dtype in dtypes_to_test:
    for rank_a in 1, 2, 4, 5:
      for rank_b in 1, 2, 4, 5:
        for num_dims in range(0, min(rank_a, rank_b) + 1):
          # TF2 does not support placeholders under eager so we skip it
          for dynamic_shape in set([False, True]):
            for testcase in _get_tensordot_tests(dtype, rank_a, rank_b,
                                                 num_dims, dynamic_shape):
              name = "%s_%s_%s_%s_%s_%s" % (testcase.__name__, dtype.__name__,
                                            rank_a, rank_b, num_dims,
                                            dynamic_shape)
              _add_test(TensordotTest, name, testcase)
  test_lib.main()
