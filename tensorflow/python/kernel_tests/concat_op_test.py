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
"""Functional tests for Concat Op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@test_util.disable_all_xla("This test never passed for XLA")
class ConcatOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testHStack(self):
    with self.session(use_gpu=True):
      p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      c = array_ops.concat([p1, p2], 0)
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:4, :], params[p1])
    self.assertAllEqual(result[4:, :], params[p2])

  @test_util.run_deprecated_v1
  def testVStack(self):
    with self.session(use_gpu=True):
      p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      c = array_ops.concat([p1, p2], 1)
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:, :4], params[p1])
    self.assertAllEqual(result[:, 4:], params[p2])

  def testInt32GPU(self):
    with test_util.use_gpu():
      p1 = np.random.rand(2, 3).astype("i")
      p2 = np.random.rand(2, 3).astype("i")
      x1 = constant_op.constant(p1)
      x2 = constant_op.constant(p2)
      c = array_ops.concat([x1, x2], 0)
      result = self.evaluate(c)
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def testRefType(self):
    with test_util.use_gpu():
      p1 = np.random.rand(4, 4).astype("f")
      p2 = np.random.rand(4, 4).astype("f")
      v1 = variables.Variable(p1)
      v2 = variables.Variable(p2)
      c = array_ops.concat([v1, v2], 0)
      self.evaluate(variables.global_variables_initializer())
      result = self.evaluate(c)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:4, :], p1)
    self.assertAllEqual(result[4:, :], p2)

  def _testRandom(self, dtype):
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    # Random number of tensors, but always > 1.
    num_tensors = np.random.randint(2, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    params = {}
    if dtype == dtypes.bfloat16:
      dtype_feed = dtypes.float32
    else:
      dtype_feed = dtype
    with self.session(use_gpu=True):
      p = []
      for i in np.arange(num_tensors):
        input_shape = shape
        input_shape[concat_dim] = np.random.randint(1, 5)
        placeholder = array_ops.placeholder(dtype_feed, shape=input_shape)
        p.append(placeholder)

        t = dtype_feed.as_numpy_dtype
        params[placeholder] = np.random.rand(*input_shape).astype(t)

      if dtype != dtype_feed:
        concat_inputs = [math_ops.cast(p_i, dtype) for p_i in p]
      else:
        concat_inputs = p
      c = array_ops.concat(concat_inputs, concat_dim)
      if dtype != dtype_feed:
        c = math_ops.cast(c, dtype_feed)
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    cur_offset = 0

    for i in np.arange(num_tensors):
      # The index into the result is the ':' along all dimensions
      # except the concat_dim. slice(0, size) is used for ':', and
      # a list of slices is used to index into result.
      ind = [slice(0, params[p[i]].shape[j]) for j in np.arange(5)]
      ind[concat_dim] = slice(cur_offset,
                              cur_offset + params[p[i]].shape[concat_dim])
      cur_offset += params[p[i]].shape[concat_dim]
      if dtype == dtype_feed:
        self.assertAllEqual(result[ind], params[p[i]])
      else:
        self.assertAllClose(result[ind], params[p[i]], 0.01)

  @test_util.run_deprecated_v1
  def testRandom(self):
    self._testRandom(dtypes.bool)
    self._testRandom(dtypes.float32)
    self._testRandom(dtypes.int16)
    self._testRandom(dtypes.int32)
    self._testRandom(dtypes.int64)
    self._testRandom(dtypes.bfloat16)
    self._testRandom(dtypes.complex64)
    self._testRandom(dtypes.complex128)

  @test_util.run_deprecated_v1
  def testInvalidConcatDimTypeAndShape(self):
    a = variables.Variable(constant_op.constant(1.0, shape=[1]))
    b = variables.Variable(constant_op.constant(2.0, shape=[1]))
    with self.assertRaises(ValueError):
      array_ops.concat(b, a)
    with self.assertRaises(TypeError):
      array_ops.concat(1, 4.2)
    with self.assertRaises(ValueError):
      array_ops.concat(1, a)
    with self.assertRaises(TypeError):
      array_ops.concat([a, b], a)
    with self.assertRaises(ValueError):
      array_ops.concat([a, b], [3])
    with self.assertRaises(ValueError):
      array_ops.concat([], 0)
    # An integer tensor for shape dim should throw no error.
    array_ops.concat(1, constant_op.constant(0, shape=[]))
    # A non-scalar tensor for shape should throw ValueError.
    with self.assertRaises(ValueError):
      array_ops.concat(1, constant_op.constant(0, shape=[1]))

  def _testGradientsSimple(self, dtype):
    # Test both positive and negative concat axis.
    # -2 and 1 correspond to the same axis for 3-dimensional tensors.
    for axis in [-2, 1]:
      with test_util.use_gpu():
        inp = []
        inp_tensors = []
        for x in [1, 2, 6]:
          shape = [10, x, 2]
          t = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
          if dtype.is_complex:
            t += -1j * t
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  t.flatten(),
                  shape=shape,
                  dtype=dtype))
        c = array_ops.concat(inp_tensors, axis)
        output_shape = [10, 9, 2]
        grad_inp = np.random.rand(*output_shape).astype(dtype.as_numpy_dtype)
        if dtype.is_complex:
          grad_inp += -1j * grad_inp
        grad_tensor = constant_op.constant(
            grad_inp.flatten(), shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, axis)
        result = self.evaluate(concated_grad)
    self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsSimple(self):
    self._testGradientsSimple(dtypes.float32)
    self._testGradientsSimple(dtypes.complex64)

  @test_util.run_deprecated_v1
  def testGradientsFirstDim(self):
    with test_util.use_gpu():
      inp = []
      inp_tensors = []
      for x in [1, 2, 6]:
        shape = [x, 10, 2]
        t = np.random.rand(*shape).astype("f")
        inp.append(t)
        inp_tensors.append(
            constant_op.constant(
                t.flatten(),
                shape=shape,
                dtype=dtypes.float32))
      c = array_ops.concat(inp_tensors, 0)
      output_shape = [9, 10, 2]
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(
          grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.concat(grad, 0)
      result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsLastDim(self):
    # Test both positive and negative concat axis.
    # -1 and 2 correspond to the same axis for 3-dimensional tensors.
    for axis in [-1, 2]:
      with test_util.use_gpu():
        inp = []
        inp_tensors = []
        for x in [1, 2, 6]:
          shape = [10, 2, x]
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  t.flatten(),
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, 2)
        output_shape = [10, 2, 9]
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            grad_inp.flatten(), shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, axis)
        result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  def _RunAndVerifyGradientsRandom(self):
    # Random dims of rank 5
    input_shape = np.random.randint(1, 5, size=5)
    # Random number of tensors
    num_tensors = np.random.randint(12, 20)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
    with test_util.use_gpu():
      inp = []
      inp_tensors = []
      for x in concat_dim_sizes:
        shape = input_shape
        shape[concat_dim] = x
        t = np.random.rand(*shape).astype("f")
        inp.append(t)
        inp_tensors.append(
            constant_op.constant(t.flatten(), shape=shape,
                                 dtype=dtypes.float32))
      c = array_ops.concat(inp_tensors, concat_dim)
      output_shape = input_shape
      output_shape[concat_dim] = concat_dim_sizes.sum()
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.concat(grad, concat_dim)
      result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsRandom(self):
    for _ in range(5):
      self._RunAndVerifyGradientsRandom()

  @test_util.run_deprecated_v1
  def testGradientWithUnknownInputDim(self):
    with self.session(use_gpu=True):
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.placeholder(dtypes.float32)
      c = array_ops.concat([x, y], 2)

      output_shape = [10, 2, 9]
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(
          grad_inp.flatten(), shape=output_shape)

      grad = gradients_impl.gradients([c], [x, y], [grad_tensor])
      concated_grad = array_ops.concat(grad, 2)
      params = {
          x: np.random.rand(10, 2, 3).astype("f"),
          y: np.random.rand(10, 2, 6).astype("f")
      }
      result = concated_grad.eval(feed_dict=params)

      self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testShapeError(self):
    # Rank doesn't match.
    with self.assertRaises(ValueError):
      array_ops.concat(
          [constant_op.constant(10.0, shape=[4, 4, 4, 4]),
           constant_op.constant(20.0, shape=[4, 4, 4])
          ], 1)

    # Dimensions don't match in a non-concat dim.
    with self.assertRaises(ValueError):
      array_ops.concat(
          [constant_op.constant(10.0, shape=[1, 2, 1]),
           constant_op.constant(20.0, shape=[3, 2, 1])
          ], 1)

    # concat_dim out of range.
    with self.assertRaises(ValueError):
      array_ops.concat(
          [constant_op.constant(10.0, shape=[4, 4, 4]),
           constant_op.constant(20.0, shape=[4, 4, 4])
          ], 3)

    # concat_dim out of range
    with self.assertRaises(ValueError):
      array_ops.concat(
          [constant_op.constant(10.0, shape=[4, 4, 4]),
           constant_op.constant(20.0, shape=[4, 4, 4])
          ], -4)

  @test_util.run_deprecated_v1
  def testShapeWithUnknownConcatDim(self):
    p1 = array_ops.placeholder(dtypes.float32)
    c1 = constant_op.constant(10.0, shape=[4, 4, 4, 4])
    p2 = array_ops.placeholder(dtypes.float32)
    c2 = constant_op.constant(20.0, shape=[4, 4, 4, 4])
    dim = array_ops.placeholder(dtypes.int32)
    concat = array_ops.concat([p1, c1, p2, c2], dim)
    self.assertEqual(4, concat.get_shape().ndims)

    # All dimensions unknown.
    concat2 = array_ops.concat([p1, p2], dim)
    self.assertEqual(None, concat2.get_shape())

    # Rank doesn't match.
    c3 = constant_op.constant(30.0, shape=[4, 4, 4])
    with self.assertRaises(ValueError):
      array_ops.concat([p1, c1, p2, c3], dim)

  @test_util.run_deprecated_v1
  def testZeroSize(self):
    # Verify that concat doesn't crash and burn for zero size inputs
    np.random.seed(7)
    with test_util.use_gpu():
      for shape0 in (), (2,):
        axis = len(shape0)
        for shape1 in (), (3,):
          for n0 in 0, 1, 2:
            for n1 in 0, 1, 2:
              x0 = np.random.randn(*(shape0 + (n0,) + shape1))
              x1 = np.random.randn(*(shape0 + (n1,) + shape1))
              correct = np.concatenate([x0, x1], axis=axis)
              # TODO(irving): Make tf.concat handle map, then drop list().
              xs = list(map(constant_op.constant, [x0, x1]))
              c = array_ops.concat(xs, axis)
              self.assertAllEqual(self.evaluate(c), correct)
              # Check gradients
              dc = np.random.randn(*c.get_shape().as_list())
              dxs = self.evaluate(gradients_impl.gradients(c, xs, dc))
              self.assertAllEqual(dc, np.concatenate(dxs, axis=axis))

  @test_util.run_deprecated_v1
  def testTensorConcatDim0Grad(self):
    x_shapes = [[20, 7, 3], [10, 7, 3], [14, 7, 3]]
    output_shape = [44, 7, 3]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      output = array_ops.concat(xs, 0)
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testTensorConcatDim1Grad(self):
    x_shapes = [[20, 7, 3], [20, 3, 3], [20, 1, 3]]
    output_shape = [20, 11, 3]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      output = array_ops.concat(xs, 1)
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testIndexedSlicesConcatDim0Grad(self):
    x_shapes = [[20, 7, 3], [10, 7, 3], [14, 7, 3]]
    output_shape = [4, 7, 3]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      x_concat = array_ops.concat(xs, 0)
      output = array_ops.gather(x_concat, [1, 2, 0, 5])
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testIndexedSlicesConcatDim1Grad(self):
    x_shapes = [[20, 7, 3], [20, 3, 3], [20, 1, 3]]
    output_shape = [4, 11, 3]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      x_concat = array_ops.concat(xs, 1)
      output = array_ops.gather(x_concat, [1, 2, 0, 5])
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testIndexedSlicesConcatDim2Grad(self):
    x_shapes = [[20, 7, 3], [20, 7, 1], [20, 7, 2]]
    output_shape = [4, 7, 6]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      x_concat = array_ops.concat(xs, 2)
      output = array_ops.gather(x_concat, [1, 2, 0, 5])
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testIndexedSlicesConcatDim1Grad_UnknownInputDim(self):
    x_shapes = [[20, 7, 3], [20, 3, 3], [20, 1, 3]]
    output_shape = [4, 11, 3]
    with self.cached_session():
      x_1 = array_ops.placeholder(dtypes.float64)
      x_2 = array_ops.placeholder(dtypes.float64)
      x_3 = array_ops.placeholder(dtypes.float64)
      xs = [x_1, x_2, x_3]

      x_concat = array_ops.concat(xs, 1)
      output = array_ops.gather(x_concat, [1, 2, 0, 5])
      params = {
          x_1: np.random.random_sample(x_shapes[0]).astype(np.float64),
          x_2: np.random.random_sample(x_shapes[1]).astype(np.float64),
          x_3: np.random.random_sample(x_shapes[2]).astype(np.float64)
      }
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape,
                                                    extra_feed_dict=params)
    self.assertLess(err, 1e-11)

  def testConcatTuple(self):
    c1 = np.random.rand(4, 4)
    c2 = np.random.rand(4, 4)
    concat_list_t = array_ops.concat([c1, c2], 0)
    concat_tuple_t = array_ops.concat((c1, c2), 0)
    self.assertAllEqual(
        self.evaluate(concat_list_t), self.evaluate(concat_tuple_t))

  @test_util.run_deprecated_v1
  def testConcatNoScalars(self):
    scalar = constant_op.constant(7)
    dim = array_ops.placeholder(dtypes.int32)
    with self.assertRaisesRegexp(
        ValueError, r"Can't concatenate scalars \(use tf\.stack instead\)"):
      array_ops.concat([scalar, scalar, scalar], dim)

  # important as gpu implementation could fail if
  # shared memory is not large for all the inputs
  @test_util.run_deprecated_v1
  def testConcatLargeNumberOfTensors(self):
    with self.session(use_gpu=True):
      for concat_dim in range(2):
        params = {}
        p = []
        shape = np.array([7, 13])
        if test.is_gpu_available():
          num_tensors = 5000
        else:
          num_tensors = 500
        for i in np.arange(num_tensors):
          input_shape = shape
          placeholder = array_ops.placeholder(dtypes.float32, shape=input_shape)
          p.append(placeholder)

          params[placeholder] = np.random.rand(*input_shape).astype(np.float32)

        concat_inputs = p
        c = array_ops.concat(concat_inputs, concat_dim)
        result = c.eval(feed_dict=params)

        self.assertEqual(result.shape, c.get_shape())
        cur_offset = 0

        for i in np.arange(num_tensors):
          # The index into the result is the ':' along all dimensions
          # except the concat_dim. slice(0, size) is used for ':', and
          # a list of slices is used to index into result.
          index = [slice(0, params[p[i]].shape[j]) for j in np.arange(2)]
          index[concat_dim] = slice(cur_offset,
                                    cur_offset + params[p[i]].shape[concat_dim])
          cur_offset += params[p[i]].shape[concat_dim]
          self.assertAllEqual(result[index], params[p[i]])

  def testConcatEmpty(self):
    with test_util.use_gpu():
      t1 = []
      t2 = []
      output = gen_array_ops.concat_v2([t1, t2], 0)
      self.assertFalse(self.evaluate(output))  # Checks that output is empty

  @test_util.run_deprecated_v1
  def testConcatInvalidAxis(self):
    with self.assertRaises(ValueError):
      with test_util.use_gpu():
        t1 = [1]
        t2 = [2]
        gen_array_ops.concat_v2([t1, t2], 1).eval()

  def testConcatNegativeAxis(self):
    with test_util.use_gpu():
      t1 = [[1, 2, 3], [4, 5, 6]]
      t2 = [[7, 8, 9], [10, 11, 12]]

      c = gen_array_ops.concat_v2([t1, t2], -2)
      self.assertEqual([4, 3], c.get_shape().as_list())
      output = self.evaluate(c)
      self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                          output)

      c = gen_array_ops.concat_v2([t1, t2], -1)
      self.assertEqual([2, 6], c.get_shape().as_list())
      output = self.evaluate(c)
      self.assertAllEqual([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], output)

  def _testGradientsForAxis(
      self, inp_tensors, axis, output_shape, feed_dict=None):
    with self.cached_session():
      c = array_ops.concat(inp_tensors, axis)
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(
          grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.concat(grad, axis)
      result = concated_grad.eval(feed_dict=feed_dict)
      self.assertAllEqual(result, grad_inp)

  def _testIndexedSlicesGradientsForAxis(
      self, inp_tensors, axis, output_shape, gather_indexes, feed_dict=None):
    with self.cached_session():
      c = array_ops.gather(
          array_ops.concat(inp_tensors, axis), gather_indexes)
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(
          grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.gather(
          array_ops.concat(grad, axis), gather_indexes)
      result = concated_grad.eval(feed_dict=feed_dict)
      self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsNegativeAxis(self):
    x1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x2 = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    inp_tensors = [constant_op.constant(x1, shape=(2, 3), dtype=dtypes.float32),
                   constant_op.constant(x2, shape=(2, 3), dtype=dtypes.float32)]

    # Test concat gradient with axis == -2
    self._testGradientsForAxis(inp_tensors, -2, output_shape=[4, 3])

    # Test concat gradient with unknown-shape tensors.
    x1_placeholder = array_ops.placeholder(dtypes.float32)
    x2_placeholder = array_ops.placeholder(dtypes.float32)
    inp_tensors_placeholders = [x1_placeholder, x2_placeholder]
    feed_dict = {x1_placeholder: x1, x2_placeholder: x2}
    self._testGradientsForAxis(
        inp_tensors_placeholders, -1, output_shape=[2, 6], feed_dict=feed_dict)

    # Test IndexedSlices concat gradient.
    self._testIndexedSlicesGradientsForAxis(
        inp_tensors, -2, output_shape=[2, 3], gather_indexes=[2, 0])

    # We don't support calculating IndexedSlices concat gradient for
    # negative indexes when rank is not known.
    with self.assertRaises(ValueError):
      self._testIndexedSlicesGradientsForAxis(
          inp_tensors_placeholders, -2, output_shape=[2, 3],
          gather_indexes=[2, 0], feed_dict=feed_dict)

  def testConcatAxisType(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with test_util.use_gpu():
        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]

        c = gen_array_ops.concat_v2([t1, t2],
                                    constant_op.constant(1, dtype=dtype))
        self.assertEqual([2, 6], c.get_shape().as_list())
        output = self.evaluate(c)
        self.assertAllEqual([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], output)


@test_util.disable_all_xla("This test never passed for XLA")
class ConcatOffsetTest(test.TestCase):

  def testBasic(self):
    with test_util.use_gpu():
      cdim = constant_op.constant(1, dtypes.int32)
      s0 = constant_op.constant([2, 3, 5], dtypes.int32)
      s1 = constant_op.constant([2, 7, 5], dtypes.int32)
      s2 = constant_op.constant([2, 20, 5], dtypes.int32)
      off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
      ans = self.evaluate(off)
      self.assertAllEqual(ans, [[0, 0, 0], [0, 3, 0], [0, 10, 0]])

  @test_util.run_deprecated_v1
  def testNotVector(self):
    cdim = constant_op.constant(1, dtypes.int32)
    s0 = constant_op.constant([[2, 3, 5]], dtypes.int32)
    s1 = constant_op.constant([[2, 7, 5]], dtypes.int32)
    off = gen_array_ops.concat_offset(cdim, [s0, s1])
    with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                 r"should be a vector"):
      self.evaluate(off)

  @test_util.run_deprecated_v1
  def testConcatDimOutOfRange(self):
    cdim = constant_op.constant(4, dtypes.int32)
    s0 = constant_op.constant([2, 3, 5], dtypes.int32)
    s1 = constant_op.constant([2, 7, 5], dtypes.int32)
    off = gen_array_ops.concat_offset(cdim, [s0, s1])
    with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                 r"Concat dim is out of range: 4 vs. 3"):
      self.evaluate(off)

  @test_util.run_deprecated_v1
  def testDimMismatch(self):
    cdim = constant_op.constant(1, dtypes.int32)
    s0 = constant_op.constant([2, 3, 5], dtypes.int32)
    s1 = constant_op.constant([2, 7, 5, 10], dtypes.int32)
    off = gen_array_ops.concat_offset(cdim, [s0, s1])
    with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                 r"should contain 3 elem"):
      self.evaluate(off)

  @test_util.run_deprecated_v1
  @test_util.disable_xla(
      "This test never passed for XLA")  # Different error message on XLA
  def testSizeMismatch(self):
    cdim = constant_op.constant(1, dtypes.int32)
    s0 = constant_op.constant([2, 3, 5], dtypes.int32)
    s1 = constant_op.constant([2, 7, 10], dtypes.int32)
    off = gen_array_ops.concat_offset(cdim, [s0, s1])
    with self.assertRaisesRegexp(
        errors_impl.InvalidArgumentError,
        r"All dimensions except 1 must match. Input 1 has shape \[2 7 10\] "
        r"and doesn't match input 0 with shape \[2 3 5\]."):
      self.evaluate(off)

  def testNegativeDim(self):
    with test_util.use_gpu():
      cdim = constant_op.constant(-2, dtypes.int32)
      s0 = constant_op.constant([2, 3, 5], dtypes.int32)
      s1 = constant_op.constant([2, 7, 5], dtypes.int32)
      s2 = constant_op.constant([2, 20, 5], dtypes.int32)
      off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
      ans = self.evaluate(off)
      self.assertAllEqual(ans, [[0, 0, 0], [0, 3, 0], [0, 10, 0]])

      cdim = constant_op.constant(-3, dtypes.int32)
      s0 = constant_op.constant([2, 3, 5], dtypes.int32)
      s1 = constant_op.constant([1, 3, 5], dtypes.int32)
      s2 = constant_op.constant([3, 3, 5], dtypes.int32)
      off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
      ans = self.evaluate(off)
      self.assertAllEqual(ans, [[0, 0, 0], [2, 0, 0], [3, 0, 0]])


if __name__ == "__main__":
  test.main()
