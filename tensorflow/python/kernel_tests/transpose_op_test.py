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
"""Functional tests for Transpose op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test


class TransposeTest(test.TestCase):

  def _np_transpose(self, x, perm):
    ret = np.copy(x)
    ret = ret.transpose(perm)
    return ret

  def _compareCpu(self, x, p, conjugate=False):
    np_ans = self._np_transpose(x, p)
    if conjugate:
      np_ans = np.conj(np_ans)
    with self.test_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      y = array_ops.transpose(inx, p, conjugate=conjugate)
      tf_ans = y.eval()
      self.assertShapeEqual(np_ans, y)
      self.assertAllEqual(np_ans, tf_ans)

      jacob_t = None
      # Gradient check on CPU.
      xs = list(np.shape(x))
      ys = list(np.shape(tf_ans))
      if x.dtype in [np.float32, np.complex64]:
        jacob_t, jacob_n = gradient_checker.compute_gradient(inx, xs, y, ys, x,
                                                             1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)
      elif x.dtype in [np.float64, np.complex128]:
        jacob_t, jacob_n = gradient_checker.compute_gradient(inx, xs, y, ys, x,
                                                             1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-6, 1e-6)

      return tf_ans, jacob_t

  def _compareGpu(self, x, p, conjugate=False):
    np_ans = self._np_transpose(x, p)
    if conjugate:
      np_ans = np.conj(np_ans)
    with self.test_session(use_gpu=True):
      inx = ops.convert_to_tensor(x)
      y = array_ops.transpose(inx, p, conjugate=conjugate)
      tf_ans = y.eval()

      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

      jacob_t = None
      # Gradient check on GPU.
      xs = list(np.shape(x))
      ys = list(np.shape(tf_ans))
      if x.dtype == np.float32:
        jacob_t, jacob_n = gradient_checker.compute_gradient(inx, xs, y, ys, x,
                                                             1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        jacob_t, jacob_n = gradient_checker.compute_gradient(inx, xs, y, ys, x,
                                                             1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-6, 1e-6)

      return tf_ans, jacob_t

  def _compare(self, x, use_gpu=False):
    n = np.ndim(x)
    # generate all permutations of [0, 1, ... n-1] in random order.
    all_perm = np.random.permutation(
        [p for p in itertools.permutations(range(n))]).astype(np.int32)
    cs = [False, True] if x.dtype in [np.complex64, np.complex128] else [False]
    for c in cs:
      for p in all_perm[:2]:
        self._compareCpu(x, p, conjugate=c)
        if use_gpu:
          self._compareGpu(x, p, conjugate=c)

  def _compare_cpu_gpu(self, x):
    n = np.ndim(x)
    # generate all permutation of [0, 1, ... n-1] in random order,
    # choose the first two.
    perms = itertools.permutations(range(n))
    for _ in range(2):
      p = np.random.permutation(next(perms)).astype(np.int32)
      tf_a_cpu, tf_g_cpu = self._compareCpu(x, p)
      tf_a_gpu, tf_g_gpu = self._compareGpu(x, p)
      assert tf_g_cpu is not None
      assert tf_g_gpu is not None
      if x.dtype == np.float32:
        self.assertAllClose(tf_a_cpu, tf_a_gpu, 1e-3, 1e-3)
        self.assertAllClose(tf_g_cpu, tf_g_gpu, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        self.assertAllClose(tf_a_cpu, tf_a_gpu, 1e-6, 1e-6)
        self.assertAllClose(tf_g_cpu, tf_g_gpu, 1e-6, 1e-6)

  def _testBoth(self, x):
    self._compare(x, use_gpu=False)
    self._compare(x, use_gpu=True)

  def testRank1(self):
    self._compareCpu(np.arange(0., 2), [0])

  def test1D(self):
    vector = np.arange(0, 2).reshape((1, 1, 1, 2, 1))
    self._compare(vector, use_gpu=False)
    self._compare(vector, use_gpu=True)

  def test5DGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return
    large_shapes = [[4, 10, 10, 10, 3], [4, 10, 10, 10, 8], [4, 10, 10, 10, 13],
                    [4, 3, 10, 10, 10], [4, 8, 10, 10, 10], [4, 13, 10, 10,
                                                             10]] * 3
    perms = [[0, 4, 1, 2, 3]] * 3 + [[0, 2, 3, 4, 1]] * 3 + [[
        4, 1, 2, 3, 0
    ]] * 6 + [[1, 2, 3, 4, 0]] * 6

    datatypes = [np.int8, np.float16, np.float32, np.float64, np.complex128]
    for datatype in datatypes:
      for input_shape, perm in zip(large_shapes, perms):
        total_size = np.prod(input_shape)
        inp = np.arange(1, total_size + 1, dtype=datatype).reshape(input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.test_session(use_gpu=True):
          inx = ops.convert_to_tensor(inp)
          y = array_ops.transpose(inx, perm)
          tf_ans = y.eval()
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)

  def test4DGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return
    large_shapes = [[4, 10, 10, 3], [4, 10, 10, 8], [4, 10, 10, 13],
                    [4, 3, 10, 10], [4, 8, 10, 10], [4, 13, 10, 10]] * 3
    perms = [[0, 3, 1, 2]] * 3 + [[0, 2, 3, 1]] * 3 + [[3, 1, 2, 0]] * 6 + [[
        1, 2, 3, 0
    ]] * 3 + [[2, 3, 0, 1]] * 3

    for input_shape, perm in zip(large_shapes, perms):
      total_size = np.prod(input_shape)
      inp = np.arange(1, total_size + 1, dtype=np.float32).reshape(input_shape)
      np_ans = self._np_transpose(inp, perm)
      with self.test_session(use_gpu=True):
        inx = ops.convert_to_tensor(inp)
        y = array_ops.transpose(inx, perm)
        tf_ans = y.eval()
      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

    # shapes related to Inception (taken from conv_ops_test.py)
    inception_shapes = [[4, 5, 5, 124], [4, 8, 8, 38], [4, 8, 8, 38], [
        4, 8, 8, 204
    ], [4, 8, 8, 44], [4, 8, 8, 204], [4, 8, 8, 204], [4, 8, 8, 204], [
        4, 8, 8, 176
    ], [4, 8, 8, 176], [4, 8, 8, 176], [4, 8, 8, 176], [4, 17, 17, 19], [
        4, 17, 17, 19
    ], [4, 17, 17, 124], [4, 17, 17, 12], [4, 17, 17, 124], [4, 17, 17, 22], [
        4, 17, 17, 19
    ], [4, 17, 17, 19], [4, 17, 17, 121], [4, 17, 17, 121], [4, 17, 17, 22], [
        4, 17, 17, 19
    ], [4, 17, 17, 19], [4, 17, 17, 115], [4, 17, 17, 115], [4, 17, 17, 19], [
        4, 17, 17, 16
    ], [4, 17, 17, 115], [4, 17, 17, 102], [4, 17, 17, 12], [4, 17, 17, 102], [
        4, 17, 17, 12
    ], [4, 17, 17, 102], [4, 17, 17, 12], [4, 17, 17, 76], [4, 17, 17, 12], [
        4, 17, 17, 12
    ], [4, 17, 17, 76], [4, 17, 17, 76], [4, 35, 35, 9], [4, 35, 35, 28], [
        4, 35, 35, 6
    ], [4, 35, 35, 28], [4, 35, 35, 25], [4, 35, 35, 4], [4, 35, 35, 25],
                        [4, 35, 35, 9], [4, 35, 35, 19], [4, 35, 35, 19],
                        [4, 35, 35, 19], [4, 73, 73, 6], [4, 73, 73,
                                                          6], [4, 147, 147, 2]]
    for input_shape in inception_shapes:
      perm = [0, 3, 1, 2]
      total_size = np.prod(input_shape)
      inp = np.arange(1, total_size + 1, dtype=np.float32).reshape(input_shape)
      np_ans = self._np_transpose(inp, perm)
      with self.test_session(use_gpu=True):
        inx = ops.convert_to_tensor(inp)
        y = array_ops.transpose(inx, perm)
        tf_ans = y.eval()
      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

  def test3DGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    datatypes = [np.int8, np.float16, np.float32, np.float64, np.complex128]
    large_shapes = [[4, 1000, 3], [4, 1000, 8], [4, 1000, 13], [4, 3, 1000],
                    [4, 8, 1000], [4, 13, 1000]] * 3
    perms = [[0, 2, 1]] * 6 + [[2, 1, 0]] * 6 + [[1, 2, 0]] * 3 + [[2, 0, 1]
                                                                  ] * 3
    for datatype in datatypes:
      for input_shape, perm in zip(large_shapes, perms):
        total_size = np.prod(input_shape)
        inp = np.arange(1, total_size + 1, dtype=datatype).reshape(input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.test_session(use_gpu=True):
          inx = ops.convert_to_tensor(inp)
          y = array_ops.transpose(inx, perm)
          tf_ans = y.eval()
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)

  def testLargeSizeGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    large_shapes = [[1000000, 31, 3], [3, 1000000, 31], [3, 31, 1000000],
                    [10000, 310, 3], [3, 10000, 310], [3, 310, 10000],
                    [2, 1000, 1000], [1000, 2, 1000], [1000, 1000, 2]]
    perms = [[0, 2, 1]] * 9

    for input_shape, perm in zip(large_shapes, perms):
      total_size = np.prod(input_shape)
      inp = np.arange(1, total_size + 1, dtype=np.float32).reshape(input_shape)
      np_ans = self._np_transpose(inp, perm)
      with self.test_session(use_gpu=True):
        inx = ops.convert_to_tensor(inp)
        y = array_ops.transpose(inx, perm)
        tf_ans = y.eval()
      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

  def testRandomizedSmallDimLargeSizeGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    # Draw 10 random shapes with large dimension sizes.
    # 40% prob to generate dim[0] size within [1, 2047]
    # 40% prob to generate dim[0] size within [2048, 4095]
    # 20% prob to generate dim[0] size within [4096, 100000]
    # 50% prob to use dim[1] as the small dim (<16)
    num_samples = 10
    total_size = 500000
    small_size_limit = 2048
    large_size_limit = 95905
    small_size_percentage = 0.4
    medium_size_percentage = 0.4
    large_size_percentage = 0.2
    perms = [[0, 2, 1]] * num_samples
    dim_zero_sizes = []
    dim_zero_sizes += list(
        np.random.randint(
            small_size_limit, size=int(small_size_percentage * num_samples)) +
        1)
    dim_zero_sizes += list(
        np.random.randint(
            small_size_limit, size=int(medium_size_percentage * num_samples)) +
        small_size_limit)
    dim_zero_sizes += list(
        np.random.randint(
            large_size_limit, size=int(large_size_percentage * num_samples)) +
        small_size_limit * 2)
    input_shapes = []
    small_dim_limit = 16
    for dim_zero_size in dim_zero_sizes:
      small_dim_size = np.random.randint(small_dim_limit - 1) + 1
      large_dim_size = int(
          total_size / dim_zero_size / small_dim_size) + small_dim_limit
      input_shapes += ([[dim_zero_size, small_dim_size, large_dim_size]]
                       if np.random.randint(2) else
                       [[dim_zero_size, large_dim_size, small_dim_size]])

    for input_shape, perm in zip(input_shapes, perms):
      # generate input data with random ints from 0 to 9.
      inp = np.random.randint(10, size=input_shape)
      np_ans = self._np_transpose(inp, perm)
      with self.test_session(use_gpu=True):
        inx = ops.convert_to_tensor(inp)
        y = array_ops.transpose(inx, perm)
        tf_ans = y.eval()
      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)
      self._ClearCachedSession()

  def testNop(self):
    self._compareCpu(np.arange(0, 6).reshape([3, 2]).astype(np.float32), [0, 1])

  def testSimple(self):
    self._compareCpu(
        np.arange(0, 8).reshape([2, 4]).astype(np.float32),
        np.array([1, 0]).astype(np.int32))

  def testPermType(self):
    for perm_dtype in [np.int64, np.int32]:
      x = np.arange(0, 8).reshape([2, 4]).astype(np.float32)
      p = np.array([1, 0]).astype(perm_dtype)
      np_ans = np.copy(x).transpose(p)
      with self.test_session(use_gpu=True):
        inx = ops.convert_to_tensor(x)
        inp = constant_op.constant(p)
        y = array_ops.transpose(inx, inp)
        tf_ans = y.eval()
        self.assertShapeEqual(np_ans, y)
        self.assertAllEqual(np_ans, tf_ans)

  def testHalf(self):
    self._compare(np.arange(0, 21).reshape([3, 7]).astype(np.float16))
    self._compare(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float16))
    self._compare(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float16))

  def testFloat(self):
    self._compare_cpu_gpu(np.arange(0, 21).reshape([3, 7]).astype(np.float32))
    self._compare_cpu_gpu(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float32))
    self._compare_cpu_gpu(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float32))

  def testDouble(self):
    self._compare_cpu_gpu(np.arange(0, 21).reshape([3, 7]).astype(np.float64))
    self._compare_cpu_gpu(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float64))
    self._compare_cpu_gpu(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float64))

  def testComplex64(self):
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 21).reshape([3, 7]).astype(np.complex64))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.complex64))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.complex64))

  def testComplex128(self):
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 21).reshape([3, 7]).astype(np.complex128))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.complex128))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.complex128))

  def testInt8(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int8))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int8))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int8))

  def testInt16(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int16))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int16))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int16))

  def testInt32(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int32))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int32))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int32))

  def testInt64(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int64))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int64))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int64))

  def testTranspose2DAuto(self):
    x_np = [[1, 2, 3], [4, 5, 6]]
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = array_ops.transpose(x_np).eval()
        self.assertAllEqual(x_tf, [[1, 4], [2, 5], [3, 6]])

  def testSingletonDims(self):
    # A singleton dimension is a dimension i with shape[i] == 1. Such dimensions
    # can be collapsed and expanded using reshape without changing the
    # underlying data storage. If all non-singleton dimensions remain in
    # ascending order, the shuffled singletons will be transposed by a reshape,
    # saving a memory allocation & copy. Since this gets a special code-path in
    # transpose_op.cc, we test that the codepath is exercised and the results
    # are as expected; we do not test that we save the memory allocation and
    # copy here.
    for shape in [[2, 1, 2], [2, 1, 2, 1, 1, 2], [1, 2, 2, 1, 1, 1],
                  [1, 1, 1, 2, 2, 2], [2, 2, 1, 1, 1]]:
      self._compare_cpu_gpu(
          np.arange(np.prod(shape)).reshape(shape).astype(np.float32))

  def testTransposeShapes(self):
    self.assertEqual(
        [],
        array_ops.transpose(array_ops.placeholder(
            dtypes.int32, shape=[])).get_shape().dims)
    self.assertEqual(
        [100],
        array_ops.transpose(array_ops.placeholder(
            dtypes.int32, shape=[100])).get_shape().dims)
    self.assertEqual(
        [37, 100],
        array_ops.transpose(
            array_ops.placeholder(
                dtypes.int32, shape=[100, 37])).get_shape().dims)
    self.assertEqual(
        [100, 37],
        array_ops.transpose(
            array_ops.placeholder(
                dtypes.int32, shape=[100, 37]), [0, 1]).get_shape().dims)
    self.assertEqual(
        [15, 37, 100],
        array_ops.transpose(
            array_ops.placeholder(
                dtypes.int32, shape=[100, 37, 15])).get_shape().dims)
    self.assertEqual(
        [15, 100, 37],
        array_ops.transpose(
            array_ops.placeholder(
                dtypes.int32, shape=[100, 37, 15]), [2, 0, 1]).get_shape().dims)
    self.assertEqual(
        tensor_shape.TensorShape(None),
        array_ops.transpose(array_ops.placeholder(dtypes.int32)).get_shape())

  def testNullTensor(self):
    with self.cached_session():
      x = constant_op.constant([], dtype=dtypes.float32, shape=[1, 4, 0])
      xt = array_ops.transpose(x, [0, 2, 1]).eval()
      self.assertAllEqual(xt.shape, (1, 0, 4))

  def _testError(self, x, p, err):
    with self.cached_session():
      with self.assertRaisesOpError(err):
        array_ops.transpose(x, p).eval()

  def testError(self):
    with self.assertRaises(ValueError):
      array_ops.transpose(
          np.arange(0., 30).reshape([2, 3, 5]), [[0, 1], [2, 3]])
    with self.assertRaises(ValueError):
      array_ops.transpose(np.arange(0., 30).reshape([2, 3, 5]), [0, 1, 3])
    self._testError(
        np.arange(0., 30).reshape([2, 3, 5]), [0, 1, 1], "2 is missing")


if __name__ == "__main__":
  test.main()
