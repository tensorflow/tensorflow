# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def _AddTest(test_class, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)


class SvdOpTest(test.TestCase):

  @test_util.run_v1_only("b/120545219")
  def testWrongDimensions(self):
    # The input to svd should be a tensor of at least rank 2.
    scalar = constant_op.constant(1.)
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be at least rank 2 but is rank 0"):
      linalg_ops.svd(scalar)
    vector = constant_op.constant([1., 2.])
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be at least rank 2 but is rank 1"):
      linalg_ops.svd(vector)

  @test_util.run_v1_only("b/120545219")
  def testConcurrentExecutesWithoutError(self):
    with self.session(use_gpu=True) as sess:
      all_ops = []
      for compute_uv_ in True, False:
        for full_matrices_ in True, False:
          matrix1 = random_ops.random_normal([5, 5], seed=42)
          matrix2 = random_ops.random_normal([5, 5], seed=42)
          if compute_uv_:
            s1, u1, v1 = linalg_ops.svd(
                matrix1, compute_uv=compute_uv_, full_matrices=full_matrices_)
            s2, u2, v2 = linalg_ops.svd(
                matrix2, compute_uv=compute_uv_, full_matrices=full_matrices_)
            all_ops += [s1, u1, v1, s2, u2, v2]
          else:
            s1 = linalg_ops.svd(
                matrix1, compute_uv=compute_uv_, full_matrices=full_matrices_)
            s2 = linalg_ops.svd(
                matrix2, compute_uv=compute_uv_, full_matrices=full_matrices_)
            all_ops += [s1, s2]
      val = self.evaluate(all_ops)
      for i in range(2):
        s = 6 * i
        self.assertAllEqual(val[s], val[s + 3])  # s1 == s2
        self.assertAllEqual(val[s + 1], val[s + 4])  # u1 == u2
        self.assertAllEqual(val[s + 2], val[s + 5])  # v1 == v2
      for i in range(2):
        s = 12 + 2 * i
        self.assertAllEqual(val[s], val[s + 1])  # s1 == s2


def _GetSvdOpTest(dtype_, shape_, use_static_shape_, compute_uv_,
                  full_matrices_):

  def CompareSingularValues(self, x, y, tol):
    self.assertAllClose(x, y, atol=(x[0] + y[0]) * tol)

  def CompareSingularVectors(self, x, y, rank, tol):
    # We only compare the first 'rank' singular vectors since the
    # remainder form an arbitrary orthonormal basis for the
    # (row- or column-) null space, whose exact value depends on
    # implementation details. Notice that since we check that the
    # matrices of singular vectors are unitary elsewhere, we do
    # implicitly test that the trailing vectors of x and y span the
    # same space.
    x = x[..., 0:rank]
    y = y[..., 0:rank]
    # Singular vectors are only unique up to sign (complex phase factor for
    # complex matrices), so we normalize the sign first.
    sum_of_ratios = np.sum(np.divide(y, x), -2, keepdims=True)
    phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
    x *= phases
    self.assertAllClose(x, y, atol=2 * tol)

  def CheckApproximation(self, a, u, s, v, full_matrices_, tol):
    # Tests that a ~= u*diag(s)*transpose(v).
    batch_shape = a.shape[:-2]
    m = a.shape[-2]
    n = a.shape[-1]
    diag_s = math_ops.cast(array_ops.matrix_diag(s), dtype=dtype_)
    if full_matrices_:
      if m > n:
        zeros = array_ops.zeros(batch_shape + (m - n, n), dtype=dtype_)
        diag_s = array_ops.concat([diag_s, zeros], a.ndim - 2)
      elif n > m:
        zeros = array_ops.zeros(batch_shape + (m, n - m), dtype=dtype_)
        diag_s = array_ops.concat([diag_s, zeros], a.ndim - 1)
    a_recon = math_ops.matmul(u, diag_s)
    a_recon = math_ops.matmul(a_recon, v, adjoint_b=True)
    self.assertAllClose(a_recon, a, rtol=tol, atol=tol)

  def CheckUnitary(self, x, tol):
    # Tests that x[...,:,:]^H * x[...,:,:] is close to the identity.
    xx = math_ops.matmul(x, x, adjoint_a=True)
    identity = array_ops.matrix_band_part(array_ops.ones_like(xx), 0, 0)
    self.assertAllClose(identity, xx, atol=tol)

  @test_util.run_v1_only("b/120545219")
  def Test(self):
    is_complex = dtype_ in (np.complex64, np.complex128)
    is_single = dtype_ in (np.float32, np.complex64)
    tol = 3e-4 if is_single else 1e-12
    if test.is_gpu_available():
      # The gpu version returns results that are much less accurate.
      tol *= 100
    np.random.seed(42)
    x_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    if is_complex:
      x_np += 1j * np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)

    with self.session(use_gpu=True) as sess:
      if use_static_shape_:
        x_tf = constant_op.constant(x_np)
      else:
        x_tf = array_ops.placeholder(dtype_)

      if compute_uv_:
        s_tf, u_tf, v_tf = linalg_ops.svd(
            x_tf, compute_uv=compute_uv_, full_matrices=full_matrices_)
        if use_static_shape_:
          s_tf_val, u_tf_val, v_tf_val = self.evaluate([s_tf, u_tf, v_tf])
        else:
          s_tf_val, u_tf_val, v_tf_val = sess.run(
              [s_tf, u_tf, v_tf], feed_dict={x_tf: x_np})
      else:
        s_tf = linalg_ops.svd(
            x_tf, compute_uv=compute_uv_, full_matrices=full_matrices_)
        if use_static_shape_:
          s_tf_val = self.evaluate(s_tf)
        else:
          s_tf_val = sess.run(s_tf, feed_dict={x_tf: x_np})

      if compute_uv_:
        u_np, s_np, v_np = np.linalg.svd(
            x_np, compute_uv=compute_uv_, full_matrices=full_matrices_)
      else:
        s_np = np.linalg.svd(
            x_np, compute_uv=compute_uv_, full_matrices=full_matrices_)
      # We explicitly avoid the situation where numpy eliminates a first
      # dimension that is equal to one.
      s_np = np.reshape(s_np, s_tf_val.shape)

      CompareSingularValues(self, s_np, s_tf_val, tol)
      if compute_uv_:
        CompareSingularVectors(self, u_np, u_tf_val, min(shape_[-2:]), tol)
        CompareSingularVectors(self,
                               np.conj(np.swapaxes(v_np, -2, -1)), v_tf_val,
                               min(shape_[-2:]), tol)
        CheckApproximation(self, x_np, u_tf_val, s_tf_val, v_tf_val,
                           full_matrices_, tol)
        CheckUnitary(self, u_tf_val, tol)
        CheckUnitary(self, v_tf_val, tol)

  return Test


class SvdGradOpTest(test.TestCase):
  pass  # Filled in below


def _NormalizingSvd(tf_a, full_matrices_):
  tf_s, tf_u, tf_v = linalg_ops.svd(
      tf_a, compute_uv=True, full_matrices=full_matrices_)
  # Singular vectors are only unique up to an arbitrary phase. We normalize
  # the vectors such that the first component of u (if m >=n) or v (if n > m)
  # have phase 0.
  m = tf_a.shape[-2]
  n = tf_a.shape[-1]
  if m >= n:
    top_rows = tf_u[..., 0:1, :]
  else:
    top_rows = tf_v[..., 0:1, :]
  if tf_u.dtype.is_complex:
    angle = -math_ops.angle(top_rows)
    phase = math_ops.complex(math_ops.cos(angle), math_ops.sin(angle))
  else:
    phase = math_ops.sign(top_rows)
  tf_u *= phase[..., :m]
  tf_v *= phase[..., :n]
  return tf_s, tf_u, tf_v


def _GetSvdGradOpTest(dtype_, shape_, compute_uv_, full_matrices_):

  @test_util.run_v1_only("b/120545219")
  def Test(self):
    np.random.seed(42)
    a = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
    if dtype_ in [np.complex64, np.complex128]:
      a += 1j * np.random.uniform(
          low=-1.0, high=1.0, size=shape_).astype(dtype_)
    # Optimal stepsize for central difference is O(epsilon^{1/3}).
    # See Equation (21) in:
    # http://www.karenkopecky.net/Teaching/eco613614/Notes_NumericalDifferentiation.pdf
    # TODO(rmlarsen): Move step size control to gradient checker.
    epsilon = np.finfo(dtype_).eps
    delta = 0.1 * epsilon**(1.0 / 3.0)
    if dtype_ in [np.float32, np.complex64]:
      tol = 3e-2
    else:
      tol = 1e-6
    with self.session(use_gpu=True):
      tf_a = constant_op.constant(a)
      if compute_uv_:
        tf_s, tf_u, tf_v = _NormalizingSvd(tf_a, full_matrices_)
        outputs = [tf_s, tf_u, tf_v]
      else:
        tf_s = linalg_ops.svd(tf_a, compute_uv=False)
        outputs = [tf_s]
      for b in outputs:
        x_init = np.random.uniform(
            low=-1.0, high=1.0, size=shape_).astype(dtype_)
        if dtype_ in [np.complex64, np.complex128]:
          x_init += 1j * np.random.uniform(
              low=-1.0, high=1.0, size=shape_).astype(dtype_)
        theoretical, numerical = gradient_checker.compute_gradient(
            tf_a,
            tf_a.get_shape().as_list(),
            b,
            b.get_shape().as_list(),
            x_init_value=x_init,
            delta=delta)
        self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)
  return Test


class SvdGradGradOpTest(test.TestCase):
  pass  # Filled in below


def _GetSvdGradGradOpTest(dtype_, shape_, compute_uv_, full_matrices_):

  @test_util.run_v1_only("b/120545219")
  def Test(self):
    np.random.seed(42)
    a = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
    if dtype_ in [np.complex64, np.complex128]:
      a += 1j * np.random.uniform(
          low=-1.0, high=1.0, size=shape_).astype(dtype_)
    # Optimal stepsize for central difference is O(epsilon^{1/3}).
    # See Equation (21) in:
    # http://www.karenkopecky.net/Teaching/eco613614/Notes_NumericalDifferentiation.pdf
    # TODO(rmlarsen): Move step size control to gradient checker.
    epsilon = np.finfo(dtype_).eps
    delta = 0.1 * epsilon**(1.0 / 3.0)
    tol = 1e-5
    with self.session(use_gpu=True):
      tf_a = constant_op.constant(a)
      if compute_uv_:
        tf_s, tf_u, tf_v = _NormalizingSvd(tf_a, full_matrices_)
        outputs = [tf_s, tf_u, tf_v]
      else:
        tf_s = linalg_ops.svd(tf_a, compute_uv=False)
        outputs = [tf_s]
      outputs_sums = [math_ops.reduce_sum(o) for o in outputs]
      tf_func_outputs = math_ops.add_n(outputs_sums)
      grad = gradients_impl.gradients(tf_func_outputs, tf_a)[0]
      x_init = np.random.uniform(
          low=-1.0, high=1.0, size=shape_).astype(dtype_)
      if dtype_ in [np.complex64, np.complex128]:
        x_init += 1j * np.random.uniform(
            low=-1.0, high=1.0, size=shape_).astype(dtype_)
      theoretical, numerical = gradient_checker.compute_gradient(
          tf_a,
          tf_a.get_shape().as_list(),
          grad,
          grad.get_shape().as_list(),
          x_init_value=x_init,
          delta=delta)
      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)
  return Test


class SVDBenchmark(test.Benchmark):

  shapes = [
      (4, 4),
      (8, 8),
      (16, 16),
      (101, 101),
      (256, 256),
      (1024, 1024),
      (2048, 2048),
      (1, 8, 8),
      (10, 8, 8),
      (100, 8, 8),
      (1000, 8, 8),
      (1, 32, 32),
      (10, 32, 32),
      (100, 32, 32),
      (1000, 32, 32),
      (1, 256, 256),
      (10, 256, 256),
      (100, 256, 256),
  ]

  def benchmarkSVDOp(self):
    for shape_ in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix_value = np.random.uniform(
            low=-1.0, high=1.0, size=shape_).astype(np.float32)
        matrix = variables.Variable(matrix_value)
        u, s, v = linalg_ops.svd(matrix)
        variables.global_variables_initializer().run()
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(u, s, v),
            min_iters=25,
            name="SVD_cpu_{shape}".format(shape=shape_))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/device:GPU:0"):
          matrix_value = np.random.uniform(
              low=-1.0, high=1.0, size=shape_).astype(np.float32)
          matrix = variables.Variable(matrix_value)
          u, s, v = linalg_ops.svd(matrix)
          variables.global_variables_initializer().run()
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(u, s, v),
              min_iters=25,
              name="SVD_gpu_{shape}".format(shape=shape_))


if __name__ == "__main__":
  dtypes_to_test = [np.float32, np.float64]
  if not test.is_built_with_rocm():
    # ROCm does not support BLAS operations for complex types
    dtypes_to_test += [np.complex64, np.complex128]
  for compute_uv in False, True:
    for full_matrices in False, True:
      for dtype in dtypes_to_test:
        for rows in 1, 2, 5, 10, 32, 100:
          for cols in 1, 2, 5, 10, 32, 100:
            for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
              shape = batch_dims + (rows, cols)
              # TF2 does not support placeholders under eager so we skip it
              for use_static_shape in set([True, tf2.enabled()]):
                name = "%s_%s_static_shape_%s__compute_uv_%s_full_%s" % (
                    dtype.__name__, "_".join(map(str, shape)), use_static_shape,
                    compute_uv, full_matrices)
                _AddTest(SvdOpTest, "Svd", name,
                         _GetSvdOpTest(dtype, shape, use_static_shape,
                                       compute_uv, full_matrices))
  for compute_uv in False, True:
    for full_matrices in False, True:
      dtypes = ([np.float32, np.float64] + [np.complex64, np.complex128] *
                (not compute_uv) * (not test.is_built_with_rocm()))
      for dtype in dtypes:
        mat_shapes = [(10, 11), (11, 10), (11, 11), (2, 2, 2, 3)]
        if not full_matrices or not compute_uv:
          mat_shapes += [(5, 11), (11, 5)]
        for mat_shape in mat_shapes:
          for batch_dims in [(), (3,)]:
            shape = batch_dims + mat_shape
            name = "%s_%s_compute_uv_%s_full_%s" % (
                dtype.__name__, "_".join(map(str, shape)), compute_uv,
                full_matrices)
            _AddTest(SvdGradOpTest, "SvdGrad", name,
                     _GetSvdGradOpTest(dtype, shape, compute_uv, full_matrices))
            # The results are too inaccurate for float32.
            if dtype in (np.float64, np.complex128):
              _AddTest(
                  SvdGradGradOpTest, "SvdGradGrad", name,
                  _GetSvdGradGradOpTest(dtype, shape, compute_uv,
                                        full_matrices))
  test.main()
