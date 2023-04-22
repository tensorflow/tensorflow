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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test as test_lib


def _AddTest(test, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class MatrixBandPartTest(test_lib.TestCase):
  pass  # Filled in below


def _GetMatrixBandPartTest(dtype_, batch_shape_, shape_):

  @test_util.run_v1_only("b/120545219")
  def Test(self):
    mat = np.ones(shape_).astype(dtype_)
    batch_mat = np.tile(mat, batch_shape_ + (1, 1))
    for lower in -1, 0, 1, shape_[-2] - 1:
      for upper in -1, 0, 1, shape_[-1] - 1:
        band_np = mat
        if lower >= 0:
          band_np = np.triu(band_np, -lower)
        if upper >= 0:
          band_np = np.tril(band_np, upper)
        if batch_shape_ != ():
          band_np = np.tile(band_np, batch_shape_ + (1, 1))
        for index_dtype in [dtypes_lib.int32, dtypes_lib.int64]:
          with self.cached_session(use_gpu=False):
            band = array_ops.matrix_band_part(
                batch_mat,
                constant_op.constant(lower, index_dtype),
                constant_op.constant(upper, index_dtype))
            self.assertAllEqual(band_np, self.evaluate(band))

  return Test


class MatrixBandPartGradTest(test_lib.TestCase):
  pass  # Filled in below


def _GetMatrixBandPartGradTest(dtype_, batch_shape_, shape_):

  @test_util.run_v1_only("b/120545219")
  def Test(self):
    shape = batch_shape_ + shape_
    x = constant_op.constant(np.random.rand(*shape), dtype=dtype_)
    with self.session(use_gpu=False):
      for lower in -1, 0, 1, shape_[-2] - 1:
        for upper in -1, 0, 1, shape_[-1] - 1:
          y = array_ops.matrix_band_part(x, lower, upper)
          error = gradient_checker.compute_gradient_error(
              x, x.get_shape().as_list(), y, y.get_shape().as_list())
          self.assertLess(error, 1e-4)

  return Test


class MatrixBandPartBenchmark(test_lib.Benchmark):

  shapes = [
      (10, 16, 16),
      (10, 101, 101),
      (10, 256, 256),
      (10, 1000, 1000),
      (10, 1024, 1024),
      (10, 2048, 2048),
      (10, 10, 4, 4),
      (10, 10, 10, 10),
      (10, 10, 16, 16),
      (10, 10, 101, 101),
      (10, 10, 256, 256),
      (10, 10, 1000, 1000),
      (10, 10, 1024, 1024),
      (10, 10, 2048, 2048),
  ]

  def benchmarkMatrixBandPartOp(self):
    for shape_ in self.shapes:
      for limits in (-1, -1), (-1, 0), (0, -1), (2, 2):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/cpu:0"):
          matrix = variables.Variable(array_ops.ones(shape_))
          band = array_ops.matrix_band_part(matrix, limits[0], limits[1])
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(band),
              min_iters=10,
              name="matrix_band_part_cpu_{shape}_{limits}".format(
                  shape=shape_, limits=limits))

        if test_lib.is_gpu_available(True):
          with ops.Graph().as_default(), \
              session.Session(config=benchmark.benchmark_config()) as sess, \
              ops.device("/gpu:0"):
            matrix = variables.Variable(array_ops.ones(shape_))
            band = array_ops.matrix_band_part(matrix, limits[0], limits[1])
            self.evaluate(variables.global_variables_initializer())
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(band),
                min_iters=10,
                name="matrix_band_part_gpu_{shape}_{limits}".format(
                    shape=shape_, limits=limits))


if __name__ == "__main__":
  dtypes = (np.bool, np.int32, np.int64, np.float16,
            dtypes_lib.bfloat16.as_numpy_dtype, np.float32, np.float64,
            np.complex64, np.complex128)
  for dtype in dtypes:
    for batch_shape in ((), (2,), (1, 3, 2)):
      for rows in 1, 2, 7, 23:
        for cols in 1, 2, 7, 23:
          shape = (rows, cols)
          name = "%s_%s" % (dtype.__name__,
                            "_".join(map(str, batch_shape + shape)))
          _AddTest(MatrixBandPartTest, "MatrixBandPart", name,
                   _GetMatrixBandPartTest(dtype, batch_shape, shape))

  for dtype in (np.float32, np.float64):
    for batch_shape in ((), (2,)):
      for rows in 1, 2, 7:
        for cols in 1, 2, 7:
          shape = (rows, cols)
          name = "%s_%s" % (dtype.__name__,
                            "_".join(map(str, batch_shape + shape)))
          _AddTest(MatrixBandPartGradTest, "MatrixBandPartGrad", name,
                   _GetMatrixBandPartGradTest(dtype, batch_shape, shape))

  test_lib.main()
