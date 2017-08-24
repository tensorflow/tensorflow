# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for low-level eager execution primitives.

Packaged as a test to ensure that this code is exercised by continuous
integration tests. To get numbers:

  bazel build -c opt :benchmarks_test &&
  ./bazel-bin/tensorflow/python/eager/benchmarks_test --iters=0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import contextlib
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import backprop  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import tensor
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

FLAGS = None


@contextlib.contextmanager
def timer(label, iters=30000):
  start = time.time()
  yield xrange(iters)
  end = time.time()
  t = (end - start) * 1e6 / iters
  print("%-40s took %.2fus (%d iterations)" % (label, t, iters))


def benchmark_create_tensor(n):
  """Benchmark overheads of creating a Tensor object."""

  def label(s):
    return "{:20s}".format(s)

  with timer(label("np.array([[3]])"), iters=n) as iters:
    for _ in iters:
      np.array([[3]])

  with timer(label("Tensor([[3]])"), iters=n) as iters:
    for _ in iters:
      tensor.Tensor([[3]])


def benchmark_matmul(shape, n, use_gpu=False):
  """Benchmark for matrix multiplication using tf.matmul."""
  transpose_b = (shape[0] != shape[1])
  m = random_ops.random_uniform(shape)
  if use_gpu:
    m = m.as_gpu_tensor()
    # Warm up the GPU - the very first kernel invocation
    # seems to require a bunch of setup.
    math_ops.matmul(m, m, transpose_b=transpose_b)

  def label(s):
    return "MatMul {}: {:30s}".format(shape, s)

  if not use_gpu:
    a = m.as_cpu_tensor().numpy()
    b = a.T if transpose_b else a
    with timer(label("np.dot"), iters=n) as iters:
      for _ in iters:
        np.dot(a, b)

  with timer(label("tf.matmul"), iters=n) as iters:
    for _ in iters:
      math_ops.matmul(m, m, transpose_b=transpose_b)

  with timer(label("gen_math_ops.mat_mul"), iters=n) as iters:
    for _ in iters:
      gen_math_ops._mat_mul(m, m, transpose_b=transpose_b)

  # pylint: disable=protected-access
  input_handles = [m._handle, m._handle]
  ctx_handle = context.context()._handle
  # pylint: enable=protected-access
  attrs = ("transpose_a", False, "transpose_b", transpose_b, "T",
           m.dtype.as_datatype_enum)
  with timer(label("TFE_Py_Execute"), iters=n) as iters:
    for _ in iters:
      pywrap_tensorflow.TFE_DeleteTensorHandle(
          pywrap_tensorflow.TFE_Py_Execute(ctx_handle, None, "MatMul",
                                           input_handles, attrs, 1)[0])

  f = function.defun(math_ops.matmul)
  with timer(label("defun(tf.matmul)"), iters=n) as iters:
    for _ in iters:
      f(m, m, transpose_b=transpose_b)


class BenchmarksTest(test_util.TensorFlowTestCase):

  def testBenchmarks(self):
    # This isn't actually a test, but benchmarks packaged as a test
    # so that continuous integration runs catch any breakages.
    print(context.context())
    benchmark_create_tensor(FLAGS.iters or 30000)
    benchmark_matmul([2, 2], FLAGS.iters or 30000)
    benchmark_matmul([100, 28 * 28], FLAGS.iters or 1000)

    if context.context().num_gpus() > 0:
      print("---- RUNNING ON GPU NOW ----")
      benchmark_matmul([2, 2], FLAGS.iters or 30000, use_gpu=True)
      benchmark_matmul([100, 28 * 28], FLAGS.iters or 1000, use_gpu=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Default iterations to 1 to keep continuos integration test times low.
  parser.add_argument(
      "--iters",
      type=int,
      default=1,
      help="Number of iterators for each test. None or 0 for auto-selection")
  FLAGS, unparsed = parser.parse_known_args()
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
