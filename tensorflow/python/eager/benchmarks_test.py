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
r"""Benchmarks for low-level eager execution primitives.

To run CPU benchmarks:
  bazel run -c opt benchmarks_test -- --benchmarks=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" benchmarks_test -- \
    --benchmarks=.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import backprop  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


CPU = "/device:CPU:0"
GPU = "/device:GPU:0"


class MicroBenchmarks(test.Benchmark):

  def __init__(self):
    # used for multiply benchmarks
    self._m_2 = random_ops.random_uniform([2])

    # used for matmul benchmarks
    self._m_2_by_2 = random_ops.random_uniform((2, 2))
    self._m_100_by_784 = random_ops.random_uniform((100, 784))
    self._num_iters_2_by_2 = 30000
    self._num_iters_100_by_784 = 1000

  def _run(self, func, num_iters):
    # call func to maybe warm up the GPU
    func()
    start = time.time()
    for _ in xrange(num_iters):
      func()
    end = time.time()
    mean_us = (end - start) * 1e6 / num_iters
    self.report_benchmark(iters=num_iters, wall_time=mean_us,
                          extras={"examples_per_sec": num_iters/(end-start)})

  def benchmark_create_np_array(self):
    func = lambda: np.array([3.0])
    self._run(func, 30000)

  def _benchmark_create_tensor(self, value, dtype, device):
    """Benchmark overheads of creating a Tensor object."""
    ctx = context.context()
    handle = ctx._handle
    if device == GPU:
      # Warmup the GPU
      ops.EagerTensor(value, context=handle, device=device)

    def func():
      ops.EagerTensor(value, context=handle, device=device, dtype=dtype)
    self._run(func, 30000)

  def benchmark_create_float_tensor_from_list_CPU(self):
    self._benchmark_create_tensor([[3.0]], dtypes.float32.as_datatype_enum, CPU)

  def benchmark_create_float_tensor_from_np_array_CPU(self):
    self._benchmark_create_tensor(
        np.array([[3.0]], dtype=np.float32), dtypes.float32.as_datatype_enum,
        CPU)

  def benchmark_create_int32_tensor_from_list_CPU(self):
    self._benchmark_create_tensor([[3]], dtypes.int32.as_datatype_enum, CPU)

  def benchmark_create_int32_tensor_from_np_array_CPU(self):
    self._benchmark_create_tensor(
        np.array([[3]], dtype=np.int32), dtypes.int32.as_datatype_enum, CPU)

  def benchmark_create_float_tensor_from_list_GPU(self):
    if not context.num_gpus():
      return
    self._benchmark_create_tensor([[3.0]], dtypes.float32.as_datatype_enum, GPU)

  def benchmark_create_float_tensor_from_np_array_GPU(self):
    if not context.num_gpus():
      return
    self._benchmark_create_tensor(
        np.array([[3.0]], dtype=np.float32), dtypes.float32.as_datatype_enum,
        GPU)

  def benchmark_create_int32_tensor_from_list_GPU(self):
    # int32's are kept on host memory even when executing on GPU.
    if not context.num_gpus():
      return
    self._benchmark_create_tensor([[3]], dtypes.int32.as_datatype_enum, GPU)

  def benchmark_create_int32_tensor_from_np_array_GPU(self):
    # int32's are kept on host memory even when executing on GPU.
    if not context.num_gpus():
      return
    self._benchmark_create_tensor(
        np.array([[3]], dtype=np.int32), dtypes.int32.as_datatype_enum, GPU)

  def _benchmark_np_multiply(self, m, num_iters):
    a = m.cpu().numpy()
    func = lambda: a * a
    self._run(func, num_iters)

  def _benchmark_tf_multiply(self, m, num_iters):
    func = lambda: m * m
    self._run(func, num_iters)

  def _benchmark_tf_multiply_op(self, m, num_iters):
    func = lambda: math_ops.multiply(m, m)
    self._run(func, num_iters)

  def benchmark_np_multiply(self):
    self._benchmark_np_multiply(self._m_2, 30000)

  def benchmark_tf_multiply_CPU(self):
    with context.device(CPU):
      m = self._m_2.cpu()
      self._benchmark_tf_multiply(m, 30000)

  def benchmark_tf_multiply_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2.gpu()
      self._benchmark_tf_multiply(m, 30000)

  def benchmark_tf_multiply_op_CPU(self):
    with context.device(CPU):
      m = self._m_2.cpu()
      self._benchmark_tf_multiply_op(m, 30000)

  def benchmark_tf_multiply_op_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2.gpu()
      self._benchmark_tf_multiply_op(m, 30000)

  def benchmark_tf_identity(self):
    m = self._m_2
    self._run(lambda: gen_array_ops.identity(m), 30000)

  def benchmark_tfe_py_execute_identity(self):
    m = self._m_2
    ctx_handle = context.context()._handle
    attrs = ("T", self._m_2.dtype.as_datatype_enum)
    inputs = [m]

    def f():
      pywrap_tensorflow.TFE_Py_Execute(
          ctx_handle, None, "Identity", inputs, attrs, 1)

    self._run(f, 30000)

  def benchmark_tf_gradient_function_identity(self):
    m = self._m_2
    self._run(
        lambda: backprop.gradients_function(gen_array_ops.identity, [0])(m),
        30000)

  def benchmark_tf_gradient_forward_identity(self):
    with backprop.GradientTape() as tape:
      m = self._m_2
      tape.watch(m)
      self._run(lambda: gen_array_ops.identity(m), 30000)

  def benchmark_tf_gradient_tape_push_pop(self):

    def f():
      with backprop.GradientTape():
        pass
    self._run(f, 30000)

  def benchmark_tf_gradient_function_no_op(self):
    m = self._m_2
    self._run(
        lambda: backprop.gradients_function(lambda x: x, [0])(m),
        30000)

  def _benchmark_np_matmul(self, m, transpose_b, num_iters):
    a = m.cpu().numpy()
    b = a.T if transpose_b else a
    func = lambda: np.dot(a, b)
    self._run(func, num_iters)

  def _benchmark_tf_matmul(self, m, transpose_b, num_iters):
    func = lambda: math_ops.matmul(m, m, transpose_b=transpose_b)
    self._run(func, num_iters)

  def _benchmark_gen_math_ops_matmul(self, m, transpose_b, num_iters):
    def func():
      gen_math_ops._mat_mul(m, m, transpose_b=transpose_b)
    self._run(func, num_iters)

  def _benchmark_tfe_py_execute_matmul(self, m, transpose_b, num_iters):
    inputs = [m, m]
    # pylint: disable=protected-access
    ctx_handle = context.context()._handle
    # pylint: enable=protected-access
    attrs = ("transpose_a", False, "transpose_b", transpose_b, "T",
             m.dtype.as_datatype_enum)
    def func():
      pywrap_tensorflow.TFE_Py_Execute(ctx_handle, None, "MatMul", inputs,
                                       attrs, 1)

    self._run(func, num_iters)

  def _benchmark_defun_matmul(self, m, transpose_b, num_iters):
    f = function.defun(math_ops.matmul)
    func = lambda: f(m, m, transpose_b)
    self._run(func, num_iters)

  # Benchmarks for A^2, A of dimension 2 by 2.
  def benchmark_np_matmul_2_by_2(self):
    self._benchmark_np_matmul(
        self._m_2_by_2, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_matmul_2_by_2_CPU(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tf_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_gen_math_ops_matmul_2_by_2_CPU(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tfe_py_execute_matmul_2_by_2_CPU(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tfe_py_execute_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_CPU(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_matmul_2_by_2_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_tf_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_gen_math_ops_matmul_2_by_2_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tfe_py_execute_matmul_2_by_2_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_tfe_py_execute_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_defun_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  # Benchmarks for AA.T, A of dimension 100 by 784.
  def benchmark_np_matmul_100_by_784(self):
    self._benchmark_np_matmul(
        self._m_100_by_784,
        transpose_b=True,
        num_iters=self._num_iters_100_by_784)

  def benchmark_tf_matmul_100_by_784_CPU(self):
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_tf_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_gen_math_ops_matmul_100_by_784_CPU(self):
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tfe_py_execute_matmul_100_by_784_CPU(self):
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_tfe_py_execute_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_defun_matmul_100_by_784_CPU(self):
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_defun_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tf_matmul_100_by_784_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_tf_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_gen_math_ops_matmul_100_by_784_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tfe_py_execute_matmul_100_by_784_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_tfe_py_execute_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_defun_matmul_100_by_784_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_defun_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)


if __name__ == "__main__":
  test.main()
