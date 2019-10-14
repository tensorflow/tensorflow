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
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python import keras
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import def_function
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager import function
from tensorflow.python.eager import profiler
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import gradient_descent

CPU = "/device:CPU:0"
GPU = "/device:GPU:0"


def c_tfe_py_fastpath_execute(a,
                              b,
                              transpose_a=False,
                              transpose_b=False,
                              name=None):
  ctx = context.context()
  assert ctx.executing_eagerly(
  ), "The prototype doesn't contain C code for graph construction"
  try:
    return pywrap_tensorflow.TFE_Py_FastPathExecute(
        ctx._handle, ctx.device_name, "MatMul", name,
        ctx.op_callbacks, a, b, "transpose_a", transpose_a,
        "transpose_b", transpose_b)
  except core._NotOkStatusException as e:
    if name is not None:
      message = e.message + " name: " + name
    else:
      message = e.message
    six.raise_from(core._status_to_exception(e.code, message), None)


class SubclassedKerasModel(keras.Model):

  def __init__(self, initializer="ones"):
    super(SubclassedKerasModel, self).__init__()
    self.layer_a = keras.layers.Dense(
        64, kernel_initializer=initializer, bias_initializer="zeros")
    self.layer_b = keras.layers.Dense(
        128, kernel_initializer=initializer, bias_initializer="zeros")
    self.layer_c = keras.layers.Dense(
        256, kernel_initializer=initializer, bias_initializer="zeros")
    self.layer_d = keras.layers.Dense(
        256, kernel_initializer=initializer, bias_initializer="zeros")
    self.layer_e = keras.layers.Dense(
        10, kernel_initializer=initializer, bias_initializer="zeros")

  def call(self, x):
    x = self.layer_a(x)
    x = self.layer_b(x)
    x = self.layer_c(x)
    x = self.layer_d(x)
    return self.layer_e(x)


def make_keras_model(initializer="ones"):
  model_input = keras.Input(shape=(10,))
  x = keras.layers.Dense(
      64, kernel_initializer=initializer, bias_initializer="zeros")(model_input)
  x = keras.layers.Dense(
      128, kernel_initializer=initializer, bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      256, kernel_initializer=initializer, bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      256, kernel_initializer=initializer, bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      10, kernel_initializer=initializer, bias_initializer="zeros")(x)
  return keras.Model(inputs=model_input, outputs=x)


def make_sequential_keras_model(initializer="ones"):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(
      64, kernel_initializer=initializer, bias_initializer="zeros",
      input_shape=(10,)))
  model.add(keras.layers.Dense(
      128, kernel_initializer=initializer, bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      256, kernel_initializer=initializer, bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      256, kernel_initializer=initializer, bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      10, kernel_initializer=initializer, bias_initializer="zeros"))
  return model


def run_benchmark(func, num_iters, execution_mode=None):
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to warm up
    for _ in xrange(100):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in xrange(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start


class MicroBenchmarks(test.Benchmark):

  def __init__(self):
    # used for multiply benchmarks
    self._m_2 = random_ops.random_uniform([2])

    # used for matmul benchmarks
    self._m_2_by_2 = random_ops.random_uniform((2, 2))
    self._m_100_by_784 = random_ops.random_uniform((100, 784))
    self._num_iters_2_by_2 = 30000
    self._num_iters_100_by_784 = 30000

  def _run(self, func, num_iters, execution_mode=None):
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={"examples_per_sec": num_iters / total_time})

  def benchmark_create_np_array(self):
    func = lambda: np.array([3.0])
    self._run(func, 30000)

  def _benchmark_create_tensor(self, value, dtype, device):
    """Benchmark overheads of creating a Tensor object."""
    ctx = context.context()
    if device == GPU:
      # Warmup the GPU
      ops.EagerTensor(value, device=device)

    def func():
      ops.EagerTensor(value, device=device, dtype=dtype)

    self._run(func, 30000)

  def _benchmark_create_constant(self, value, dtype):
    def func():
      constant_op.constant(value, dtype=dtype)

    with ops.device("GPU:0" if context.num_gpus() else "CPU:0"):
      for _ in range(1000):
        func()  # Warmup.
      self._run(func, 3000)

  def benchmark_create_float_constant(self):
    self._benchmark_create_constant(42.0, dtype=None)

  def benchmark_create_int32_constant(self):
    if context.num_gpus():
      return  # int32 constants are always allocated on CPU.

    self._benchmark_create_constant(42, dtype=dtypes.int32)

  def _benchmark_add_scalars(self, a, b):
    def func():
      return memoryview(math_ops.add(a, b))

    with ops.device("GPU:0" if context.num_gpus() else "CPU:0"):
      for _ in range(1000):
        func()  # Warmup.
      self._run(func, 30000)

  def benchmark_add_float_scalars(self):
    self._benchmark_add_scalars(42.0, 24.0)

  def benchmark_add_int32_scalars(self):
    self._benchmark_add_scalars(42, 24)

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

  def benchmark_index_tensor_with_literal(self):
    func = lambda: constant_op.constant([3.0])[0]
    self._run(func, 30000)

  def benchmark_index_tensor_with_tensor(self):
    func = lambda idx=constant_op.constant(0): constant_op.constant([3.0])[idx]
    self._run(func, 30000)

  def benchmark_index_tensor_with_np_array(self):
    func = lambda idx=np.array(0): constant_op.constant([3.0])[idx]
    self._run(func, 30000)

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

  def benchmark_slowpath_tf_identity(self):
    self._run(lambda: gen_array_ops.identity(1), 30000)

  def benchmark_tfe_py_execute_identity(self):
    m = self._m_2
    ctx_handle = context.context()._handle
    attrs = ("T", self._m_2.dtype.as_datatype_enum)
    inputs = [m]

    def f():
      pywrap_tensorflow.TFE_Py_Execute(ctx_handle, None, "Identity", inputs,
                                       attrs, 1)

    self._run(f, 30000)

  def benchmark_tf_gradient_function_identity(self):
    with context.device(CPU):
      m = gen_array_ops.identity(self._m_2)
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
    with context.device(CPU):
      m = gen_array_ops.identity(self._m_2)
      self._run(lambda: backprop.gradients_function(lambda x: x, [0])(m), 30000)

  def _benchmark_np_matmul(self, m, transpose_b, num_iters):
    a = m.cpu().numpy()
    b = a.T if transpose_b else a
    func = lambda: np.dot(a, b)
    self._run(func, num_iters)

  def _benchmark_tf_matmul(self, m, transpose_b, num_iters,
                           execution_mode=None):
    func = lambda: math_ops.matmul(m, m, transpose_b=transpose_b)
    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_gen_math_ops_matmul(self, m, transpose_b, num_iters):

    def func():
      gen_math_ops.mat_mul(m, m, transpose_b=transpose_b)

    self._run(func, num_iters)

  def _benchmark_tfe_py_fastpath_execute_matmul(self, m, transpose_b,
                                                num_iters):

    def func():
      c_tfe_py_fastpath_execute(m, m, transpose_b=transpose_b)

    self._run(func, num_iters)

  def _benchmark_tfe_py_execute_matmul(self, m, transpose_b, num_iters):
    inputs = [m, m]
    # pylint: disable=protected-access
    ctx_handle = context.context()._handle
    # pylint: enable=protected-access
    device = context.context().device_name
    attrs = ("transpose_a", False, "transpose_b", transpose_b, "T",
             m.dtype.as_datatype_enum)

    def func():
      pywrap_tensorflow.TFE_Py_Execute(ctx_handle, device, "MatMul", inputs,
                                       attrs, 1)

    self._run(func, num_iters)

  def _benchmark_defun_matmul(self,
                              m,
                              transpose_b,
                              num_iters,
                              execution_mode=None):
    f = function.defun(math_ops.matmul)
    func = lambda: f(m, m, transpose_b=transpose_b)
    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_nested_defun_matmul(self, m, transpose_b, num_iters):
    inner = function.defun(math_ops.matmul)

    @function.defun
    def outer(a, b, c, transpose_b):
      return math_ops.matmul(inner(a, b, transpose_b=transpose_b), c)

    func = lambda: outer(m, m, m, transpose_b=transpose_b)
    # Warmup before benchmark
    for _ in range(1000):
      func()
    self._run(func, num_iters)

  def _benchmark_defun_matmul_forward_backward(self,
                                               m,
                                               transpose_b,
                                               num_iters,
                                               execution_mode=None):
    f = function.defun(math_ops.matmul)

    def func():
      with backprop.GradientTape() as gt:
        gt.watch(m)
        y = f(m, m, transpose_b=transpose_b)
      _ = gt.gradient(y, m)

    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_read_variable(self, m, num_iters):
    self._run(m.value, num_iters)

  def _benchmark_matmul_read_variable(self, m, num_iters):
    self._benchmark_gen_math_ops_matmul(
        m, transpose_b=False, num_iters=num_iters)

  def _benchmark_matmul_read_variable_with_tape(self, m, num_iters):
    with backprop.GradientTape() as tape:
      tape.watch(m)
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=False, num_iters=num_iters)

  def _benchmark_read_variable_with_tape(self, m, num_iters):
    with backprop.GradientTape() as tape:
      tape.watch(m)
      self._run(m.value, num_iters)

  # Benchmarks for A^2, A of dimension 2 by 2.
  def benchmark_np_matmul_2_by_2(self):
    self._benchmark_np_matmul(
        self._m_2_by_2, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_matmul_2_by_2_CPU(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tf_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_matmul_2_by_2_CPU_async(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tf_matmul(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def benchmark_gen_math_ops_matmul_2_by_2_CPU(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tfe_py_fastpath_execute_matmul_2_by_2_CPU(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tfe_py_fastpath_execute_matmul(
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

  def benchmark_defun_matmul_2_by_2_CPU_async(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def benchmark_defun_matmul_forward_backward_2_by_2_CPU(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul_forward_backward(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_forward_backward_2_by_2_CPU_async(self):
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul_forward_backward(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def benchmark_tf_matmul_2_by_2_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_tf_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_matmul_2_by_2_GPU_async(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_tf_matmul(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

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

  def benchmark_defun_matmul_2_by_2_GPU_async(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_defun_matmul(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def benchmark_nested_defun_matmul_2_by_2(self):
    m = self._m_2_by_2.cpu()
    self._benchmark_nested_defun_matmul(
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

  def benchmark_tf_matmul_100_by_784_CPU_async(self):
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_tf_matmul(
          m,
          transpose_b=True,
          num_iters=self._num_iters_100_by_784,
          execution_mode=context.ASYNC)

  def benchmark_gen_math_ops_matmul_100_by_784_CPU(self):
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tfe_py_fastpath_execute_matmul_100_by_784_CPU(self):
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_tfe_py_fastpath_execute_matmul(
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

  def benchmark_tf_matmul_100_by_784_GPU_async(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_tf_matmul(
          m,
          transpose_b=True,
          num_iters=self._num_iters_100_by_784,
          execution_mode=context.ASYNC)

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

  def benchmark_nested_defun_matmul_100_by_784(self):
    m = self._m_100_by_784.gpu()
    self._benchmark_nested_defun_matmul(
        m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def _benchmark_forwardprop_matmul_CPU(self, shape):
    with ops.device(CPU):
      m = random_ops.random_uniform(shape).cpu()
      tangent = random_ops.random_uniform(shape).cpu()

      def func():
        with forwardprop.ForwardAccumulator(m, tangent) as acc:
          result = math_ops.matmul(m, m, transpose_b=True)
        return result, acc.jvp(result)

      # Warmup before benchmark
      for _ in range(100):
        func()
      self._run(func, 3000)

  def _benchmark_forwardprop_in_defun_matmul_CPU(self, shape):
    with ops.device(CPU):
      @def_function.function
      def compiled_function(x, tangent):
        with forwardprop.ForwardAccumulator(x, tangent) as acc:
          result = math_ops.matmul(x, x, transpose_b=True)
        return result, acc.jvp(result)

      m = random_ops.random_uniform(shape).cpu()
      tangent = random_ops.random_uniform(shape).cpu()
      func = lambda: compiled_function(m, tangent)

      # Warmup before benchmark
      for _ in range(100):
        func()
      self._run(func, 3000)

  def _benchmark_forwardprop_in_defun_of_defun_matmul_CPU(self, shape):
    with ops.device(CPU):
      matmul = def_function.function(math_ops.matmul)

      @def_function.function()
      def compiled_function(x, tangent):
        with forwardprop.ForwardAccumulator(x, tangent) as acc:
          result = matmul(x, x, transpose_b=True)
        return result, acc.jvp(result)

      m = random_ops.random_uniform(shape).cpu()
      tangent = random_ops.random_uniform(shape).cpu()
      func = lambda: compiled_function(m, tangent)

      # Warmup before benchmark
      for _ in range(100):
        func()
      self._run(func, 3000)

  def _benchmark_forwardprop_of_defun_matmul_CPU(self, shape):
    with ops.device(CPU):
      m = random_ops.random_uniform(shape).cpu()
      tangent = random_ops.random_uniform(shape).cpu()
      matmul = def_function.function(math_ops.matmul)

      def func():
        with forwardprop.ForwardAccumulator(m, tangent) as acc:
          result = matmul(m, m, transpose_b=True)
        return result, acc.jvp(result)

      # Warmup before benchmark
      for _ in range(100):
        func()
      self._run(func, 3000)

  def benchmark_forwardprop_matmul_256_by_2096_CPU(self):
    self._benchmark_forwardprop_matmul_CPU(shape=(256, 2096))

  def benchmark_forwardprop_in_defun_matmul_256_by_2096_CPU(self):
    self._benchmark_forwardprop_in_defun_matmul_CPU(shape=(256, 2096))

  def benchmark_forwardprop_in_defun_of_defun_matmul_256_by_2096_CPU(self):
    self._benchmark_forwardprop_in_defun_of_defun_matmul_CPU(shape=(256, 2096))

  def benchmark_forwardprop_of_defun_matmul_256_by_2096_CPU(self):
    self._benchmark_forwardprop_of_defun_matmul_CPU(shape=(256, 2096))

  def benchmark_forwardprop_matmul_100_by_784_CPU(self):
    self._benchmark_forwardprop_matmul_CPU(shape=(100, 784))

  def benchmark_forwardprop_in_defun_matmul_100_by_784_CPU(self):
    self._benchmark_forwardprop_in_defun_matmul_CPU(shape=(100, 784))

  def benchmark_forwardprop_in_defun_of_defun_matmul_100_by_784_CPU(self):
    self._benchmark_forwardprop_in_defun_of_defun_matmul_CPU(shape=(100, 784))

  def benchmark_forwardprop_of_defun_matmul_100_by_784_CPU(self):
    self._benchmark_forwardprop_of_defun_matmul_CPU(shape=(100, 784))

  def _benchmark_tf_reduce_logsum_exp(self, device=CPU):
    with context.device(device):
      x = constant_op.constant([[1, 0.], [0., 0.]])
      func = lambda: math_ops.reduce_logsumexp(x)
      self._run(func, 3000)

  def benchmark_tf_reduce_logsumexp_CPU(self):
    self._benchmark_tf_reduce_logsum_exp()

  def benchmark_tf_reduce_logsumexp_GPU(self):
    self._benchmark_tf_reduce_logsum_exp(device=GPU)

  def benchmark_defun_without_signature(self):

    def func(t1, t2, t3, t4, t5, t6, t7, t8):
      del t1, t2, t3, t4, t5, t6, t7, t8
      return None

    defined = function.defun(func)
    t = constant_op.constant(0.0)
    cache_computation = lambda: defined(t, t, t, t, t, t, t, t)
    self._run(cache_computation, 30000)

  def benchmark_defun_without_signature_and_with_kwargs(self):

    def func(t1, t2, t3, t4, t5, t6, t7, t8):
      del t1, t2, t3, t4, t5, t6, t7, t8
      return None

    defined = function.defun(func)
    t = constant_op.constant(0.0)
    def cache_computation():
      return defined(t1=t, t2=t, t3=t, t4=t, t5=t, t6=t, t7=t, t8=t)
    self._run(cache_computation, 30000)

  def benchmark_defun_with_signature(self):

    def func(t1, t2, t3, t4, t5, t6, t7, t8):
      del t1, t2, t3, t4, t5, t6, t7, t8
      return None

    defined = function.defun(
        func, input_signature=[tensor_spec.TensorSpec([], dtypes.float32)] * 8)
    t = constant_op.constant(0.0)
    signature_computation = lambda: defined(t, t, t, t, t, t, t, t)
    self._run(signature_computation, 30000)

  def benchmark_defun_with_signature_and_kwargs(self):

    def func(t1, t2, t3, t4, t5, t6, t7, t8):
      del t1, t2, t3, t4, t5, t6, t7, t8
      return None

    defined = function.defun(
        func, input_signature=[tensor_spec.TensorSpec([], dtypes.float32)] * 8)
    t = constant_op.constant(0.0)
    def signature_computation():
      return defined(t1=t, t2=t, t3=t, t4=t, t5=t, t6=t, t7=t, t8=t)
    self._run(signature_computation, 30000)

  def benchmark_matmul_read_variable_op_2_by_2_CPU(self):
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_matmul_read_variable(m, num_iters=self._num_iters_2_by_2)

  def benchmark_matmul_read_variable_op_with_tape_2_by_2_CPU(self):
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_matmul_read_variable_with_tape(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_read_variable_op_2_by_2_CPU(self):
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_read_variable(m, num_iters=self._num_iters_2_by_2)

  def benchmark_read_variable_op_2_by_2_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2.gpu())
      self._benchmark_read_variable(m, num_iters=self._num_iters_2_by_2)

  def benchmark_read_variable_op_with_tape_2_by_2_CPU(self):
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_read_variable_with_tape(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_read_variable_op_with_tape_2_by_2_GPU(self):
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2.gpu())
      self._benchmark_read_variable_with_tape(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_keras_model_subclassed(self):
    model = SubclassedKerasModel()
    data = random_ops.random_uniform((10, 10))

    func = lambda: model(data)
    # First call is more expensive (creates variables etc.), discount that.
    func()

    # The whole point of this test is to contrast subclassing with
    # the functional style of keras model building, so validate that
    # the models are equivalent.
    assert np.equal(func(), make_keras_model()(data)).all()

    self._run(func, 30000)

  def benchmark_keras_model_functional(self):
    model = make_keras_model()
    data = random_ops.random_uniform((10, 10))
    func = lambda: model(data)
    # Symmetry with benchmark_keras_model_subclassed
    func()
    assert np.equal(func(), SubclassedKerasModel()(data)).all()
    self._run(func, 30000)

  def benchmark_keras_model_sequential(self):
    model = make_sequential_keras_model()
    data = random_ops.random_uniform((10, 10))
    func = lambda: model(data)
    # Symmetry with benchmark_keras_model_functional
    func()
    assert np.equal(func(), make_keras_model()(data)).all()
    self._run(func, 30000)

  def _benchmark_keras_model_fit(self, model, run_eagerly=False):
    data = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    labels = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    dataset = dataset_ops.Dataset.from_tensors((data, labels)).repeat()
    model.compile(
        gradient_descent.GradientDescentOptimizer(learning_rate=0.001),
        loss="mse", run_eagerly=run_eagerly)
    func = lambda: model.fit(dataset, epochs=1, steps_per_epoch=1000, verbose=0)
    # First call is more expensive (creates variables etc.), discount that.
    model.fit(dataset, epochs=1, steps_per_epoch=1, verbose=0)

    self._run(func, 1)

  def _benchmark_keras_model_evaluate(self, model, run_eagerly=False):
    data = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    labels = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    dataset = dataset_ops.Dataset.from_tensors((data, labels)).repeat()
    model.compile(
        gradient_descent.GradientDescentOptimizer(learning_rate=0.001),
        loss="mse", run_eagerly=run_eagerly)
    func = lambda: model.evaluate(dataset, steps=1000, verbose=0)
    # First call is more expensive (creates variables etc.), discount that.
    model.evaluate(dataset, steps=1, verbose=0)

    self._run(func, 1)

  def _benchmark_keras_model_predict(self, model, run_eagerly=False):
    data = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    dataset = dataset_ops.Dataset.from_tensors(tuple([data])).repeat()
    model.compile(
        gradient_descent.GradientDescentOptimizer(learning_rate=0.001),
        loss="mse", run_eagerly=run_eagerly)
    func = lambda: model.predict(dataset, steps=1000, verbose=0)
    # First call is more expensive (creates variables etc.), discount that.
    model.predict(dataset, steps=1, verbose=0)

    self._run(func, 1)

  def benchmark_keras_model_subclassed_fit(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_subclassed_fit_graph_mode(self):
    with context.graph_mode():
      model = SubclassedKerasModel(initializer="glorot_uniform")
      self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_subclassed_fit_run_model_eagerly(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model, run_eagerly=True)

  def benchmark_keras_model_functional_fit(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_functional_fit_graph_mode(self):
    with context.graph_mode():
      model = make_keras_model(initializer="glorot_uniform")
      self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_functional_fit_graph_mode_with_profiler(self):
    profiler.start()
    with context.graph_mode():
      model = make_keras_model(initializer="glorot_uniform")
      self._benchmark_keras_model_fit(model)
    result = profiler.stop()
    assert result is not None

  def benchmark_keras_model_functional_fit_run_model_eagerly(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model, run_eagerly=True)

  def benchmark_keras_model_functional_fit_run_model_eagerly_with_profiler(
      self):
    profiler.start()
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model, run_eagerly=True)
    result = profiler.stop()
    assert result is not None

  def benchmark_keras_model_sequential_fit(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_sequential_fit_graph_mode(self):
    with context.graph_mode():
      model = make_sequential_keras_model(initializer="glorot_uniform")
      self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_sequential_fit_run_model_eagerly(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model, run_eagerly=True)

  def benchmark_keras_model_subclassed_evaluate(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model)

  def benchmark_keras_model_subclassed_evaluate_run_model_eagerly(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model, run_eagerly=True)

  def benchmark_keras_model_functional_evaluate(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model)

  def benchmark_keras_model_functional_evaluate_run_model_eagerly(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model, run_eagerly=True)

  def benchmark_keras_model_sequential_evaluate(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model)

  def benchmark_keras_model_sequential_evaluate_run_model_eagerly(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model, run_eagerly=True)

  def benchmark_keras_model_subclassed_predict(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model)

  def benchmark_keras_model_subclassed_predict_run_model_eagerly(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model, run_eagerly=True)

  def benchmark_keras_model_functional_predict(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model)

  def benchmark_keras_model_functional_predict_run_model_eagerly(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model, run_eagerly=True)

  def benchmark_keras_model_sequential_predict(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model)

  def benchmark_keras_model_sequential_predict_run_model_eagerly(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model, run_eagerly=True)

  def benchmarkScan(self):
    elems = math_ops.range(1600)

    def scan():
      return functional_ops.scan(
          lambda a, x: a + x, elems, parallel_iterations=1)

    self._run(scan, 100)

  def benchmarkScanDefun(self):
    elems = math_ops.range(1600)

    @function.defun
    def scan():
      return functional_ops.scan(
          lambda a, x: a + x, elems, parallel_iterations=1)

    self._run(scan, 100)

  def benchmark_fastpath_conversion_type_inference(self):
    c = constant_op.constant(1., dtype=dtypes.float32)

    def fn():
      return gen_math_ops.add(c, 1)

    self._run(fn, 10000)

  def benchmark_convert_3x_list_to_tensor(self):
    xs = [1, 2, 3]
    self._run(lambda: ops.convert_to_tensor(xs), 1000)

  def benchmark_convert_3x_array_to_tensor(self):
    xs = np.array([1, 2, 3], dtype=np.int32)
    self._run(lambda: ops.convert_to_tensor(xs), 1000)

  def benchmark_constant_40x2_list_to_tensor(self):
    xs = [[0] * 2] * 40
    self._run(lambda: constant_op.constant(xs), 1000)

  def benchmark_constant_40x2_array_to_tensor(self):
    xs = np.array([[0] * 2] * 40, dtype=np.int32)
    self._run(lambda: constant_op.constant(xs), 1000)

  def benchmark_constant_40x_list_of_2x_arrays_to_tensor(self):
    xs = [np.array([0] * 2, dtype=np.int32)] * 40
    self._run(lambda: constant_op.constant(xs), 1000)

  def _benchmarkFunctionWithResourceInputs(self, num_resources, num_iters):
    @def_function.function
    def add_all(*args):
      return math_ops.add_n(*args)

    with context.device(CPU):
      resources = []
      for _ in range(num_resources):
        resources.append(resource_variable_ops.ResourceVariable(self._m_2))
      self._run(lambda: add_all(resources), num_iters)

  def benchmarkFunctionWithFiveResourceInputs(self):
    self._benchmarkFunctionWithResourceInputs(5, 1000)

  def benchmarkFunctionWithFiveHundredResourceInputs(self):
    self._benchmarkFunctionWithResourceInputs(500, 100)


if __name__ == "__main__":
  test.main()
