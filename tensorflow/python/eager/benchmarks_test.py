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
from tensorflow.python.eager import backprop  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops

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
        ctx._post_execution_callbacks, a, b, "transpose_a", transpose_a,
        "transpose_b", transpose_b)
  except core._NotOkStatusException as e:
    if name is not None:
      message = e.message + " name: " + name
    else:
      message = e.message
    six.raise_from(core._status_to_exception(e.code, message), None)


class SubclassedKerasModel(keras.Model):

  def __init__(self):
    super(SubclassedKerasModel, self).__init__()
    self.layer_a = keras.layers.Dense(
        64, kernel_initializer="ones", bias_initializer="zeros")
    self.layer_b = keras.layers.Dense(
        128, kernel_initializer="ones", bias_initializer="zeros")
    self.layer_c = keras.layers.Dense(
        256, kernel_initializer="ones", bias_initializer="zeros")
    self.layer_d = keras.layers.Dense(
        256, kernel_initializer="ones", bias_initializer="zeros")
    self.layer_e = keras.layers.Dense(
        10, kernel_initializer="ones", bias_initializer="zeros")

  def call(self, x):
    x = self.layer_a(x)
    x = self.layer_b(x)
    x = self.layer_c(x)
    x = self.layer_d(x)
    return self.layer_e(x)


def make_keras_model():
  model_input = keras.Input(shape=(10,))
  x = keras.layers.Dense(
      64, kernel_initializer="ones", bias_initializer="zeros")(model_input)
  x = keras.layers.Dense(
      128, kernel_initializer="ones", bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      256, kernel_initializer="ones", bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      256, kernel_initializer="ones", bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      10, kernel_initializer="ones", bias_initializer="zeros")(x)
  return keras.Model(inputs=model_input, outputs=x)


def make_sequential_keras_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(
      64, kernel_initializer="ones", bias_initializer="zeros",
      input_shape=(10,)))
  model.add(keras.layers.Dense(
      128, kernel_initializer="ones", bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      256, kernel_initializer="ones", bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      256, kernel_initializer="ones", bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      10, kernel_initializer="ones", bias_initializer="zeros"))
  return model


class MicroBenchmarks(test.Benchmark):

  def __init__(self):
    # used for multiply benchmarks
    self._m_2 = random_ops.random_uniform([2])

    # used for matmul benchmarks
    self._m_2_by_2 = random_ops.random_uniform((2, 2))
    self._m_100_by_784 = random_ops.random_uniform((100, 784))
    self._num_iters_2_by_2 = 30000
    self._num_iters_100_by_784 = 1000

  def _run(self, func, num_iters, execution_mode=None):
    # call func to maybe warm up the GPU
    ctx = context.context()
    with ctx.execution_mode(execution_mode):
      func()
      if execution_mode == context.ASYNC:
        ctx.async_wait()
      start = time.time()
      for _ in xrange(num_iters):
        func()
      if execution_mode == context.ASYNC:
        ctx.async_wait()
      end = time.time()
      mean_us = (end - start) * 1e6 / num_iters
      self.report_benchmark(
          iters=num_iters,
          wall_time=mean_us,
          extras={"examples_per_sec": num_iters / (end - start)})

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
    func = lambda: f(m, m, transpose_b)
    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_defun_matmul_forward_backward(self,
                                               m,
                                               transpose_b,
                                               num_iters,
                                               execution_mode=None):
    f = function.defun(math_ops.matmul)

    def func():
      with backprop.GradientTape() as gt:
        gt.watch(m)
        y = f(m, m, transpose_b)
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


if __name__ == "__main__":
  test.main()
