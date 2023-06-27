# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests single device compilation + execution using the Device API (aka PjRt).

This feature is still under active development and is protected behind the
`--tf_xla_use_device_api` flag in the `TF_XLA_FLAGS` environment variable.
"""
import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


class PjrtCompileTest(test.TestCase):

  def test_compile_on_demand(self):
    if not test.is_gpu_available() or not test.is_built_with_gpu_support():
      test.skipTest("Test only applicable on GPU")

    with ops.device("/device:XLA_GPU:0"):
      a = constant_op.constant([1.0, 2.0])
      b = constant_op.constant([2.0, 3.0])
      c = a + b
      self.assertAllClose([3.0, 5.0], c, atol=1e-05)

      v = variables.Variable([0.0, 1.0])
      v.assign([1.0, 2.0])
      self.assertAllClose([1.0, 2.0], v.value(), atol=1e-05)
      v.assign_add([1.0, 2.0])
      self.assertAllClose([2.0, 4.0], v.value(), atol=1e-05)

      d = c + v
      self.assertAllClose([5.0, 9.0], d, atol=1e-05)

  # Tests compilation and execution of a jit_compiled function using PjRt.
  def test_xla_local_launch(self):
    if not test.is_gpu_available() or not test.is_built_with_gpu_support():
      test.skipTest("Test only applicable on GPU")

    @def_function.function(jit_compile=True)
    def foo(x, y):
      return x + y + 1

    @def_function.function(jit_compile=True)
    def bar(x, y):
      x.assign(y)
      y.assign_add([1.0, 1.0])

    # Currently PjRt only supports compilation and execution for the XLA_GPU
    # device to unblock development. Support for non-XLA devices (CPU/GPU/single
    # core TPU) is going to be added soon, after which support for XLA_* devices
    # will be dropped.
    # TODO(b/255826209): Modify the test as we progress towards supporting
    # non-XLA devices.
    with ops.device("/device:XLA_GPU:0"):
      # Function call with scalars
      self.assertEqual(self.evaluate(foo(1, 2)), 4)

      # Function call with tensors
      a = constant_op.constant([1.0, 2.0])
      b = constant_op.constant([2.0, 3.0])
      self.assertAllClose([4.0, 6.0], foo(a, b), atol=1e-05)

      # Function call with variables
      x = variables.Variable([0.0, 1.0])
      y = variables.Variable([1.0, 2.0])
      self.assertAllClose([2.0, 4.0], foo(x, y), atol=1e-05)

      # Function call with constant and variable
      self.assertAllClose([2.0, 4.0], foo(a, x), atol=1e-05)

      # Function call that updates variables
      bar(x, y)
      self.assertAllClose([1.0, 2.0], x.value(), atol=1e-05)
      self.assertAllClose([2.0, 3.0], y.value(), atol=1e-05)

  # TODO(b/286458275): Add autoclustering tests when b/274176440 is fixed and
  # when we can use PJRT with the GPU device. XLA_GPU device doesn't work well
  # with autoclustering - sometimes it tries to wrap functions in an XlaLaunch
  # op instead of _XlaCompile/_XlaRun ops.
  def test_xla_compile_and_run(self):
    pass

  def test_xla_launch_and_tf_kernel_on_gpu_device(self):
    @def_function.function(jit_compile=True)
    def const_fn():
      return constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    @def_function.function(jit_compile=True)
    def matmul_fn(x):
      return math_ops.matmul(x, x)

    # The following tests XlaLaunch kernel interoperation with legacy TF
    # GPU kernel.
    #
    # The following use case is tested:
    # host -> h2d transfer -> XLA op (GPU) -> TF op (GPU) -> dh2 transfer.
    host_t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    with ops.device("/device:GPU:0"):
      xla_t = const_fn()
      xla_result = matmul_fn(host_t)
      result = math_ops.matmul(xla_result, xla_t)  # TF GPU kernel.

    ref_t = np.array([[1.0, 2.0], [3.0, 4.0]])
    ref_result = np.matmul(np.matmul(ref_t, ref_t), ref_t)
    self.assertAllClose(result.numpy(), ref_result, atol=1e-05)


if __name__ == "__main__":
  test.main()
