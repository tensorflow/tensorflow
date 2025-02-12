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
    # host -> h2d transfer (creating a tf::Tensor with PjRtTensorBuffer)
    # -> XLA op (GPU) -> TF op (GPU) -> d2h transfer.
    host_tensor = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    with ops.device("/device:GPU:0"):
      xla_tensor = const_fn()
      xla_result = matmul_fn(host_tensor)
      result = math_ops.matmul(xla_result, xla_tensor)  # TF GPU kernel.

    # Using numpy to calculate the reference result ("ref_*").
    ref_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    ref_result = np.matmul(np.matmul(ref_tensor, ref_tensor), ref_tensor)
    self.assertAllClose(result.numpy(), ref_result, atol=1e-05)

    # The following use case is tested:
    # host -> h2d transfer (creating a tf::Tensor with PjRtTensorBuffer)
    # -> TF op (GPU, creating a tf::Tensor with device mem)
    # -> XLA op (GPU, accepting a tf::Tensor with device mem) -> d2h transfer.
    with ops.device("/device:GPU:0"):
      tf_matmul_tensor = math_ops.matmul(host_tensor, host_tensor)
      xla_result = matmul_fn(tf_matmul_tensor)

    ref_matmul_tensor = np.matmul(ref_tensor, ref_tensor)
    ref_result_2 = np.matmul(ref_matmul_tensor, ref_matmul_tensor)
    self.assertAllClose(xla_result.numpy(), ref_result_2, atol=1e-05)

  def test_xla_launch_with_var_on_gpu_device(self):
    @def_function.function(jit_compile=True)
    def foo(x, y):
      return x + y + 1

    @def_function.function(jit_compile=True)
    def bar(x, y):
      x.assign(y)
      y.assign_add([1.0, 1.0])

    with ops.device("/device:GPU:0"):
      a = constant_op.constant([1.0, 2.0])
      x = variables.Variable([0.0, 1.0])
      result_tensor = foo(x, a)

    self.assertAllClose(result_tensor.numpy(), [2.0, 4.0], atol=1e-05)

    # The following use case is tested:
    # Variable updates following an XLA computation that reads the updated
    # variables.
    with ops.device("/device:GPU:0"):
      var_a = variables.Variable([0.0, 1.0])
      var_b = variables.Variable([1.0, 2.0])
      bar(var_a, var_b)
      result = foo(var_a, var_b)

    self.assertAllClose([1.0, 2.0], var_a.value(), atol=1e-05)
    self.assertAllClose([2.0, 3.0], var_b.value(), atol=1e-05)
    self.assertAllClose(result, [4.0, 6.0], atol=1e-05)


if __name__ == "__main__":
  test.main()
