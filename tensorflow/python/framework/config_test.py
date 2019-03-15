# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests that the system configuration methods work properly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def reset_eager(fn):
  def wrapper(*args, **kwargs):
    try:
      return fn(*args, **kwargs)
    finally:
      del context._context
      context._context = context.Context()
      ops.enable_eager_execution()

  return wrapper


class ConfigTest(test.TestCase):

  @test_util.run_gpu_only
  @reset_eager
  def testDevicePolicy(self):
    self.assertEqual(context.DEVICE_PLACEMENT_SILENT,
                     context.context().device_policy)

    # If no op has been executed we should be able to set the device policy as
    # well as any init-time configs.
    config.set_intra_op_parallelism_threads(1)
    config.set_device_policy('silent')
    config.set_intra_op_parallelism_threads(2)

    # Excute a dummy op to ensure that the context has been initialized
    constant_op.constant(1)

    def copy_tensor(dtype=dtypes.int32):
      cpu_tensor = constant_op.constant(1, dtype=dtype)
      gpu_tensor = cpu_tensor.gpu()
      self.assertAllEqual(cpu_tensor + gpu_tensor, 2.0)

    config.set_device_policy('silent')
    self.assertEqual(config.get_device_policy(), 'silent')
    self.assertEqual(context.DEVICE_PLACEMENT_SILENT,
                     context.context().device_policy)
    copy_tensor()

    config.set_device_policy('silent_for_int32')
    self.assertEqual(config.get_device_policy(), 'silent_for_int32')
    self.assertEqual(context.DEVICE_PLACEMENT_SILENT_FOR_INT32,
                     context.context().device_policy)
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'Tensors on conflicting devices'):
      copy_tensor(dtypes.float32)
    copy_tensor()

    config.set_device_policy('warn')
    self.assertEqual(config.get_device_policy(), 'warn')
    self.assertEqual(context.DEVICE_PLACEMENT_WARN,
                     context.context().device_policy)
    copy_tensor()

    config.set_device_policy('explicit')
    self.assertEqual(config.get_device_policy(), 'explicit')
    self.assertEqual(context.DEVICE_PLACEMENT_EXPLICIT,
                     context.context().device_policy)
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'Tensors on conflicting devices'):
      copy_tensor()

    config.set_device_policy(None)
    self.assertEqual(config.get_device_policy(), 'silent')

  @reset_eager
  def testExecutionMode(self):
    self.assertTrue(config.get_synchronous_execution())
    self.assertEqual(context.SYNC, context.context().execution_mode)

    # If no op has been executed we should be able to set the execution mode as
    # well as any init-time configs.
    config.set_intra_op_parallelism_threads(1)
    config.set_synchronous_execution(False)
    config.set_intra_op_parallelism_threads(2)

    config.set_synchronous_execution(True)
    self.assertTrue(config.get_synchronous_execution())
    self.assertEqual(context.SYNC, context.context().execution_mode)
    config.set_synchronous_execution(False)
    self.assertFalse(config.get_synchronous_execution())
    self.assertEqual(context.ASYNC, context.context().execution_mode)

  @reset_eager
  def testGpuPerProcessMemoryFraction(self):
    config.set_gpu_per_process_memory_fraction(0.5)
    self.assertEqual(
        config.get_gpu_per_process_memory_fraction(),
        context.context().gpu_per_process_memory_fraction)

    constant_op.constant(1)
    with self.assertRaises(RuntimeError):
      config.set_gpu_per_process_memory_fraction(0.5)

  @reset_eager
  def testGpuPerProcessMemoryGrowth(self):
    self.assertFalse(config.get_gpu_per_process_memory_growth())

    config.set_gpu_per_process_memory_growth(True)
    self.assertTrue(config.get_gpu_per_process_memory_growth())
    self.assertEqual(
        config.get_gpu_per_process_memory_growth(),
        context.context().gpu_per_process_memory_growth)

    config.set_gpu_per_process_memory_growth(False)
    self.assertFalse(config.get_gpu_per_process_memory_growth())
    self.assertEqual(
        config.get_gpu_per_process_memory_growth(),
        context.context().gpu_per_process_memory_growth)

    constant_op.constant(1)
    with self.assertRaises(RuntimeError):
      config.set_gpu_per_process_memory_growth(True)

  @reset_eager
  def testIntraOpParallelismThreads(self):
    config.set_intra_op_parallelism_threads(10)
    self.assertEqual(
        config.get_intra_op_parallelism_threads(),
        context.context().intra_op_parallelism_threads)

    constant_op.constant(1)
    with self.assertRaises(RuntimeError):
      config.set_intra_op_parallelism_threads(1)

  @reset_eager
  def testInterOpParallelismThreads(self):
    config.set_inter_op_parallelism_threads(10)
    self.assertEqual(
        config.get_inter_op_parallelism_threads(),
        context.context().inter_op_parallelism_threads)

    constant_op.constant(1)
    with self.assertRaises(RuntimeError):
      config.set_inter_op_parallelism_threads(1)

  @test_util.run_gpu_only
  @reset_eager
  def testSoftPlacement(self):
    self.assertEqual(config.get_soft_device_placement(), True)

    @def_function.function
    def mod():
      with ops.device('/device:GPU:0'):
        a = constant_op.constant(1.0)
        b = constant_op.constant(1.0)
        return math_ops.mod(a, b)

    # Since soft placement is enabled, the mod operation should work with CPU
    mod()

    config.set_soft_device_placement(False)
    self.assertEqual(config.get_soft_device_placement(), False)
    self.assertEqual(
        config.get_soft_device_placement(),
        context.context().soft_device_placement)

    # Since soft placement is disabled, the mod operation should fail on GPU
    with self.assertRaises(errors.InvalidArgumentError):
      mod()

    config.set_soft_device_placement(True)
    self.assertEqual(config.get_soft_device_placement(), True)
    self.assertEqual(
        config.get_soft_device_placement(),
        context.context().soft_device_placement)

    # Since soft placement is re-enabled, the mod operation should work with CPU
    mod()

  @reset_eager
  def testLogDevicePlacement(self):
    self.assertEqual(context.get_log_device_placement(), False)

    context.set_log_device_placement(True)
    self.assertEqual(context.get_log_device_placement(), True)
    self.assertEqual(
        context.get_log_device_placement(),
        context.context().log_device_placement)

    context.set_log_device_placement(False)
    self.assertEqual(context.get_log_device_placement(), False)
    self.assertEqual(
        context.get_log_device_placement(),
        context.context().log_device_placement)

    constant_op.constant(1)
    with self.assertRaises(RuntimeError):
      context.set_log_device_placement(True)
    with self.assertRaises(RuntimeError):
      context.set_log_device_placement(False)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
