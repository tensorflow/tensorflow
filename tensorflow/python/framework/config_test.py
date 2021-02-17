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

from absl.testing import parameterized

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


def reset_eager(fn):
  def wrapper(*args, **kwargs):
    try:
      return fn(*args, **kwargs)
    finally:
      # Reset the context.
      context._context = None
      ops.enable_eager_execution_internal()
      assert context._context is not None

  return wrapper


class ConfigTest(test.TestCase, parameterized.TestCase):

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

    context.ensure_initialized()

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
    with self.assertRaisesRegex(errors.InvalidArgumentError,
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
    with self.assertRaisesRegex(errors.InvalidArgumentError,
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
  def testIntraOpParallelismThreads(self):
    config.set_intra_op_parallelism_threads(10)
    self.assertEqual(
        config.get_intra_op_parallelism_threads(),
        context.context().intra_op_parallelism_threads)

    context.ensure_initialized()

    with self.assertRaises(RuntimeError):
      config.set_intra_op_parallelism_threads(1)

    config.set_intra_op_parallelism_threads(10)

  @reset_eager
  def testInterOpParallelismThreads(self):
    config.set_inter_op_parallelism_threads(10)
    self.assertEqual(
        config.get_inter_op_parallelism_threads(),
        context.context().inter_op_parallelism_threads)

    context.ensure_initialized()

    with self.assertRaises(RuntimeError):
      config.set_inter_op_parallelism_threads(1)

    config.set_inter_op_parallelism_threads(10)

  @test_util.run_gpu_only
  @reset_eager
  def testSoftPlacement(self):
    if context.executing_eagerly():
      self.assertTrue(config.get_soft_device_placement())
    else:
      self.assertFalse(config.get_soft_device_placement())

    def mod():
      with ops.device('/device:GPU:0'):
        a = constant_op.constant(1.0)
        b = constant_op.constant(1.0)
        return math_ops.mod(a, b)

    config.set_soft_device_placement(True)
    self.assertEqual(config.get_soft_device_placement(), True)
    self.assertEqual(
        config.get_soft_device_placement(),
        context.context().soft_device_placement)

    # Since soft placement is enabled, the mod operation should fallback to CPU
    # with pure eager execution as well as functions
    mod()
    def_function.function(mod)()

    config.set_soft_device_placement(False)
    self.assertEqual(config.get_soft_device_placement(), False)
    self.assertEqual(
        config.get_soft_device_placement(),
        context.context().soft_device_placement)

    # Since soft placement is disabled, the mod operation should fail on GPU
    # with pure eager execution as well as functions
    with self.assertRaises(errors.InvalidArgumentError):
      mod()
    with self.assertRaises(errors.InvalidArgumentError):
      def_function.function(mod)()

  @reset_eager
  def testLogDevicePlacement(self):
    self.assertFalse(context.get_log_device_placement())

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

    context.ensure_initialized()

    # Changing the device placement should not throw an exception
    context.set_log_device_placement(True)

  @reset_eager
  def testEnableMlirBridge(self):
    # Default value of enable_mlir_bridge is false.
    self.assertFalse(context.context().config.experimental.enable_mlir_bridge)
    self.assertEqual(
        context.context().config.experimental.mlir_bridge_rollout,
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_UNSPECIFIED)

    # Tests enabling mlir bridge.
    config.enable_mlir_bridge()
    self.assertTrue(context.context().config.experimental.enable_mlir_bridge)
    self.assertEqual(
        context.context().config.experimental.mlir_bridge_rollout,
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_ENABLED)

    # Tests disabling mlir bridge.
    config.disable_mlir_bridge()
    self.assertFalse(context.context().config.experimental.enable_mlir_bridge)
    self.assertEqual(
        context.context().config.experimental.mlir_bridge_rollout,
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_DISABLED)

  @reset_eager
  def testEnableMlirGraphOptimization(self):
    # Default value of enable_mlir_graph_optimization is false.
    self.assertFalse(
        context.context().config.experimental.enable_mlir_graph_optimization)

    # Tests enabling mlir graph optimization.
    config.enable_mlir_graph_optimization()
    self.assertTrue(
        context.context().config.experimental.enable_mlir_graph_optimization)

    # Tests disabling mlir graph optimization.
    config.disable_mlir_graph_optimization()
    self.assertFalse(
        context.context().config.experimental.enable_mlir_graph_optimization)

  @test_util.run_gpu_only
  @reset_eager
  def testJit(self):
    self.assertEqual(config.get_optimizer_jit(), '')

    # the following function should cause Op fusion to occur. However, there is
    # unfortunately no straightforward way to ensure this. We will just have to
    # settle for creating a test that can trigger JIT.
    @def_function.function
    def fun(a, b):
      c = a * b
      d = c + a
      return d

    a = constant_op.constant([2., 2.])
    b = constant_op.constant([2., 2.])

    self.evaluate(fun(a, b))

    config.set_optimizer_jit('autoclustering')
    self.assertEqual(config.get_optimizer_jit(), 'autoclustering')

    self.evaluate(fun(a, b))

    config.set_optimizer_jit('')
    self.assertEqual(config.get_optimizer_jit(), '')

    self.evaluate(fun(a, b))

  @parameterized.named_parameters(
      ('LayoutOptimizer', 'layout_optimizer'),
      ('ConstantFolding', 'constant_folding'),
      ('ShapeOptimization', 'shape_optimization'),
      ('Remapping', 'remapping'),
      ('ArithmeticOptimization', 'arithmetic_optimization'),
      ('DependencyOptimization', 'dependency_optimization'),
      ('LoopOptimization', 'loop_optimization'),
      ('FunctionOptimization', 'function_optimization'),
      ('DebugStripper', 'debug_stripper'),
      ('ScopedAllocatorOptimization', 'scoped_allocator_optimization'),
      ('ImplementationSelector', 'implementation_selector'),
      ('AutoMixedPrecision', 'auto_mixed_precision'))
  @reset_eager
  def testOptimizerToggleOption(self, field):
    # TODO(b/128531235): Improve testing of option
    options = config.get_optimizer_experimental_options()
    self.assertIsNone(options.get(field))

    config.set_optimizer_experimental_options({field: True})
    options[field] = True
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(
        context.context().get_optimizer_experimental_options(), options)

    config.set_optimizer_experimental_options({field: False})
    options[field] = False
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(
        context.context().get_optimizer_experimental_options(), options)

  @parameterized.named_parameters(
      ('DisableModelPruning', 'disable_model_pruning'),
      ('DisableMetaOptimizer', 'disable_meta_optimizer'))
  @reset_eager
  def testOptimizerBoolOption(self, field):
    # TODO(b/128531235): Improve testing of option
    options = config.get_optimizer_experimental_options()
    self.assertFalse(options.get(field))

    config.set_optimizer_experimental_options({field: True})
    options[field] = True
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(
        context.context().get_optimizer_experimental_options(), options)

    config.set_optimizer_experimental_options({field: False})
    options[field] = False
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(
        context.context().get_optimizer_experimental_options(), options)

  @test_util.run_gpu_only
  @reset_eager
  def testOptimizerToggleOptionPinToHost(self):
    options = config.get_optimizer_experimental_options()
    self.assertIsNone(options.get('pin_to_host_optimization'))

    @def_function.function
    def fun():
      op = test_ops.device_placement_op()
      return op

    # Force optimizer to run for all graphs
    config.set_optimizer_experimental_options({'min_graph_nodes': -1})
    options['min_graph_nodes'] = -1

    # Since pin to host is disabled, the operation should go on GPU
    gpu = self.evaluate(fun())
    self.assertIn(compat.as_bytes('GPU'), gpu)

    config.set_optimizer_experimental_options(
        {'pin_to_host_optimization': True})
    options['pin_to_host_optimization'] = True
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(
        context.context().get_optimizer_experimental_options(), options)

    # Since pin to host is enabled, the operation should go on CPU
    cpu = self.evaluate(fun())
    self.assertIn(compat.as_bytes('CPU'), cpu)

    config.set_optimizer_experimental_options(
        {'pin_to_host_optimization': False})
    options['pin_to_host_optimization'] = False
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(
        context.context().get_optimizer_experimental_options(), options)

    # Since pin to host is disabled again, the operation should go on GPU
    gpu2 = self.evaluate(fun())
    self.assertIn(compat.as_bytes('GPU'), gpu2)


class DeviceTest(test.TestCase):

  @reset_eager
  def testPhysicalDevices(self):
    cpus = config.list_physical_devices('CPU')
    self.assertGreater(len(cpus), 0)
    if test_util.is_gpu_available():
      gpus = config.list_physical_devices('GPU')
      self.assertGreater(len(gpus), 0)

  @reset_eager
  def testCpuMultiple(self):
    cpus = config.list_physical_devices('CPU')
    self.assertEqual(len(cpus), 1)

    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('CPU')
    self.assertEqual(len(vcpus), 2)

    with ops.device('/device:CPU:0'):
      a = constant_op.constant(1.0)
      self.evaluate(a)
    with ops.device('/device:CPU:1'):
      b = constant_op.constant(1.0)
      self.evaluate(b)
    with ops.device('/device:CPU:2'):
      c = constant_op.constant(1.0)
      self.evaluate(c)
    self.assertIn('CPU:0', c.device)

    # Ensure we can place ops on each of the device names
    for vcpu in vcpus:
      with ops.device(vcpu.name):
        d = constant_op.constant(1.0)
        self.evaluate(d)

    # Modifying the CPU configuration is not supported
    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      config.set_logical_device_configuration(cpus[0], [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration()
      ])

    # Setting the same CPU configuration is fine
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

  @test_util.run_gpu_only
  @reset_eager
  def testGpuNone(self):
    config.set_soft_device_placement(False)
    gpus = config.list_physical_devices('GPU')
    self.assertGreater(len(gpus), 0)

    cpus = config.list_physical_devices('CPU')
    self.assertEqual(len(cpus), 1)

    self.assertEqual(len(config.get_visible_devices('CPU')), 1)
    self.assertGreater(len(config.get_visible_devices('GPU')), 0)

    self.assertEqual(len(config.get_visible_devices('XLA_GPU')), 0)

    config.set_visible_devices(cpus[0])
    self.assertEqual(len(config.get_visible_devices('CPU')), 1)
    self.assertEqual(len(config.get_visible_devices('GPU')), 0)
    self.assertEqual(len(config.list_logical_devices('XLA_GPU')), 0)

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Could not satisfy'):
      with ops.device('/device:GPU:0'):
        a = array_ops.identity(1.0)
        self.evaluate(a)

    # Modifying the visible devices is not supported
    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      config.set_visible_devices(gpus)

    # Setting the same visible devices is fine
    config.set_visible_devices(cpus[0])

  @reset_eager
  def testGpuMultiple(self):
    gpus = config.list_physical_devices('GPU')
    if len(gpus) < 2:
      self.skipTest('Need at least 2 GPUs')

    context.ensure_initialized()

    for i in range(0, len(gpus)):
      with ops.device('/device:GPU:' + str(i)):
        a = constant_op.constant(1.0)
        self.evaluate(a)

    with self.assertRaisesRegex(RuntimeError, 'unknown device'):
      with ops.device('/device:GPU:' + str(len(gpus))):
        a = constant_op.constant(1.0)
        self.evaluate(a)

  @reset_eager
  def testDeviceDetails(self):
    (cpu,) = config.list_physical_devices('CPU')
    details = config.get_device_details(cpu)
    self.assertEqual(details, {})

    if not test_util.is_gpu_available():
      return

    gpus = config.list_physical_devices('GPU')
    details = config.get_device_details(gpus[0])
    self.assertIsInstance(details['device_name'], str)
    self.assertNotEmpty(details['device_name'])
    if test.is_built_with_rocm():
      # AMD GPUs do not have a compute capability
      self.assertNotIn('compute_capability', details)
    else:
      cc = details['compute_capability']
      self.assertIsInstance(cc, tuple)
      major, minor = cc
      self.assertGreater(major, 0)
      self.assertGreaterEqual(minor, 0)

    # Test GPU returned from get_visible_devices
    if len(gpus) > 2:
      config.set_visible_devices(gpus[1], 'GPU')
      (visible_gpu,) = config.get_visible_devices('GPU')
      details = config.get_device_details(visible_gpu)
      self.assertIsInstance(details['device_name'], str)

  @reset_eager
  def testDeviceDetailsErrors(self):
    logical_devices = config.list_logical_devices()
    with self.assertRaisesRegex(ValueError,
                                'must be a tf.config.PhysicalDevice'):
      config.get_device_details(logical_devices[0])

    phys_dev = context.PhysicalDevice('/physical_device:CPU:100', 'CPU')
    with self.assertRaisesRegex(
        ValueError, 'The PhysicalDevice must be one obtained from '
        'calling `tf.config.list_physical_devices`'):
      config.get_device_details(phys_dev)

  @test_util.run_gpu_only
  @reset_eager
  def testVirtualGpu(self):
    config.set_soft_device_placement(False)
    gpus = config.list_physical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)

    self.assertIsNone(config.get_logical_device_configuration(gpus[-1]))
    config.set_logical_device_configuration(gpus[-1], [
        context.LogicalDeviceConfiguration(memory_limit=10),
        context.LogicalDeviceConfiguration(memory_limit=10)
    ])
    self.assertEqual(len(config.get_logical_device_configuration(gpus[-1])), 2)

    logical_gpus = config.list_logical_devices('GPU')
    self.assertTrue(len(logical_gpus), len(gpus) + 1)
    for i in range(0, len(logical_gpus)):
      with ops.device('/device:GPU:' + str(i)):
        a = array_ops.identity(1.0)
        self.evaluate(a)

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Could not satisfy'):
      with ops.device('/device:GPU:' + str(len(logical_gpus))):
        a = array_ops.identity(1.0)
        self.evaluate(a)

    # Modifying the GPU configuration is not supported
    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      config.set_logical_device_configuration(gpus[-1], [
          context.LogicalDeviceConfiguration(memory_limit=20),
          context.LogicalDeviceConfiguration(memory_limit=20)
      ])

    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      config.set_logical_device_configuration(gpus[-1], [
          context.LogicalDeviceConfiguration(memory_limit=10),
          context.LogicalDeviceConfiguration(memory_limit=10),
          context.LogicalDeviceConfiguration(memory_limit=10)
      ])

    # Setting the same GPU configuration is fine
    config.set_logical_device_configuration(gpus[-1], [
        context.LogicalDeviceConfiguration(memory_limit=10),
        context.LogicalDeviceConfiguration(memory_limit=10)
    ])

  @test_util.run_gpu_only
  @reset_eager
  def testGpuGrowth(self):
    gpus = config.list_physical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)

    self.assertIsNone(config.get_memory_growth(gpus[-1]))
    for gpu in gpus:
      config.set_memory_growth(gpu, True)

    c = context.context().config
    self.assertTrue(c.gpu_options.allow_growth)

    logical_gpus = config.list_logical_devices('GPU')
    self.assertTrue(len(logical_gpus), len(gpus))

    # Modifying the GPU configuration is not supported
    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      for gpu in gpus:
        config.set_memory_growth(gpu, False)

    # Setting the same GPU configuration is fine
    for gpu in gpus:
      config.set_memory_growth(gpu, True)

  @test_util.run_gpu_only
  @reset_eager
  def testGetMemoryInfoBasic(self):
    device = array_ops.zeros([]).backing_device
    info = config.get_memory_info(device)
    self.assertGreater(info['current'], 0)
    self.assertGreater(info['peak'], 0)
    self.assertEqual(info.keys(), {'current', 'peak'})
    self.assertEqual(config.get_memory_usage(device), info['current'])

  @test_util.run_gpu_only
  @reset_eager
  def testGetMemoryUsageSubstring(self):
    info = config.get_memory_info('GPU:0')
    self.assertGreater(info['current'], 0)

  @reset_eager
  def testGetMemoryInfoCPU(self):
    with self.assertRaisesRegex(ValueError, 'CPU does not support'):
      config.get_memory_info('CPU:0')
    with self.assertRaisesRegex(ValueError, 'CPU does not support'):
      config.get_memory_usage('CPU:0')

  @reset_eager
  def testGetMemoryInfoUnknownDevice(self):
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      config.get_memory_info('unknown_device')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      config.get_memory_usage('unknown_device')

  @test_util.run_gpu_only
  @reset_eager
  def testPeakMemoryUsage(self):
    x1 = array_ops.zeros((1000, 1000))
    peak1 = config.get_memory_info('GPU:0')['peak']
    self.assertGreaterEqual(peak1, 4 * 1000 * 1000)
    x2 = array_ops.ones((1000, 1000))
    peak2 = config.get_memory_info('GPU:0')['peak']
    self.assertGreaterEqual(peak2, peak1 + 4 * 1000 * 1000)
    del x1, x2  # With CPython, causes tensor memory to be immediately freed
    peak3 = config.get_memory_info('GPU:0')['peak']
    self.assertGreaterEqual(peak3, peak2)
    self.assertGreaterEqual(peak3, config.get_memory_info('GPU:0')['current'])

  @test_util.run_gpu_only
  @reset_eager
  def testGetMemoryUsageAmbiguousDevice(self):
    if len(config.list_physical_devices('GPU')) < 2:
      self.skipTest('Need at least 2 GPUs')
    with self.assertRaisesRegex(ValueError, 'Multiple devices'):
      config.get_memory_usage('GPU')

  @test_util.run_gpu_only
  @reset_eager
  def testGpuInvalidConfig(self):
    gpus = config.list_physical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)

    if len(gpus) > 1:
      # Assert if other GPUs were not configured
      config.set_memory_growth(gpus[0], True)
      with self.assertRaisesRegex(ValueError, 'cannot differ'):
        c = context.context().config

      # If we limit visibility to GPU 0, growth is fine
      config.set_visible_devices(gpus[0], 'GPU')
      c = context.context().config
      self.assertTrue(c.gpu_options.allow_growth)

      # Default setting for second GPU is False and works if we set visibility
      config.set_visible_devices(gpus[1], 'GPU')
      c = context.context().config
      self.assertFalse(c.gpu_options.allow_growth)

      # Growth now fails because all the GPUs are visible and not the same
      config.set_visible_devices(gpus, 'GPU')
      with self.assertRaisesRegex(ValueError, 'cannot differ'):
        c = context.context().config

    for gpu in gpus:
      config.set_memory_growth(gpu, True)

    c = context.context().config
    self.assertTrue(c.gpu_options.allow_growth)

    with self.assertRaisesRegex(ValueError, 'memory limit'):
      config.set_logical_device_configuration(gpus[-1], [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration()
      ])

    self.assertIsNone(config.get_logical_device_configuration(gpus[-1]))
    config.set_logical_device_configuration(gpus[-1], [
        context.LogicalDeviceConfiguration(memory_limit=10),
        context.LogicalDeviceConfiguration(memory_limit=10)
    ])

    c = context.context().config
    self.assertFalse(c.gpu_options.allow_growth)

    with self.assertRaisesRegex(ValueError, 'virtual devices'):
      config.set_memory_growth(gpus[-1], False)

  @test_util.run_gpu_only
  @reset_eager
  def testRemote(self):
    gpus = config.list_logical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)

    context.ensure_initialized()

    gpus = config.list_logical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)
    for gpu in gpus:
      self.assertIsNotNone(gpu.name)

    context.ensure_initialized()

    job_name = 'test'
    cluster_def = cluster_pb2.ClusterDef()
    job_def = cluster_def.job.add()
    job_def.name = job_name
    job_def.tasks[0] = 'localhost:0'

    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name=job_name, task_index=0, protocol='grpc')

    context.set_server_def(server_def)

    gpus = config.list_logical_devices('GPU')
    for gpu in gpus:
      self.assertIsNotNone(gpu.name)

  @reset_eager
  def testV1CompatibilityDummyInvisibleDeviceList(self):
    gpus = config.list_physical_devices('GPU')
    if gpus:
      self.skipTest('Test requires no GPUs')

    # Ensure GPU options left untouched on CPU only environments
    context.context()._physical_devices = None
    context.context()._config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(visible_device_list='0'))
    new_config = context.context().config
    self.assertEqual(new_config.gpu_options.visible_device_list, '0')

  @test_util.run_gpu_only
  @reset_eager
  def testV1Compatibility(self):
    # Ensure we set 1 CPU by default
    context.context()._config = config_pb2.ConfigProto()
    new_config = context.context().config
    self.assertEqual(new_config.device_count['CPU'], 1)
    context.context()._physical_devices = None

    # Ensure CPU is split
    context.context()._config = config_pb2.ConfigProto(device_count={'CPU': 2})
    new_config = context.context().config
    self.assertEqual(new_config.device_count['CPU'], 2)
    context.context()._physical_devices = None

    # Handle empty visible device list
    context.context()._config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(visible_device_list=''))
    gpus = config.list_physical_devices('GPU')
    gpu_count = len(gpus)
    new_config = context.context().config
    self.assertEqual(new_config.gpu_options.visible_device_list,
                     ','.join(str(i) for i in range(len(gpus))))
    context.context()._physical_devices = None

    # Handle invalid visible device list
    context.context()._config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(visible_device_list=str(gpu_count)))
    with self.assertRaisesRegex(ValueError, 'Invalid visible device index'):
      gpus = config.list_physical_devices('GPU')
      new_config = context.context().config
    context.context()._physical_devices = None

    # Handle single visible device list
    context.context()._config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(visible_device_list=str(gpu_count-1)))
    gpus = config.list_physical_devices('GPU')
    new_config = context.context().config
    self.assertEqual(new_config.gpu_options.visible_device_list,
                     str(gpu_count-1))
    context.context()._physical_devices = None

  def testConfigureCollectiveOps(self):
    context.context().configure_collective_ops(
        collective_leader='/job:worker/replica:0/task:0',
        scoped_allocator_enabled_ops=('CollectiveReduce',),
        use_nccl_communication=False,
        device_filters=['/job:worker/task:1'])
    new_config = context.context().config

    # Verify group leader
    self.assertEqual('/job:worker/replica:0/task:0',
                     new_config.experimental.collective_group_leader)

    # Verify device filters.
    self.assertEqual(['/job:worker/task:1'], new_config.device_filters)

    # Verify rewrite options.
    new_rewrite_options = new_config.graph_options.rewrite_options
    self.assertEqual(rewriter_config_pb2.RewriterConfig.ON,
                     new_rewrite_options.scoped_allocator_optimization)
    self.assertEqual(['CollectiveReduce'],
                     new_rewrite_options.scoped_allocator_opts.enable_op)


class TensorFloat32Test(test.TestCase):

  def setUp(self):
    super(TensorFloat32Test, self).setUp()
    if not test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      self.skipTest('TensorFloat-32 requires an NVIDIA GPU with compute '
                    'capability of at least 8.0')

  def tearDown(self):
    super(TensorFloat32Test, self).tearDown()
    config.enable_tensor_float_32_execution(True)

  def test_tensor_float_32_enabled(self):
    self.assertTrue(config.tensor_float_32_execution_enabled())

    x = array_ops.fill((8, 8), 1 + 2**-20)
    y = array_ops.ones((8, 8))
    out = math_ops.matmul(x, y)
    # In TensorFloat-32, each element of x is rounded to 1, so the output will
    # be 8s.
    expected = array_ops.fill((8, 8), 8)
    self.assertAllEqual(out, expected)

  def test_tensor_float_32_disabled(self):
    self.assertTrue(config.tensor_float_32_execution_enabled())
    config.enable_tensor_float_32_execution(False)
    self.assertFalse(config.tensor_float_32_execution_enabled())

    x = array_ops.fill((8, 8), 1 + 2**-20)
    y = array_ops.ones((8, 8))
    out = math_ops.matmul(x, y)
    expected = array_ops.fill((8, 8), 8 * (1 + 2**-20))
    self.assertAllEqual(out, expected)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
