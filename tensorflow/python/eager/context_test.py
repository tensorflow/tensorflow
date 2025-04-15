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

import weakref

from absl.testing import parameterized
import numpy as np

from xla.service import hlo_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ContextTest(test.TestCase, parameterized.TestCase):

  def testSetGlobalSeed(self):
    c = context.Context()
    c._set_global_seed(123)
    for t in [np.int32, np.int64, np.uint32, np.uint64]:
      c._set_global_seed(t(123))
      c._set_global_seed(np.array(123, dtype=t))
      c._set_global_seed(ops.convert_to_tensor(123, dtype=t))

  def testContextIsDestroyedAfterTensors(self):
    # Create a new context
    new_context = context.Context()
    weak_c = weakref.ref(new_context)
    new_context.ensure_initialized()

    # Create a tensor with the new context as default.
    # Make sure to restore the original context.
    original_context = context.context()
    try:
      context._set_context(new_context)
      # Use a 2D tensor so that it is not cached.
      tensor1 = constant_op.constant([[3.]])
      # Produce a tensor as an operation output. This uses a different code path
      # from tensors created from Python.
      tensor2 = tensor1 * tensor1
      context._set_context(original_context)
    except:
      context._set_context(original_context)
      raise

    # Deleting our context reference should not delete the underlying object.
    del new_context
    self.assertIsNot(weak_c(), None)

    # Deleting the first tensor should not delete the context since there is
    # another tensor.
    del tensor1
    self.assertIsNot(weak_c(), None)

    # Deleting the last tensor should result in deleting its context.
    del tensor2
    self.assertIs(weak_c(), None)

  def testSimpleGraphCollection(self):

    @def_function.function
    def f(x):
      with ops.device('CPU:0'):
        return x + constant_op.constant(1.)

    with context.collect_graphs() as graphs:
      with ops.device('CPU:0'):
        x = constant_op.constant(1.)
      f(x)

    self.assertLen(graphs, 1)
    graph, = graphs
    self.assertIn('CPU:0', graph.node[1].device)

  @test_util.disable_tfrt(
      'b/171600738: tfrt does not support exporting post-optimization graph')
  def testGraphCollectionAfterDevicePlacement(self):

    @def_function.function
    def f(x):
      return x + constant_op.constant(1.)

    with context.collect_graphs() as graphs:
      with ops.device('CPU:0'):
        f(constant_op.constant(1.))

    self.assertLen(graphs, 1)
    graph, = graphs
    self.assertIn('CPU:0', graph.node[0].device)

  def testGetFunctionDef(self):

    @def_function.function
    def f():
      return constant_op.constant(1.)

    concrete = f.get_concrete_function()
    function_def = context.get_function_def(concrete.name)

    self.assertIsNot(function_def, None)

    found_const_node = False
    for node_def in function_def.node_def:
      if node_def.op == 'Const':
        found_const_node = True
        break
    self.assertTrue(found_const_node)

    with self.assertRaises(errors.NotFoundError):
      _ = context.get_function_def('this_should_not_be_found')

  @test_util.run_gpu_only
  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryInfo(self):
    array_ops.zeros([10])  # Allocate some memory on the GPU.
    self.assertGreater(context.context().get_memory_info('GPU:0')['current'], 0)

  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryInfoCPU(self):
    if test_util.IsMklEnabled():
      # TODO(gzmkl) work with Google team to address design issue in allocator.h
      self.skipTest('MklCPUAllocator does not throw exception. So skip test.')

    with self.assertRaisesRegex(ValueError, 'Allocator stats not available'):
      context.context().get_memory_info('CPU:0')

  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryInfoUnknownDevice(self):
    with self.assertRaisesRegex(ValueError, 'No matching devices found'):
      context.context().get_memory_info('unknown_device:0')

  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryInfoUnparsableDevice(self):
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU:')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU:CPU')

  def testListFunctionNames(self):

    @def_function.function
    def f():
      return constant_op.constant(1.)

    concrete = f.get_concrete_function()
    self.assertIn(concrete.name.decode(),
                  context.context().list_function_names())

  def testSetLogicalDeviceAfterContextInitialization(self):
    ctx = context.Context()
    ctx.set_logical_cpu_devices(4)
    self.assertIs(len(ctx.list_logical_devices('CPU')), 4)

    # Cannot set logical device twice.
    with self.assertRaisesRegex(RuntimeError, 'Virtual CPUs already set'):
      ctx.set_logical_cpu_devices(8)

  def testSetLogicalLocalCpuDevice(self):
    ctx = context.Context()

    # Manually add a remote CPU device into logical device list.
    ctx._logical_devices = []  # pylint: disable=protected-access
    dev = context.LogicalDevice(name='/job:worker/replica:0/task:1',
                                device_type='CPU')
    ctx._logical_devices.append(dev)  # pylint: disable=protected-access
    self.assertIs(len(ctx.list_logical_devices('CPU')), 1)

    # This would pass the check since the previously added device is not local.
    ctx.set_logical_cpu_devices(4)
    # Logical device list would be overwritten after initialization.
    self.assertIs(len(ctx.list_logical_devices('CPU')), 4)

  @parameterized.named_parameters([(f'_{stage}', stage) for stage in [
      'hlo', 'hlo_serialized', 'optimized_hlo', 'optimized_hlo_serialized',
      'optimized_hlo_proto_serialized', 'optimized_hlo_dot'
  ]])
  def testGetCompilerIr(self, stage):

    @def_function.function(jit_compile=True)
    def test_func(x):
      return 2 * x

    a = array_ops.ones((1000, 1000))  # 4 * 1000 * 1000 in bytes
    result = test_func.experimental_get_compiler_ir(a)(stage=stage)
    self.assertNotEmpty(result)
    if stage == 'optimized_hlo_proto_serialized':
      hlo_proto = hlo_pb2.HloProto.FromString(result)
      allocations = hlo_proto.buffer_assignment.buffer_allocations
      buffer_size = sum(
          getattr(allocation, 'size') for allocation in allocations)
      # The sizes of input and output are both 4 * 1000 * 1000 in bytes.
      self.assertGreaterEqual(buffer_size, 2 * 4 * 1000 * 1000)
      self.assertLess(buffer_size, 4 * 4 * 1000 * 1000)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
