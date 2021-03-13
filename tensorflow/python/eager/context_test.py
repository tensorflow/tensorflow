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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ContextTest(test.TestCase):

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
      f(constant_op.constant(1.))

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
  def testGetMemoryUsage(self):
    array_ops.zeros([10]) # Allocate some memory on the GPU.
    self.assertGreater(
        context.context().get_total_memory_usage('GPU:0'), 0)

  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryUsageCPU(self):
    with self.assertRaisesRegex(ValueError, 'CPU does not support'):
      context.context().get_total_memory_usage('CPU:0')

  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryUsageUnknownDevice(self):
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_total_memory_usage('unknown_device')

  @test_util.run_gpu_only
  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryUsageAmbiguousDevice(self):
    if len(context.context().list_physical_devices('GPU')) < 2:
      self.skipTest('Need at least 2 GPUs')
    with self.assertRaisesRegex(ValueError, 'Multiple devices'):
      context.context().get_total_memory_usage('GPU')

  def testListFunctionNames(self):

    @def_function.function
    def f():
      return constant_op.constant(1.)

    concrete = f.get_concrete_function()
    self.assertIn(concrete.name.decode(),
                  context.context().list_function_names())


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
