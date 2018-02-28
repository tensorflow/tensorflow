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
"""Tests for core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import execute as execute_lib
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


def execute(op_name, num_outputs, inputs, attrs=None):
  return execute_lib.execute(
      op_name, num_outputs, inputs, attrs, context.context())


def truncated_normal(shape):
  return execute(
      b'TruncatedNormal',
      1,
      inputs=[shape],
      attrs=('dtype', dtypes.float32.as_datatype_enum, 'T',
             shape.dtype.as_datatype_enum, 'seed', 0, 'seed2', 0))[0]


class TFETest(test_util.TensorFlowTestCase):

  def testContext(self):
    ctx = context.Context()
    self.assertFalse(ctx.in_graph_mode())
    self.assertTrue(ctx.in_eager_mode())

    self.assertEqual('', ctx.scope_name)
    ctx.scope_name = 'foo'
    self.assertEqual('foo', ctx.scope_name)

    self.assertIsNone(ctx.summary_writer_resource)
    ctx.summary_writer_resource = 'mock'
    self.assertEqual('mock', ctx.summary_writer_resource)

    self.assertEqual('', ctx.device_name)
    self.assertEqual(ctx.device_name, ctx.device_spec.to_string())
    with ctx.device('GPU:0'):
      self.assertEqual('/job:localhost/replica:0/task:0/device:GPU:0',
                       ctx.device_name)
      self.assertEqual(ctx.device_name, ctx.device_spec.to_string())
      with ctx.device(None):
        self.assertEqual('', ctx.device_name)
        self.assertEqual(ctx.device_name, ctx.device_spec.to_string())
        with ctx.device('CPU:0'):
          self.assertEqual('/job:localhost/replica:0/task:0/device:CPU:0',
                           ctx.device_name)
          self.assertEqual(ctx.device_name, ctx.device_spec.to_string())

    has_cpu_device = False
    for x in ctx.devices():
      has_cpu_device = has_cpu_device or 'CPU' in x
    self.assertTrue(has_cpu_device)
    del ctx

  def testRunMetadata(self):
    context.enable_run_metadata()
    t = constant_op.constant(1.0)
    _ = t + t  # Runs an operation which will be in the RunMetadata
    run_metadata = context.export_run_metadata()
    context.disable_run_metadata()
    step_stats = run_metadata.step_stats
    self.assertGreater(len(step_stats.dev_stats), 0)
    cpu_stats = step_stats.dev_stats[0]
    self.assertEqual('/job:localhost/replica:0/task:0/device:CPU:0',
                     cpu_stats.device)
    self.assertEqual(len(cpu_stats.node_stats), 1)
    self.assertEqual(cpu_stats.node_stats[0].node_name, 'Add')

  def testShouldCopy(self):
    if not context.context().num_gpus():
      self.skipTest('No devices other than CPUs found')
    with ops.device('gpu:0'):
      x = constant_op.constant(1.0)
    y = array_ops.identity(x)
    # The value we're testing y.device against will depend on what the behavior
    # of not explicitly specifying a device in the context is.  This behavior is
    # subject to change (for example, in the future we may want to use GPUs, if
    # available, when no device is explicitly provided)
    self.assertEqual(y.device, '/job:localhost/replica:0/task:0/device:CPU:0')

  def testContextStackContainsEagerMode(self):
    # Eager execution has been enabled, and no other context
    # switch has occurred, so `context_stack` should contain
    # exactly one entry.
    self.assertEqual(len(context.context_stack.stack), 1)
    stack_entry = context.context_stack.stack[0]

    # The entry should log that eager mode was entered.
    self.assertIs(stack_entry.enter_context_fn, context.eager_mode)

    # It is not possible to build a graph function when eager execution
    # is enabled; the stack entry should reflect this fact.
    self.assertFalse(stack_entry.is_building_function)

  def testInt32GPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    with ops.device('gpu:0'):
      xent = nn_ops.sparse_softmax_cross_entropy_with_logits(
          logits=[[0.0, 0.0]], labels=[0])
    self.assertAllClose(xent, [0.69314718])

  def _runInThread(self, target, args):
    t = threading.Thread(target=target, args=args)
    try:
      t.start()
      t.join()
    except Exception as e:
      raise e

  # Test that different thread local values are initialized to the same values
  # in different threads.
  def testContextThreadLocalMembers(self):

    def get_context_values(ctx):
      return [
          ctx.in_graph_mode(),
          ctx.in_eager_mode(), ctx.scope_name, ctx.summary_writer_resource,
          ctx.device_name, ctx.num_gpus()
      ]

    def get_values(ctx, values):
      values.extend(get_context_values(ctx))

    context_values = []
    ctx = context.Context()
    self._runInThread(get_values, (ctx, context_values))
    self.assertAllEqual(context_values, get_context_values(ctx))

  def testContextConfig(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    ctx = context.Context(config=config_pb2.ConfigProto(
        device_count={'GPU': 0}))
    self.assertEquals(0, ctx.num_gpus())

  def testTensorPlacement(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    x = constant_op.constant(1.).gpu()
    with context.device('gpu:0'):
      y = constant_op.constant(2.)
    # Add would fail if t2 were not on GPU
    result = execute(
        b'Add', 1, inputs=[x, y],
        attrs=('T', x.dtype.as_datatype_enum))[0].cpu().numpy()
    self.assertEqual(3, result)

  def testCopyBetweenDevices(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    x = constant_op.constant([[1., 2.], [3., 4.]])
    x = x.cpu()
    x = x.gpu()
    x = x.gpu()
    x = x.cpu()

    # Invalid device
    with self.assertRaises(RuntimeError):
      x.gpu(context.context().num_gpus() + 1)

  def testCopyScope(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    constant = constant_op.constant(1.0)
    with ops.device('gpu:0'):
      with context.context().device_policy(context.DEVICE_PLACEMENT_SILENT):
        c = constant + 1.0
    self.assertAllEqual(c, 2.0)

  def testNumpyForceCPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    cpu = constant_op.constant([[1., 2.], [3., 4.]])
    c2g = cpu.gpu()
    self.assertAllEqual(c2g, cpu.numpy())

  def testCopyFromCPUToCPU(self):
    ta = constant_op.constant([[1, 2], [3, 4]])
    tb = ta.cpu()

    self.assertNotEqual(id(ta), id(tb))
    self.assertAllEqual(ta, tb.numpy())

  def testRegisterExceptionClass(self):
    with self.assertRaises(TypeError):
      pywrap_tensorflow.TFE_Py_RegisterExceptionClass(str)
    pywrap_tensorflow.TFE_Py_RegisterExceptionClass(core._NotOkStatusException)  # pylint: disable=protected-access

  # TODO(agarwal): add tests passing incorrect typed values to attrs.
  def testExecuteBasic(self):
    three = constant_op.constant(3)
    five = constant_op.constant(5)
    product = execute(
        b'Mul',
        num_outputs=1,
        inputs=[three, five],
        attrs=('T', three.dtype.as_datatype_enum))[0]
    self.assertAllEqual(15, product)

  def testExecuteTooManyNumOutputs(self):
    # num_outputs provided is 50, but only one output is produced.
    # That should be okay.
    product = execute(
        b'Mul',
        num_outputs=50,
        inputs=[constant_op.constant(3), constant_op.constant(5)],
        attrs=('T', dtypes.int32.as_datatype_enum))[0]
    self.assertAllEqual(15, product)

  def testMatMulGPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    three = constant_op.constant([[3.]]).gpu()
    five = constant_op.constant([[5.]]).gpu()
    product = execute(
        b'MatMul',
        num_outputs=1,
        inputs=[three, five],
        attrs=('transpose_a', False, 'transpose_b', False, 'T',
               three.dtype.as_datatype_enum))[0]
    self.assertAllEqual([[15.0]], product)

  def testExecuteStringAttr(self):
    checked_three = execute(
        b'CheckNumerics',
        num_outputs=1,
        inputs=[constant_op.constant(3.)],
        attrs=('message', 'just checking', 'T',
               dtypes.float32.as_datatype_enum))[0]
    self.assertEqual([[3]], checked_three.numpy())

  def testExecuteStringAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      _ = execute(
          b'CheckNumerics',
          num_outputs=1,
          inputs=[constant_op.constant(3.)],
          attrs=('message', 1, 'T', dtypes.float32.as_datatype_enum))

  def testExecuteFloatAttr(self):
    almost_equal = execute(
        b'ApproximateEqual',
        num_outputs=1,
        inputs=[constant_op.constant(3.0), constant_op.constant(2.9)],
        attrs=('tolerance', 0.3, 'T', dtypes.float32.as_datatype_enum))[0]
    self.assertTrue(almost_equal)

  def testExecuteFloatAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      _ = execute(
          b'ApproximateEqual',
          num_outputs=1,
          inputs=[constant_op.constant(3.0), constant_op.constant(2.9)],
          attrs=('tolerance', '0.3', 'T', dtypes.float32.as_datatype_enum))

  def testExecuteIntAttr(self):
    total = execute(
        b'AddN',
        num_outputs=1,
        inputs=[constant_op.constant(3), constant_op.constant(4)],
        attrs=('T', dtypes.int32.as_datatype_enum, 'N', 2))[0]
    self.assertAllEqual(7, total)

  def testExecuteIntAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      _ = execute(
          b'AddN',
          num_outputs=1,
          inputs=[constant_op.constant(3), constant_op.constant(4)],
          attrs=('T', dtypes.int32.as_datatype_enum, 'N', '2'))

  # Looks like we don't have an existing op with list(bool) attrs.
  def testExecuteBoolAttr(self):
    product = execute(
        b'MatMul',
        num_outputs=1,
        inputs=[constant_op.constant([[3]]),
                constant_op.constant([[5]])],
        attrs=('transpose_a', True, 'transpose_b', False, 'T',
               dtypes.int32.as_datatype_enum))[0]
    self.assertAllEqual([[15]], product)

  def testExecuteShapeAttr(self):
    execute(
        b'VarHandleOp',
        num_outputs=1,
        inputs=[],
        attrs=('shape', [1, 2], 'dtype', dtypes.int32.as_datatype_enum,
               'container', '', 'shared_name', ''))

  def testExecuteShapeAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'VarHandleOp',
          num_outputs=1,
          inputs=[],
          attrs=('shape', 1, 'dtype', dtypes.int32.as_datatype_enum,
                 'container', '', 'shared_name', ''))

  def testExecuteListStringAttr(self):
    execute(
        b'TensorSummary',
        num_outputs=1,
        inputs=[constant_op.constant(3.0)],
        attrs=('T', dtypes.float32.as_datatype_enum, 'description',
               'tensor_summary', 'labels', ['3',
                                            'summary'], 'display_name', 'test'))

  def testExecuteListStringAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'TensorSummary',
          num_outputs=1,
          inputs=[constant_op.constant(3.0)],
          attrs=('T', dtypes.float32.as_datatype_enum, 'description', '',
                 'labels', 3, 'display_name', 'test'))

  def testExecuteListStringAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'TensorSummary',
          num_outputs=1,
          inputs=[constant_op.constant(3.0)],
          attrs=('T', dtypes.float32.as_datatype_enum, 'description', '',
                 'labels', [3], 'display_name', 'test'))

  def testExecuteListFloatAttr(self):
    b = execute(
        b'Bucketize',
        num_outputs=1,
        inputs=[constant_op.constant([3.0, 5.0, 7.0])],
        attrs=('T', dtypes.float32.as_datatype_enum, 'boundaries', [4.0,
                                                                    6.0]))[0]
    self.assertAllEqual([0, 1, 2], b)

  def testExecuteListFloatAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Bucketize',
          num_outputs=1,
          inputs=[constant_op.constant([3.0, 5.0, 7.0])],
          attrs=('T', dtypes.float32.as_datatype_enum, 'boundaries', 4.0))

  def testExecuteListFloatAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Bucketize',
          num_outputs=1,
          inputs=[constant_op.constant([3.0, 5.0, 7.0])],
          attrs=('T', dtypes.float32.as_datatype_enum, 'boundaries',
                 ['4.0', '6.0']))

  def testExecuteListIntAttr(self):
    b = execute(
        b'Squeeze',
        num_outputs=1,
        inputs=[constant_op.constant([[[3.0]]])],
        attrs=('T', dtypes.float32.as_datatype_enum, 'squeeze_dims', [0, 2]))[0]
    self.assertAllEqual([3], b)

  def testExecuteListIntAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Squeeze',
          num_outputs=1,
          inputs=[constant_op.constant([[[3.0]]])],
          attrs=('T', dtypes.float32.as_datatype_enum, 'squeeze_dims', 0))

  def testExecuteListIntAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Squeeze',
          num_outputs=1,
          inputs=[constant_op.constant([[[3.0]]])],
          attrs=('T', dtypes.float32.as_datatype_enum, 'squeeze_dims',
                 ['0', '2']))

  def testExecuteListTypeListShapeAttr(self):
    execute(
        b'Barrier',
        num_outputs=1,
        inputs=[],
        attrs=('component_types', [dtypes.float64.as_datatype_enum], 'shapes',
               [[1, 2]], 'capacity', -1, 'container', '', 'shared_name', ''))

  def testExecuteListTypeAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Barrier',
          num_outputs=1,
          inputs=[],
          attrs=('component_types', dtypes.float64.as_datatype_enum, 'shapes',
                 [[1, 2]], 'capacity', -1, 'container', '', 'shared_name', ''))

  def testExecuteListTypeAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Barrier',
          num_outputs=1,
          inputs=[],
          attrs=('component_types', '1', 'shapes', [[1, 2]], 'capacity', -1,
                 'container', '', 'shared_name', ''))

  def testExecuteListShapeAttrBadValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Barrier',
          num_outputs=1,
          inputs=[],
          attrs=('component_types', [dtypes.float64.as_datatype_enum], 'shapes',
                 [1, 2], 'capacity', -1, 'container', '', 'shared_name', ''))

  def testExecuteListShapeAttrBadListValue(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Barrier',
          num_outputs=1,
          inputs=[],
          attrs=('component_types', [dtypes.float64.as_datatype_enum], 'shapes',
                 [1], 'capacity', -1, 'container', '', 'shared_name', ''))

  def testExecuteMultipleOutputs(self):
    split_dim = 1
    value = [[0, 1, 2], [3, 4, 5]]
    x1, x2, x3 = execute(
        b'Split',
        num_outputs=3,
        inputs=[constant_op.constant(split_dim),
                constant_op.constant(value)],
        attrs=('num_split', 3, 'T', dtypes.int32.as_datatype_enum))
    self.assertAllEqual([[0], [3]], x1)
    self.assertAllEqual([[1], [4]], x2)
    self.assertAllEqual([[2], [5]], x3)

  def testExecuteBadNumOutputsArgument(self):
    with self.assertRaises(TypeError):
      execute(
          b'Relu', [],
          inputs=[constant_op.constant(3.0)],
          attrs=('T', dtypes.float32.as_datatype_enum))

  def testExecuteUnknownOp(self):
    with self.assertRaises(errors.NotFoundError):
      execute(b'BlahBlahBlah', num_outputs=1, inputs=[], attrs=None)

  def testExecuteUnknownAttr(self):
    with self.assertRaises(errors.InvalidArgumentError):
      execute(
          b'Identity',
          num_outputs=1,
          inputs=[constant_op.constant(3)],
          attrs=('T', dtypes.int32.as_datatype_enum, 'unknown_attr', 'blah'))

  def testComposition(self):

    def add(x, y):
      return execute(
          b'Add',
          num_outputs=1,
          inputs=[x, y],
          attrs=('T', dtypes.int32.as_datatype_enum))[0]

    x = constant_op.constant(1)
    three_x = add(add(x, x), x)
    self.assertEquals(dtypes.int32, three_x.dtype)
    self.assertAllEqual(3, three_x)

  def testOperationWithNoInputsRunsOnDevice(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    shape = constant_op.constant([], dtype=dtypes.int32)

    # x: Run the "TruncatedNormal" op CPU and copy result to GPU.
    x = truncated_normal(shape).gpu()
    # y: Explicitly run the "TruncatedNormal" op on GPU.
    with context.device('gpu:0'):
      y = truncated_normal(shape)
    # Add would fail if x and y were not on the same device.
    execute(
        b'Add', 1, inputs=[x, y], attrs=('T', x.dtype.as_datatype_enum))

  def testInvalidDevice(self):
    with self.assertRaises(ValueError):
      with context.device('pu:0'):
        _ = constant_op.constant(1)

  def testConvertMixedEagerTensors(self):
    array = np.zeros((), dtype=np.float32)
    tensor = constant_op.constant(0., dtype=dtypes.float32)
    types, tensors = execute_lib.convert_to_mixed_eager_tensors(
        [array, tensor], context.context())
    for typ, t in zip(types, tensors):
      self.assertEquals(typ, dtypes.float32)
      self.assertIsInstance(t, ops.EagerTensor)


if __name__ == '__main__':
  test.main()
