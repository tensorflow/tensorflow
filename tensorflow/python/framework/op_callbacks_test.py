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
"""Unit tests for op_callback."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import compat


# Keep all the hard-coded op type strings in one place so they are easy to
# change all at once in the face of any possible future op type name changes.
_ADD_OP = b"AddV2"
_ASSIGN_ADD_VARIABLE_OP = b"AssignAddVariableOp"
_CONSTANT_OP = b"Const"
_COS_OP = b"Cos"
_ENTER_OP = b"Enter"
_EXIT_OP = b"Exit"
_GREATER_OP = b"Greater"
_IDENTITY_OP = b"Identity"
_IF_OP = b"If"
_LESS_OP = b"Less"
_LOG_OP = b"Log"
_MERGE_OP = b"Merge"
_MATMUL_OP = b"MatMul"
_MUL_OP = b"Mul"
_NEXT_ITERATION_OP = b"NextIteration"
_PLACEHOLDER_OP = b"Placeholder"
_POW_OP = b"Pow"
_READ_VARIABLE_OP = b"ReadVariableOp"
_SIN_OP = b"Sin"
_SPARSE_TENSOR_DENSE_MATMUL_OP = b"SparseTensorDenseMatMul"
_SQRT_OP = b"Sqrt"
_SQUARE_OP = b"Square"
_STATELESS_IF_OP = b"StatelessIf"
_SWITCH_OP = b"Switch"
_UNIQUE_OP = b"Unique"
_VAR_HANDLE_OP = b"VarHandleOp"
_WHILE_OP = b"While"


class _NumpyFunctionCallback(object):

  def __init__(self, instrument_graph_ops=True, float_only=False):
    self.instrument_graph_ops = instrument_graph_ops
    self._float_only = float_only
    self.reset()

  def callback(self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
    is_eager = not graph
    if is_eager:
      self.eager_op_types.append(
          compat.as_bytes(op_type) if op_type else op_type)
      self.eager_op_names.append(
          compat.as_bytes(op_name) if op_name else op_name)
      self.eager_attrs.append(attrs)
      self.eager_graphs.append(graph)
      self.eager_inputs.append(inputs)
    else:
      self.graph_op_types.append(
          compat.as_bytes(op_type) if op_type else op_type)
      self.graph_op_names.append(
          compat.as_bytes(op_name) if op_name else op_name)
      self.graph_attrs.append(attrs)
      self.graph_graphs.append(graph)
      self.graph_graph_versions.append(graph.version)
      self.graph_inputs.append(inputs)

      if not self.instrument_graph_ops:
        return outputs

      # Instrument the graph with numpy_function.
      instrumented_outputs = []
      for output in outputs:
        if compat.as_bytes(op_type) in (_ENTER_OP, _EXIT_OP, _IF_OP, _MERGE_OP,
                                        _NEXT_ITERATION_OP, _STATELESS_IF_OP,
                                        _SWITCH_OP, _WHILE_OP, _IDENTITY_OP,
                                        _VAR_HANDLE_OP, _PLACEHOLDER_OP,
                                        _CONSTANT_OP):
          # TODO(cais): Overriding the output of StatelessIf, If and While ops
          # currently fails with error. Investigate (b/139668453).
          # Avoid instrumenting Identity ops as well, as they are inserted
          # by tf.function/AutoGraph for marshalling outputs.
          instrumented_output = output
        else:
          def record(ndarray_value):
            if compat.as_bytes(op_name) not in self.graph_internal_ndarrays:
              self.graph_internal_ndarrays[compat.as_bytes(op_name)] = []
            self.graph_internal_ndarrays[compat.as_bytes(op_name)].append(
                ndarray_value)
            return ndarray_value

          if self._float_only and not output.dtype.is_floating:
            instrumented_output = output
          else:
            instrumented_output = script_ops.numpy_function(
                record, [output], output.dtype)
            instrumented_output.set_shape(output.shape)
        instrumented_outputs.append(instrumented_output)

      return instrumented_outputs

  def reset(self):
    self.eager_op_types = []
    self.eager_op_names = []
    self.eager_attrs = []
    self.eager_graphs = []
    self.eager_inputs = []
    self.graph_op_types = []
    self.graph_op_names = []
    self.graph_attrs = []
    self.graph_graphs = []
    self.graph_graph_versions = []
    self.graph_inputs = []

    # A dict mapping tensor name (e.g., "MatMut_10") to a list of ndarrays.
    # The list is the history of the tensor's computation result inside
    # `tf.Graph`s (`FuncGraph`s).
    # For an op with multiple output tensors, the outputs are interleaved in
    # the list.
    self.graph_internal_ndarrays = {}


class OpCallbacksTest(test_util.TensorFlowTestCase):

  def tearDown(self):
    op_callbacks.clear_op_callbacks()
    super(OpCallbacksTest, self).tearDown()

  def testSingleThreadedStack(self):
    ctx = context.context()
    instrument_0 = _NumpyFunctionCallback()
    instrument_1 = _NumpyFunctionCallback()

    op_callbacks.add_op_callback(instrument_0.callback)
    self.assertEqual(1, len(ctx.op_callbacks))
    self.assertIn(instrument_0.callback, ctx.op_callbacks)

    op_callbacks.add_op_callback(instrument_1.callback)
    self.assertEqual(2, len(ctx.op_callbacks))
    self.assertIn(instrument_0.callback, ctx.op_callbacks)
    self.assertIn(instrument_1.callback, ctx.op_callbacks)

    op_callbacks.remove_op_callback(instrument_1.callback)
    self.assertEqual(1, len(ctx.op_callbacks))
    self.assertIn(instrument_0.callback, ctx.op_callbacks)

    op_callbacks.remove_op_callback(instrument_0.callback)
    self.assertEqual(0, len(ctx.op_callbacks))

  def testMultiThreadedStacks(self):
    # Instrument for the main thread.
    instrument_0 = _NumpyFunctionCallback()

    # Instrument for the to-be-created thread.
    instrument_1 = _NumpyFunctionCallback()

    def thread1_job():
      op_callbacks.add_op_callback(instrument_1.callback)

      @def_function.function
      def func1(x):
        return math_ops.sqrt(math_ops.log(x))

      x = constant_op.constant(4.0)
      self.assertAllClose(func1(x), np.sqrt(np.log(4.0)))

    thread1 = threading.Thread(target=thread1_job)

    # Start job on separate thread.
    thread1.start()

    # Run something on the main thread.
    op_callbacks.add_op_callback(instrument_0.callback)

    @def_function.function
    def func0(x):
      return math_ops.square(math_ops.sin(x))

    x = constant_op.constant(4.0)
    self.assertAllClose(func0(x), np.square(np.sin(4.0)))

    thread1.join()

    # Assert that there is no cross-talk between the main thread
    # and the created thread.
    self.assertIn(_PLACEHOLDER_OP, instrument_1.graph_op_types)
    self.assertIn(_LOG_OP, instrument_1.graph_op_types)
    self.assertIn(_SQRT_OP, instrument_1.graph_op_types)
    self.assertNotIn(_SIN_OP, instrument_1.graph_op_types)
    self.assertNotIn(_SQUARE_OP, instrument_1.graph_op_types)

    self.assertNotIn(_LOG_OP, instrument_0.graph_op_types)
    self.assertNotIn(_SQRT_OP, instrument_0.graph_op_types)
    self.assertIn(_SIN_OP, instrument_0.graph_op_types)
    self.assertIn(_SQUARE_OP, instrument_0.graph_op_types)

  def testEagerOpExecution(self):
    instrument = _NumpyFunctionCallback()

    x = constant_op.constant(6.0)
    op_callbacks.add_op_callback(instrument.callback)
    y = math_ops.square(math_ops.log(x))
    self.assertAllClose(y, np.square(np.log(6.0)))

    self.assertEqual(instrument.eager_op_types, [_LOG_OP, _SQUARE_OP])
    # Op names are unavailable under eager mode.
    self.assertEqual(instrument.eager_op_names, [None, None])
    self.assertEqual(instrument.eager_graphs, [None, None])
    self.assertEqual(len(instrument.eager_inputs), 2)
    self.assertEqual(len(instrument.eager_inputs[0]), 1)
    self.assertIsInstance(instrument.eager_inputs[0], tuple)
    self.assertEqual(instrument.eager_inputs[0][0], x)
    self.assertEqual(len(instrument.eager_inputs[1]), 1)
    self.assertIsInstance(instrument.eager_inputs[1], tuple)
    self.assertAllClose(instrument.eager_inputs[1][0], np.log(6.0))
    self.assertFalse(instrument.graph_op_types)
    self.assertFalse(instrument.graph_op_names)
    self.assertFalse(instrument.graph_attrs)
    self.assertFalse(instrument.graph_graphs)
    self.assertFalse(instrument.graph_inputs)

  def testMultiThreadedEagerOpExecution(self):
    # Instrument for the main thread.
    instrument_0 = _NumpyFunctionCallback()

    # Instrument for the to-be-created thread.
    instrument_1 = _NumpyFunctionCallback()

    def thread_1_job():
      x = constant_op.constant(6.0)
      op_callbacks.add_op_callback(instrument_1.callback)
      y = math_ops.square(math_ops.log(x))
      op_callbacks.remove_op_callback(instrument_1.callback)
      return y

    thread_1 = threading.Thread(target=thread_1_job)
    thread_1.start()

    # While thread_1 is ongoing, do something on the main thread.
    x = constant_op.constant(2.0)
    op_callbacks.add_op_callback(instrument_0.callback)
    y = math_ops.cos(x)
    self.assertAllClose(y, np.cos(2.0))
    op_callbacks.remove_op_callback(instrument_0.callback)

    thread_1.join()

    self.assertEqual(instrument_0.eager_op_types, [_COS_OP])
    self.assertEqual(instrument_0.eager_op_names, [None])
    self.assertEqual(instrument_1.eager_op_types, [_LOG_OP, _SQUARE_OP])
    self.assertEqual(instrument_1.eager_op_names, [None, None])

  def testEagerFunctionExecution(self):
    instrument = _NumpyFunctionCallback()

    @def_function.function
    def square_log(x):
      return math_ops.square(math_ops.log(x))

    # Call the function once, so that the graph construction won't show up
    # in the callback.
    x_float32 = constant_op.constant(6.0, dtype=dtypes.float32)
    x_float64 = constant_op.constant(6.0, dtype=dtypes.float64)
    square_log(x_float32)
    square_log(x_float64)

    op_callbacks.add_op_callback(instrument.callback)
    y = square_log(x_float32)
    self.assertAllClose(y, np.square(np.log(6.0)))
    y = square_log(x_float64)
    self.assertAllClose(y, np.square(np.log(6.0)))

    self.assertEqual(instrument.eager_op_names, [None, None])
    self.assertFalse(instrument.graph_op_types)
    self.assertFalse(instrument.graph_op_names)
    self.assertFalse(instrument.graph_inputs)

    # Each of the two dtypes should be associated with its own FuncGraph.
    self.assertIn(
        square_log.get_concrete_function(x_float32).name,
        instrument.eager_op_types)
    self.assertIn(
        square_log.get_concrete_function(x_float64).name,
        instrument.eager_op_types)

    self.assertEqual(len(instrument.eager_inputs), 2)
    self.assertIsInstance(instrument.eager_inputs[0], tuple)
    self.assertEqual(instrument.eager_inputs[0][0], x_float32)
    self.assertIsInstance(instrument.eager_inputs[1], tuple)
    self.assertEqual(instrument.eager_inputs[1][0], x_float64)

  def testMultiThreadedEagerFunctionExecution(self):
    # Instrument for the main thread.
    instrument_0 = _NumpyFunctionCallback()

    # Instrument for the to-be-created thread.
    instrument_1 = _NumpyFunctionCallback()

    @def_function.function
    def square_log(x):
      return math_ops.square(math_ops.log(x))

    # Call the function once, so that the graph construction won't show up
    # in the callback.
    x_float32 = constant_op.constant(6.0, dtype=dtypes.float32)
    x_float64 = constant_op.constant(6.0, dtype=dtypes.float64)
    square_log(x_float32)
    square_log(x_float64)

    def thread_1_job():
      op_callbacks.add_op_callback(instrument_1.callback)
      square_log(x_float32)

    thread_1 = threading.Thread(target=thread_1_job)
    thread_1.start()

    # In the meantime, run some computation on the main thread.
    op_callbacks.add_op_callback(instrument_0.callback)
    square_log(x_float64)

    thread_1.join()

    # Each of the two dtypes should be associated with its own FuncGraph.
    self.assertIn(
        square_log.get_concrete_function(x_float64).name,
        instrument_0.eager_op_types)
    self.assertEqual(instrument_0.eager_op_names, [None])
    self.assertFalse(instrument_0.graph_op_types)
    self.assertIn(
        square_log.get_concrete_function(x_float32).name,
        instrument_1.eager_op_types)
    self.assertEqual(instrument_1.eager_op_names, [None])
    self.assertFalse(instrument_1.graph_op_types)

  @test_util.run_in_graph_and_eager_modes
  def testSimpleGraphConstructionScopeOutsideFunction(self):
    instrument = _NumpyFunctionCallback()

    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def log_2plus_unique_x(x):
      unique_values, unique_pos = array_ops.unique(x)
      return math_ops.log(2.0 + unique_values), unique_pos

    x = constant_op.constant([-1.0, -1.0, 0.0], dtype=dtypes.float32)
    y1, y2 = log_2plus_unique_x(x)
    self.assertAllClose(y1, [0.0, np.log(2.0)])
    self.assertAllClose(y2, [0, 0, 1])

    self.assertIn(_UNIQUE_OP, instrument.graph_op_types)
    self.assertIn(_ADD_OP, instrument.graph_op_types)
    self.assertIn(_LOG_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.graph_op_names), len(instrument.graph_op_types))

    # Check the graph internal ndarrays recorded at runtime.
    unique_op_outputs = instrument.graph_internal_ndarrays[_UNIQUE_OP]
    if context.executing_eagerly():
      # b/140810696: The run_in_graph_and_eager_modes decorator runs
      # Session.run() twice. We can't assert on the number of outputs in
      # that case.
      self.assertEqual(len(unique_op_outputs), 2)
    self.assertAllClose(unique_op_outputs[0], [-1.0, 0.0])
    self.assertAllClose(unique_op_outputs[1], [0, 0, 1])
    add_op_outputs = instrument.graph_internal_ndarrays[b"add"]
    if context.executing_eagerly():
      self.assertEqual(len(add_op_outputs), 1)
    self.assertAllClose(add_op_outputs[0], [1.0, 2.0])
    log_op_outputs = instrument.graph_internal_ndarrays[_LOG_OP]
    if context.executing_eagerly():
      self.assertEqual(len(log_op_outputs), 1)
    self.assertAllClose(log_op_outputs[0], [0.0, np.log(2.0)])

  @test_util.run_in_graph_and_eager_modes
  def testPadOp(self):
    instrument = _NumpyFunctionCallback()

    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def my_pad(x, padding):
      return array_ops.pad(x, padding)

    x = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
    paddings = [[1, 1], [2, 2]]

    y = my_pad(x, paddings)
    expected_output = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 3, 4, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    self.assertAllClose(y, expected_output)
    self.assertAllClose(
        instrument.graph_internal_ndarrays[b"Pad"][0], expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testSimpleGraphConstructionWithCallbackReturningNone(self):
    """Test that callbacks that return None works."""
    op_types = []
    def no_return_callback(op_type,
                           inputs,
                           attrs,
                           outputs,
                           op_name=None,
                           graph=None):
      del inputs, attrs, outputs, op_name, graph  # Unused.
      op_types.append(compat.as_bytes(op_type))

    op_callbacks.add_op_callback(no_return_callback)

    @def_function.function
    def log1p(x):
      return math_ops.log(1.0 + x)
    x = constant_op.constant(3.0)
    y = log1p(x)

    self.assertAllClose(y, np.log(4.0))
    self.assertIn(_ADD_OP, op_types)
    self.assertIn(_LOG_OP, op_types)

  @test_util.run_in_graph_and_eager_modes
  def testGraphConstructionInputsAndGraphAreCapturedCorrectly(self):
    instrument = _NumpyFunctionCallback(instrument_graph_ops=False)

    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def log_2plus_unique_x(x):
      unique_values, unique_pos = array_ops.unique(x)
      return math_ops.log(2.0 + unique_values), unique_pos

    x = constant_op.constant([-1.0, -1.0, 0.0], dtype=dtypes.float32)
    y1, y2 = log_2plus_unique_x(x)
    self.assertAllClose(y1, [0.0, np.log(2.0)])
    self.assertAllClose(y2, [0, 0, 1])

    # Check the recorded input tensors.
    self.assertEqual(
        len(instrument.graph_inputs), len(instrument.graph_op_types))
    unique_inputs = instrument.graph_inputs[instrument.graph_op_types.index(
        _UNIQUE_OP)]
    self.assertIsInstance(unique_inputs, tuple)
    self.assertEqual(len(unique_inputs), 1)
    self.assertEqual(
        compat.as_bytes(unique_inputs[0].op.op_def.name), _PLACEHOLDER_OP)

    add_inputs = instrument.graph_inputs[instrument.graph_op_types.index(
        _ADD_OP)]
    self.assertIsInstance(add_inputs, tuple)
    self.assertEqual(len(add_inputs), 2)
    self.assertEqual(
        compat.as_bytes(add_inputs[0].op.op_def.name), _CONSTANT_OP)
    self.assertEqual(compat.as_bytes(add_inputs[1].op.op_def.name), _UNIQUE_OP)

    log_inputs = instrument.graph_inputs[instrument.graph_op_types.index(
        _LOG_OP)]
    self.assertIsInstance(log_inputs, tuple)
    self.assertEqual(len(log_inputs), 1)
    self.assertEqual(compat.as_bytes(log_inputs[0].op.op_def.name), _ADD_OP)

    # Check the recorded graphs.
    self.assertEqual(
        len(instrument.graph_graphs), len(instrument.graph_op_types))
    self.assertGreater(len(instrument.graph_graph_versions), 1)
    if context.executing_eagerly():
      for i in range(len(instrument.graph_graph_versions) - 1):
        self.assertGreater(instrument.graph_graph_versions[i + 1],
                           instrument.graph_graph_versions[i])

  @test_util.run_in_graph_and_eager_modes
  def testEagerGraphOpConstructionSimpleGraphScopeInsideFunction(self):
    instrument = _NumpyFunctionCallback()

    @def_function.function
    def log_2plus_unique_x(x):
      op_callbacks.add_op_callback(instrument.callback)
      unique_values, _ = array_ops.unique(x)
      y = math_ops.log(2.0 + unique_values)
      op_callbacks.remove_op_callback(instrument.callback)
      return math_ops.sin(y)

    x = constant_op.constant([-1.0, -1.0, 0.0], dtype=dtypes.float32)
    output = log_2plus_unique_x(x)
    self.assertAllClose(output, np.sin([0.0, np.log(2.0)]))

    # The following ops should have been captured by the callback
    # because they were constructed within the scope of `op_callback()`.
    self.assertIn(_UNIQUE_OP, instrument.graph_op_types)
    self.assertIn(_ADD_OP, instrument.graph_op_types)
    self.assertIn(_LOG_OP, instrument.graph_op_types)
    # The "Sin" op should not have been captured, because it was constructed
    # outside the scope of `op_callback()`.
    self.assertNotIn(_SIN_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.graph_op_names), len(instrument.graph_op_types))

    # Check the graph internal ndarrays recorded at runtime.
    unique_op_outputs = instrument.graph_internal_ndarrays[_UNIQUE_OP]
    self.assertEqual(len(unique_op_outputs), 2)
    self.assertAllClose(unique_op_outputs[0], [-1.0, 0.0])
    self.assertAllClose(unique_op_outputs[1], [0, 0, 1])
    add_op_outputs = instrument.graph_internal_ndarrays[b"add"]
    self.assertEqual(len(add_op_outputs), 1)
    self.assertAllClose(add_op_outputs[0], [1.0, 2.0])
    log_op_outputs = instrument.graph_internal_ndarrays[_LOG_OP]
    self.assertEqual(len(log_op_outputs), 1)
    self.assertAllClose(log_op_outputs[0], [0.0, np.log(2.0)])

  def testEagerOpAttributesAreCapture(self):
    instrument = _NumpyFunctionCallback()
    m = constant_op.constant([[1.0, -1.0], [0.0, 1.0]])
    x = constant_op.constant([[-2.0], [3.0]])
    op_callbacks.add_op_callback(instrument.callback)
    y = math_ops.matmul(m, x, transpose_a=True, transpose_b=False)
    self.assertAllClose(y, [[-2.0], [5.0]])

    self.assertEqual(len(instrument.eager_attrs), 1)
    self.assertIsInstance(instrument.eager_attrs[0], tuple)
    self.assertEqual(
        instrument.eager_attrs[0][instrument.eager_attrs[0].index("transpose_a")
                                  + 1], True)
    self.assertEqual(
        instrument.eager_attrs[0][instrument.eager_attrs[0].index("transpose_b")
                                  + 1], False)
    self.assertEqual(len(instrument.graph_attrs), 0)

  @test_util.run_in_graph_and_eager_modes
  def testGraphOpAttributesAreCapture(self):
    instrument = _NumpyFunctionCallback()

    @def_function.function
    def my_matmul(m, x):
      return math_ops.matmul(m, x, transpose_a=True, transpose_b=False)

    m = constant_op.constant([[1.0, -1.0], [0.0, 1.0]])
    x = constant_op.constant([[-2.0], [3.0]])
    op_callbacks.add_op_callback(instrument.callback)
    y = my_matmul(m, x)
    self.assertAllClose(y, [[-2.0], [5.0]])

    index = instrument.graph_op_types.index(_MATMUL_OP)
    self.assertIsInstance(instrument.graph_attrs[index], tuple)
    self.assertEqual(
        instrument.graph_attrs[index][
            instrument.graph_attrs[index].index("transpose_a") + 1].b, True)
    self.assertEqual(
        instrument.graph_attrs[index][
            instrument.graph_attrs[index].index("transpose_b") + 1].b, False)
    if context.executing_eagerly():
      self.assertEqual(len(instrument.eager_attrs), 1)
      self.assertIsInstance(instrument.eager_attrs[0], tuple)

  @test_util.run_in_graph_and_eager_modes
  def testEagerGraphOpConstructionIfControlFlow(self):
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def my_function_with_cond(x):
      if math_ops.greater(x, 0.0):
        return x**2.0
      else:
        return x**3.0

    x = constant_op.constant(-4.0)
    self.assertAllClose(my_function_with_cond(x), -64.0)

    self.assertIn(_IF_OP, instrument.graph_op_types)
    self.assertIn(_GREATER_OP, instrument.graph_op_types)
    self.assertIn(_POW_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.graph_op_names), len(instrument.graph_op_types))

    # Check the graph internal ndarrays recorded at runtime.
    greater_op_outputs = instrument.graph_internal_ndarrays[_GREATER_OP]
    self.assertEqual(len(greater_op_outputs), 1)
    self.assertAllClose(greater_op_outputs[0], False)
    # This was needed for backwards compatibility with TF2 Estimators which
    # rely on variable names.
    prefix = b"cond/" if context.executing_eagerly() else b""
    pow_op_outputs = instrument.graph_internal_ndarrays[b"%spow" % prefix]
    self.assertEqual(len(pow_op_outputs), 1)
    self.assertAllClose(pow_op_outputs[0], -64.0)

  def testEagerGraphOpConstructionWhileLoopControlFlow(self):
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def my_function_with_while(counter, lim, accum):
      while math_ops.less(counter, lim):
        accum.assign_add(accum)
        counter.assign_add(1.0)

    counter = variables.Variable(0.0)
    lim = constant_op.constant(4.0, dtype=dtypes.float32)
    accum = variables.Variable(1.0)
    my_function_with_while(counter, lim, accum)

    self.assertAllClose(accum.read_value(), 16.0)
    self.assertIn(_WHILE_OP, instrument.graph_op_types)
    self.assertIn(_LESS_OP, instrument.graph_op_types)
    self.assertIn(_ASSIGN_ADD_VARIABLE_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.graph_op_names), len(instrument.graph_op_types))

    # Check the graph internal ndarrays recorded at runtime.
    read_variable_op_outputs = instrument.graph_internal_ndarrays[
        b"while/" + _READ_VARIABLE_OP]
    self.assertAllClose(read_variable_op_outputs, [1.0, 2.0, 4.0, 8.0])
    less_op_outputs = instrument.graph_internal_ndarrays[b"while/" + _LESS_OP]
    self.assertAllClose(less_op_outputs, [True, True, True, True, False])

  # TODO(cais): The following isn't decorated with
  # `@test_util.run_in_graph_and_eager_modes` because of some apparent
  # between `Dataset.map()` and `numpy_function()` used by
  # `_NumpyFunctionCallback`. Maybe investigate.
  def testDatasetMapTest(self):
    instrument = _NumpyFunctionCallback()

    op_callbacks.add_op_callback(instrument.callback)

    tensor = constant_op.constant(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    def map_fn(x):
      return math_ops.log(math_ops.square(x) + 1)

    dataset = dataset_ops.Dataset.from_tensor_slices(tensor).batch(2).map(
        map_fn)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    self.assertAllClose(iterator.next(), np.log([1.25, 2]))
    self.assertAllClose(iterator.next(), np.log([3.25, 5]))

    self.assertIn(_SQUARE_OP, instrument.graph_op_types)
    self.assertIn(_ADD_OP, instrument.graph_op_types)
    self.assertIn(_LOG_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.eager_op_types), len(instrument.eager_op_names))

  def testSparseTensorEagerExecution(self):
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    indices = [[1, 2], [2, 0], [3, 4]]
    values = [0.0, 8.0, -2.0]
    shape = [4, 5]
    sp = sparse_tensor.SparseTensorValue(indices, values, shape)
    w = ops.convert_to_tensor(np.ones([5, 1], np.float32))

    y = sparse_ops.sparse_tensor_dense_matmul(sp, w)
    self.assertAllClose(y, [[0.0], [0.0], [8.0], [-2.0]])
    self.assertIn(_SPARSE_TENSOR_DENSE_MATMUL_OP, instrument.eager_op_types)
    self.assertFalse(instrument.graph_op_types)

  @test_util.run_in_graph_and_eager_modes
  def testSparseTensorFuncGraph(self):
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def dense_matmul(sp, w):
      return sparse_ops.sparse_tensor_dense_matmul(sp, w)

    indices = [[1, 2], [2, 0], [3, 4]]
    values = [0.0, 8.0, -2.0]
    shape = [4, 5]
    sp = sparse_tensor.SparseTensorValue(indices, values, shape)
    w = ops.convert_to_tensor(np.ones([5, 1], np.float32))
    y = dense_matmul(sp, w)
    self.assertAllClose(y, [[0.0], [0.0], [8.0], [-2.0]])
    self.assertIn(_SPARSE_TENSOR_DENSE_MATMUL_OP, instrument.graph_op_types)
    if context.executing_eagerly():
      self.assertIn(
          dense_matmul.get_concrete_function(sp, w).name,
          instrument.eager_op_types)

    # Check the graph internal ndarrays recorded at runtime.
    sparse_matmul_outputs = instrument.graph_internal_ndarrays[
        _SPARSE_TENSOR_DENSE_MATMUL_OP + b"/" + _SPARSE_TENSOR_DENSE_MATMUL_OP]
    if context.executing_eagerly():
      self.assertEqual(len(sparse_matmul_outputs), 1)
    self.assertAllClose(sparse_matmul_outputs[0], [[0.0], [0.0], [8.0], [-2.0]])

  @test_util.run_in_graph_and_eager_modes
  def testOverrideDTypeInFuncGraph(self):
    def to_float64(op_type, inputs, attrs, outputs, op_name=None, graph=None):
      del inputs, attrs, op_name, graph  # Unused.
      if op_type in ("Const", "Placeholder"):
        return outputs
      else:
        return [math_ops.cast(output, dtypes.float64) for output in outputs]

    op_callbacks.add_op_callback(to_float64)

    @def_function.function
    def add_1_times_2(x):
      return (x + 1.0) * 2.0

    x = constant_op.constant(3.0, dtype=dtypes.float32)
    y = add_1_times_2(x)
    self.assertEqual(y.dtype, dtypes.float64)
    self.assertAllClose(y, 8.0)

  def testNoOutputOpUnderEagerExecution(self):
    instrument = _NumpyFunctionCallback()

    x = constant_op.constant(10.0)
    y = constant_op.constant(20.0)
    op_callbacks.add_op_callback(instrument.callback)
    z = x + y
    w = control_flow_ops.group([z])
    self.assertIsNone(w)
    self.assertEqual(instrument.eager_op_types, [_ADD_OP])

  def testOpCallbackCapturesConstTensors(self):
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def times_two_plus_three(x):
      return x * 2.0 + 3.0

    self.assertAllClose(times_two_plus_three(constant_op.constant(10.0)), 23.0)
    self.assertEqual(instrument.graph_op_types.count(b"Const"), 2)

  @test_util.run_in_graph_and_eager_modes
  def testOpCallbackWorksWithGradientTape(self):
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    v = variables.Variable(3.0, dtype=dtypes.float32)
    if not context.executing_eagerly():
      self.evaluate(v.initializer)

    @def_function.function
    def get_gradients():
      with backprop.GradientTape() as tape:
        loss = math_ops.sin(math_ops.square(v))
        gradients = tape.gradient(loss, v)
      return gradients

    gradients = get_gradients()
    # Applying the chain rule.
    self.assertAllClose(gradients, np.cos(3.0 * 3.0) * 3.0 * 2.0)
    self.assertIn(_SQUARE_OP, instrument.graph_op_types)
    self.assertIn(_SIN_OP, instrument.graph_op_types)
    # The mul and cos ops are created for backprop.
    self.assertIn(_MUL_OP, instrument.graph_op_types)
    self.assertIn(_COS_OP, instrument.graph_op_types)

    # Check the ndarrays from runtime.
    cos_op_outputs = instrument.graph_internal_ndarrays[b"gradient_tape/" +
                                                        _COS_OP]
    self.assertEqual(len(cos_op_outputs), 1)
    self.assertAllClose(cos_op_outputs[0], np.cos(3.0 * 3.0))


class OpCallbacksErrorConditionsTest(test_util.TensorFlowTestCase):

  def tearDown(self):
    op_callbacks.clear_op_callbacks()
    super(OpCallbacksErrorConditionsTest, self).tearDown()

  def testNonCallableObjectArgErrors(self):
    with self.assertRaisesRegex(ValueError, r"is expected to be callable"):
      op_callbacks.add_op_callback(1337)

  def testRemoveUnregisteredCallbackLeadsToError(self):
    instrument = _NumpyFunctionCallback()
    with self.assertRaisesRegex(KeyError, r"has not been registered"):
      op_callbacks.remove_op_callback(instrument.callback)

  def testRemovingCallbackTwiceLeadsToError(self):
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)
    op_callbacks.remove_op_callback(instrument.callback)
    with self.assertRaisesRegex(KeyError, r"has not been registered"):
      op_callbacks.remove_op_callback(instrument.callback)

  def testOverridingWithWrongNumberOfTensorOutputsErrors(self):
    def wrong_outputs_callback(op_type,
                               inputs,
                               attrs,
                               outputs,
                               op_name=None,
                               graph=None):
      del op_type, inputs, attrs, op_name, graph  # Unused.
      return outputs[0], math_ops.negative(outputs[0])

    @def_function.function
    def log1p(x):
      return math_ops.log(1.0 + x)

    x = constant_op.constant(3.0)
    op_callbacks.add_op_callback(wrong_outputs_callback)
    with self.assertRaisesRegex(
        ValueError,
        r"returned 2 tensors, .* does not match .* \(1\)"):
      log1p(x)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
