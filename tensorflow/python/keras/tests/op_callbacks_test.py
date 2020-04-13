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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import script_ops
from tensorflow.python.util import compat


# Keep all the hard-coded op type strings in one place so they are easy to
# change all at once in the face of any possible future op type name changes.
_ENTER_OP = b"Enter"
_EXIT_OP = b"Exit"
_IDENTITY_OP = b"Identity"
_IF_OP = b"If"
_MERGE_OP = b"Merge"
_NEXT_ITERATION_OP = b"NextIteration"
_PLACEHOLDER_OP = b"Placeholder"
_STATELESS_IF_OP = b"StatelessIf"
_SWITCH_OP = b"Switch"
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
                                        _VAR_HANDLE_OP, _PLACEHOLDER_OP):
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

  @test_util.run_in_graph_and_eager_modes
  def testKerasLSTMPredict(self):
    instrument = _NumpyFunctionCallback(float_only=True)

    op_callbacks.add_op_callback(instrument.callback)

    model = keras.Sequential()
    model.add(keras.layers.LSTM(1, input_shape=(2, 4)))
    model.compile(loss="mse", optimizer="sgd")

    xs = np.zeros([8, 2, 4], dtype=np.float32)
    ys = model.predict(xs)

    self.assertAllClose(ys, np.zeros([8, 1]))
    # We avoid asserting on the internal details of the LSTM implementation.
    # Instead, we just assert that some graph-internal execution states are
    # recorded by the callback.
    self.assertTrue(instrument.graph_internal_ndarrays)

  @test_util.run_in_graph_and_eager_modes
  def testKeraModelFit(self):
    # TODO(cais): The purely PyFunc (numpy_function) based instrumentation
    # doesn't work for the entire Keras model and its fit() call, due to some
    # shape inference limitations. Use tfdbg's gen_debug_ops for testing
    # instead (b/139668469).
    instrument = _NumpyFunctionCallback(instrument_graph_ops=False)
    op_callbacks.add_op_callback(instrument.callback)

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, input_shape=(8,), activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam")

    batch_size = 4
    xs = np.ones([batch_size, 8])
    ys = np.zeros([batch_size, 1])
    history = model.fit(xs, ys, epochs=2, verbose=0)

    # Simply assert that the training proceeded as expected and that
    # op callbacks are invoked. We prefer not to assert on the details of the
    # graph construction and the execution, in order to avoid future
    # maintenance cost.
    self.assertEqual(len(history.history["loss"]), 2)
    self.assertTrue(instrument.graph_op_types)
    self.assertEqual(len(instrument.graph_op_types),
                     len(instrument.graph_op_names))
    if context.executing_eagerly():
      self.assertTrue(instrument.eager_op_types)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
