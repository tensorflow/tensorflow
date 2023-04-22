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
"""Tests for convert_to_constants.py."""

import os

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import testing_utils
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.util import nest


class VariablesToConstantsTest(test.TestCase):

  def _freezeModel(self, model):
    """Freezes the model.

    Args:
      model: Function.

    Returns:
      root: AutoTrackable object with original ConcreteFunction.
      output_func: frozen ConcreteFunction.
    """
    root = module.Module()
    root.f = model
    input_func = root.f.get_concrete_function()

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func, lower_control_flow=False)
    return root, output_func

  def _hasStatefulPartitionedCallOp(self, graph_def):
    """Determines if a StatefulPartitionedCall op exists in the graph."""
    for node in graph_def.node:
      if node.op == "StatefulPartitionedCall":
        return True
    return False

  def _getNumVariables(self, graph_def):
    """Returns the number of ReadVariableOp in the graph."""
    return sum(node.op == "ReadVariableOp" for node in graph_def.node)

  def _testConvertedFunction(self, obj, func, converted_concrete_func,
                             input_data):
    # Ensure the converted graph has no variables and no function calls.
    constant_graph_def = converted_concrete_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    # Check that the converted ConcreteFunction produces the same result as the
    # original Function.
    expected_value = nest.flatten(func(**input_data))
    actual_value = nest.flatten(converted_concrete_func(**input_data))

    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual.numpy())

    # Ensure the shape is retained.
    for tensor in converted_concrete_func.inputs:
      actual_shape = input_data[tensor.name.split(":")[0]].shape
      self.assertEqual(tensor.shape, actual_shape)

    # Save the converted ConcreteFunction as a signature.
    save_dir = os.path.join(self.get_temp_dir(), "frozen_saved_model")
    root = module.Module()
    root.f = converted_concrete_func
    save(root, save_dir, {"mykey": converted_concrete_func})

    # Load it back and make sure it works.
    loaded_obj = load(save_dir)
    actual_value = nest.flatten(loaded_obj.signatures["mykey"](**input_data))
    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual.numpy())

  @testing_utils.run_v2_only
  def testKerasModel(self):
    """Test a basic Keras model with Variables."""
    input_data = {"x": constant_op.constant(1., shape=[1, 1])}

    # Create a simple Keras model.
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]

    model = keras.models.Sequential(
        [keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(x, y, epochs=1)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[1, 1], dtype=dtypes.float32)
    ])
    def to_save(x):
      return model(x)

    root, output_func = self._freezeModel(to_save)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @testing_utils.run_v2_only
  def testKerasLSTM(self):
    """Test a Keras LSTM containing dynamic_rnn ops."""
    input_data = {
        "x":
            constant_op.constant(
                np.array(
                    np.random.random_sample((10, 10, 10)), dtype=np.float32))
    }

    model = keras.models.Sequential(
        [keras.layers.LSTM(units=10, input_shape=(10, 10))])

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[10, 10, 10], dtype=dtypes.float32)
    ])
    def to_save(x):
      return model(x)

    root, output_func = self._freezeModel(to_save)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @testing_utils.run_v2_only
  def testEmbeddings(self):
    """Test model with embeddings."""
    input_data = {
        "x":
            constant_op.constant(
                np.array(np.random.random_sample((20)), dtype=np.int32))
    }

    class EmbeddingModel(keras.Model):

      def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.shared_weights = self.add_weight(
            "weights",
            shape=(2000, 300),
            dtype=dtypes.float32,
            initializer=init_ops.random_normal_initializer(
                mean=0.0, stddev=300**(-0.5)))

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=(20), dtype=dtypes.int32)
      ])
      def func(self, x):
        return array_ops.gather(self.shared_weights, x)

    model = EmbeddingModel()
    root, output_func = self._freezeModel(model.func)
    self._testConvertedFunction(root, root.f, output_func, input_data)


if __name__ == "__main__":
  test.main()
