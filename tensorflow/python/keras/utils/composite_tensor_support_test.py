# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras composite tensor support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import test


# Define test-only Layer classes to validate passing Sparse and Ragged tensors
# between layers.
class ToDense(Layer):
  """Create a dense (standard) tensor from the given input tensor."""

  def __init__(self, default_value, **kwargs):
    super(ToDense, self).__init__(**kwargs)
    self._default_value = default_value

  def call(self, inputs):
    if isinstance(inputs, ragged_tensor.RaggedTensor):
      return inputs.to_tensor(default_value=self._default_value)
    elif isinstance(inputs, sparse_tensor.SparseTensor):
      return sparse_ops.sparse_tensor_to_dense(
          inputs, default_value=self._default_value)
    elif isinstance(inputs, ops.Tensor):
      return inputs
    else:
      raise TypeError("Unexpected tensor type %s" % type(inputs).__name__)


class ToRagged(Layer):
  """Create a ragged tensor based on a given dense tensor."""

  def __init__(self, padding, ragged_rank=1, **kwargs):
    super(ToRagged, self).__init__(**kwargs)
    self._padding = padding
    self._ragged_rank = ragged_rank

  def call(self, inputs):
    return ragged_tensor.RaggedTensor.from_tensor(
        inputs, padding=self._padding, ragged_rank=self._ragged_rank)


class ToSparse(Layer):
  """Create a sparse tensor based on a given dense tensor."""

  def call(self, inputs):
    indices = array_ops.where(math_ops.not_equal(inputs, 0))
    values = array_ops.gather_nd(inputs, indices)
    shape = array_ops.shape(inputs, out_type=dtypes.int64)
    return sparse_tensor.SparseTensor(indices, values, dense_shape=shape)


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class InternalCompositeTest(keras_parameterized.TestCase,
                            ragged_test_util.RaggedTensorTestCase):

  def test_model_with_internal_ragged_tensors(self):
    # Create a model that accepts an input, converts it to Ragged, and
    # converts the ragged tensor back to a dense tensor.
    layers = [ToRagged(padding=0), ToDense(default_value=-1)]
    model = testing_utils.get_model_from_layers(layers, input_shape=(None,))

    # Define some input data with additional padding.
    input_data = np.array([[1, 0, 0], [2, 3, 0]])
    expected_output = np.array([[1, -1], [2, 3]])
    output = model.predict(input_data)
    self.assertAllEqual(expected_output, output)

  def test_model_with_internal_sparse_tensors(self):
    # Create a model that accepts an input, converts it to Sparse, and
    # converts the sparse tensor back to a dense tensor.
    layers = [ToSparse(), ToDense(default_value=-1)]
    model = testing_utils.get_model_from_layers(layers, input_shape=(None,))

    # Define some input data with additional padding.
    input_data = np.array([[1, 0, 0], [2, 3, 0]])
    expected_output = np.array([[1, -1, -1], [2, 3, -1]])
    output = model.predict(input_data)
    self.assertAllEqual(expected_output, output)

  def test_training_model_with_internal_ragged_tensors(self):

    # Create a model that implements y=Mx. This is easy to learn and will
    # demonstrate appropriate gradient passing. (We have to use RaggedTensors
    # for this test, as ToSparse() doesn't support gradient propagation through
    # the layer.) TODO(b/124796939): Investigate this.
    layers = [core.Dense(2), ToRagged(padding=0), ToDense(default_value=-1)]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))

    input_data = np.random.rand(1024, 1)
    expected_data = np.concatenate((input_data * 3, input_data * .5), axis=-1)

    model.compile(
        loss="mse",
        optimizer="adam",
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(input_data, expected_data, epochs=10, verbose=0)

    # If the model trained, the loss stored at history[0] should be different
    # than the one stored at history[-1].
    self.assertNotEqual(history.history["loss"][-1], history.history["loss"][0])

  def test_model_with_ragged_tensor_outputs(self):
    # Create a model that accepts an input, converts it to Ragged, and
    # converts the ragged tensor back to a dense tensor.
    layers = [ToRagged(padding=0)]
    model = testing_utils.get_model_from_layers(layers, input_shape=(None,))

    # Define some input data with additional padding.
    input_data = np.array([[1, 0, 0], [2, 3, 0]])
    output = model.predict(input_data)

    expected_values = [[1], [2, 3]]
    self.assertRaggedEqual(expected_values, output)

  def test_model_with_ragged_tensor_rebatched_outputs(self):
    # Create a model that accepts an input, converts it to Ragged, and
    # converts the ragged tensor back to a dense tensor.
    layers = [ToRagged(padding=0)]
    model = testing_utils.get_model_from_layers(layers, input_shape=(None,))

    # Define some input data with additional padding.
    input_data = np.array([[1, 0, 0], [2, 3, 0], [4, 0, 0], [5, 6, 0]])
    output = model.predict(input_data, batch_size=2)

    expected_values = [[1], [2, 3], [4], [5, 6]]
    self.assertRaggedEqual(expected_values, output)

  def test_model_with_sparse_tensor_outputs(self):
    # Create a model that accepts an input, converts it to Ragged, and
    # converts the ragged tensor back to a dense tensor.
    layers = [ToSparse()]
    model = testing_utils.get_model_from_layers(layers, input_shape=(None,))

    # Define some input data with additional padding.
    input_data = np.array([[1, 0, 0], [2, 3, 0]])
    output = model.predict(input_data)

    expected_indices = np.array([[0, 0], [1, 0], [1, 1]])
    expected_values = np.array([1, 2, 3])
    expected_dense_shape = np.array([2, 3])

    self.assertAllEqual(output.indices, expected_indices)
    self.assertAllEqual(output.values, expected_values)
    self.assertAllEqual(output.dense_shape, expected_dense_shape)

  def test_model_with_sparse_tensor_rebatched_outputs(self):
    # Create a model that accepts an input, converts it to Ragged, and
    # converts the ragged tensor back to a dense tensor.
    layers = [ToSparse()]
    model = testing_utils.get_model_from_layers(layers, input_shape=(None,))

    # Define some input data with additional padding.
    input_data = np.array([[1, 0, 0], [2, 3, 0], [4, 0, 0], [5, 6, 0]])
    output = model.predict(input_data, batch_size=2)

    expected_indices = np.array([[0, 0], [1, 0], [1, 1], [2, 0], [3, 0], [3,
                                                                          1]])
    expected_values = np.array([1, 2, 3, 4, 5, 6])
    expected_dense_shape = np.array([4, 3])

    self.assertAllEqual(output.indices, expected_indices)
    self.assertAllEqual(output.values, expected_values)
    self.assertAllEqual(output.dense_shape, expected_dense_shape)


if __name__ == "__main__":
  test.main()
