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
"""Tests for model.fit calls with a Dataset object passed as validation_data."""

import io
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import core
from tensorflow.python.platform import test


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class ValidationDatasetNoLimitTest(keras_parameterized.TestCase):

  def create_dataset(self, num_samples, batch_size):
    input_data = np.random.rand(num_samples, 1)
    expected_data = input_data * 3
    dataset = dataset_ops.Dataset.from_tensor_slices((input_data,
                                                      expected_data))
    return dataset.shuffle(10 * batch_size).batch(batch_size)

  def test_validation_dataset_with_no_step_arg(self):
    # Create a model that learns y=Mx.
    layers = [core.Dense(1)]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))
    model.compile(loss="mse", optimizer="adam", metrics=["mean_absolute_error"])

    train_dataset = self.create_dataset(num_samples=200, batch_size=10)
    eval_dataset = self.create_dataset(num_samples=50, batch_size=25)

    history = model.fit(x=train_dataset, validation_data=eval_dataset, epochs=2)
    evaluation = model.evaluate(x=eval_dataset)

    # If the fit call used the entire dataset, then the final val MAE error
    # from the fit history should be equal to the final element in the output
    # of evaluating the model on the same eval dataset.
    self.assertAlmostEqual(history.history["val_mean_absolute_error"][-1],
                           evaluation[-1], places=5)


class PrintTrainingInfoTest(keras_parameterized.TestCase,
                            parameterized.TestCase):

  @test_util.run_v1_only("Only relevant in graph mode.")
  def test_print_info_with_datasets(self):
    """Print training info should work with val datasets (b/133391839)."""

    model = keras.models.Sequential([keras.layers.Dense(1, input_shape=(1,))])
    model.compile(loss="mse", optimizer="sgd")

    dataset = dataset_ops.Dataset.from_tensors(
        ([1.], [1.])).repeat(100).batch(10)

    val_dataset = dataset_ops.Dataset.from_tensors(
        ([1.], [1.])).repeat(50).batch(10)

    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, "stdout", mock_stdout):
      model.fit(dataset, epochs=2, validation_data=val_dataset)

    self.assertIn(
        "Train on 10 steps, validate on 5 steps", mock_stdout.getvalue())

  @parameterized.named_parameters(
      ("with_validation", True), ("without_validation", False))
  @test_util.run_v1_only("Only relevant in graph mode.")
  def test_print_info_with_numpy(self, do_validation):
    """Print training info should work with val datasets (b/133391839)."""

    model = keras.models.Sequential([keras.layers.Dense(1, input_shape=(2,))])
    model.compile(loss="mse", optimizer="sgd")

    dataset = np.arange(200).reshape(100, 2)

    if do_validation:
      val_data = (np.arange(100).reshape(50, 2), np.arange(50).reshape(50, 1))
    else:
      val_data = None

    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, "stdout", mock_stdout):
      model.fit(dataset, batch_size=10, epochs=2, validation_data=val_data)

    self.assertIn("Train on 100 samples", mock_stdout.getvalue())

    if do_validation:
      self.assertIn(", validate on 50 samples", mock_stdout.getvalue())

  @keras_parameterized.run_all_keras_modes
  def test_dict_float64_input(self):

    class MyModel(keras.Model):

      def __init__(self):
        super(MyModel, self).__init__(self)
        self.dense1 = keras.layers.Dense(10, activation="relu")
        self.dense2 = keras.layers.Dense(10, activation="relu")
        self.concat = keras.layers.Concatenate()
        self.dense3 = keras.layers.Dense(1, activation="sigmoid")

      def call(self, inputs):
        d1 = self.dense1(inputs["one"])
        d2 = self.dense2(inputs["two"])
        concat = self.concat([d1, d2])
        return self.dense3(concat)

    model = MyModel()
    model.compile(
        loss="mae",
        optimizer="adam",
        run_eagerly=testing_utils.should_run_eagerly())

    model.fit(
        x={
            "one": np.random.rand(100, 10, 1),
            "two": np.random.rand(100, 10, 1)
        },
        y=np.random.rand(100, 10, 1))

  def test_dict_validation_input(self):
    """Test case for GitHub issue 30122."""
    train_input_0 = np.random.rand(1000, 1)
    train_input_1 = np.random.rand(1000, 1)
    train_labels = np.random.rand(1000, 1)
    val_input_0 = np.random.rand(1000, 1)
    val_input_1 = np.random.rand(1000, 1)
    val_labels = np.random.rand(1000, 1)

    input_0 = keras.Input(shape=(None,), name="input_0")
    input_1 = keras.Input(shape=(None,), name="input_1")

    class my_model(keras.Model):

      def __init__(self):
        super(my_model, self).__init__(self)
        self.hidden_layer_0 = keras.layers.Dense(100, activation="relu")
        self.hidden_layer_1 = keras.layers.Dense(100, activation="relu")
        self.concat = keras.layers.Concatenate()
        self.out_layer = keras.layers.Dense(1, activation="sigmoid")

      def call(self, inputs=[input_0, input_1]):
        activation_0 = self.hidden_layer_0(inputs["input_0"])
        activation_1 = self.hidden_layer_1(inputs["input_1"])
        concat = self.concat([activation_0, activation_1])
        return self.out_layer(concat)

    model = my_model()
    model.compile(loss="mae", optimizer="adam")

    model.fit(
        x={
            "input_0": train_input_0,
            "input_1": train_input_1
        },
        y=train_labels,
        validation_data=({
            "input_0": val_input_0,
            "input_1": val_input_1
        }, val_labels))


if __name__ == "__main__":
  test.main()
