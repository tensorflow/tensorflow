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
"""Correctness tests for tf.keras DNN model using DistributionStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.distribute import keras_correctness_test_base
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.ops import math_ops


class TestTwoInputLayersCorrectness(
    keras_correctness_test_base.TestDistributionStrategyCorrectnessBase):

  _batch_size = 8000
  _num_users = 7000
  _num_items = 7000

  def get_model(self, initial_weights=None, distribution=None):
    with keras_correctness_test_base.MaybeDistributionScope(distribution):

      real_batch_size = (1 if distribution else
                         self._distribution_to_test.num_replicas_in_sync)

      user_input = keras.layers.Input(
          shape=(self._batch_size,),
          batch_size=real_batch_size,
          name="users",
          dtype=dtypes.int32)

      item_input = keras.layers.Input(
          shape=(self._batch_size,),
          batch_size=real_batch_size,
          name="items",
          dtype=dtypes.int32)

      concat = keras.layers.concatenate([user_input, item_input], axis=-1)
      logits = keras.layers.Dense(
          1, name="rating")(
              math_ops.cast(concat, dtypes.float32))

      keras_model = keras.Model(inputs=[user_input, item_input], outputs=logits)

      if initial_weights:
        keras_model.set_weights(initial_weights)

      keras_model.compile(loss="mse", optimizer=gradient_descent_keras.SGD(0.5))
      return keras_model

  def get_data(self):
    users, items, labels = self._get_raw_data()
    x_train = {"users": users, "items": items}
    y_train = labels
    data = x_train, y_train
    dataset = dataset_ops.Dataset.from_tensors(data).repeat()
    dataset = dataset.batch(self._distribution_to_test.num_replicas_in_sync)
    return dataset, None, None

  def _get_raw_data(self):
    np.random.seed(1337)

    users = np.random.randint(0, self._num_users, size=(self._batch_size,))
    items = np.random.randint(0, self._num_users, size=(self._batch_size,))
    labels = np.random.randint(0, 10000, size=(self._batch_size,))

    users = users.astype("int32")
    items = items.astype("int32")
    labels = labels.astype("int32")

    return users, items, labels

  def get_input_for_correctness_test(self, **kwargs):
    update_freq = None
    if (isinstance(self._distribution_to_test, tpu_strategy.TPUStrategy) and
        self._distribution_to_test.extended.steps_per_run > 1):
      # For TPUStrategy with steps_per_run > 1, the callback is not invoked
      # every step. So, to compare the CPU/TPU, we let the CPU to behave the
      # same as TPU.
      update_freq = self._distribution_to_test.extended.steps_per_run

    dataset, _, _ = self.get_data()
    learning_rate_scheduler = (
        keras_correctness_test_base.LearningRateBatchScheduler(update_freq))
    training_inputs = {
        "x": dataset,
        "epochs": 1,
        "steps_per_epoch": 1,
        "verbose": 2,
        "callbacks": [learning_rate_scheduler]
    }

    return training_inputs, None, None

  def get_input_for_dynamic_lr_test(self, **kwargs):
    dataset, _, _ = self.get_data()
    training_inputs = {
        "x": dataset,
        "epochs": 1,
        "steps_per_epoch": 1,
        "verbose": 2
    }

    return training_inputs, None, None

  def run_correctness_test(self,
                           distribution,
                           use_numpy,
                           use_validation_data,
                           with_batch_norm=False,
                           is_stateful_model=False):
    self._distribution_to_test = distribution
    super(TestTwoInputLayersCorrectness, self).run_correctness_test(
        distribution,
        use_numpy,
        use_validation_data,
        with_batch_norm=False,
        is_stateful_model=False)

  @combinations.generate(
      keras_correctness_test_base
      .all_strategies_excluding_tpu_and_input_config_combinations())
  def test_dnn_correctness(self, distribution, use_numpy, use_validation_data):
    self.run_correctness_test(distribution, use_numpy, use_validation_data)

  def run_dynamic_lr_test(self, distribution):
    self._distribution_to_test = distribution
    super(TestTwoInputLayersCorrectness, self).run_dynamic_lr_test(distribution)

  @combinations.generate(
      keras_correctness_test_base
      .all_strategies_excluding_tpu_and_input_config_combinations())
  def test_dnn_with_dynamic_learning_rate(self, distribution, use_numpy,
                                          use_validation_data):
    self.run_dynamic_lr_test(distribution)


if __name__ == "__main__":
  test.main()
