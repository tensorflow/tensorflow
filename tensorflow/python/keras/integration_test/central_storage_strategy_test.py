# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for KPL + CentralStorageStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.utils import kpl_test_utils


# TODO(b/182278926): Combine this test with other strategies.
@ds_combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
        ],
        mode=["eager"]))
class CentralStorageStrategyTest(tf.test.TestCase, parameterized.TestCase):

  def testTrainAndServeWithKPL(self, distribution):
    use_adapt = False
    test_utils_obj = kpl_test_utils.DistributeKplTestUtils()
    with distribution.scope():
      feature_mapper, label_mapper = test_utils_obj.define_kpls_for_training(
          use_adapt)
      model = test_utils_obj.define_model()
      optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
      accuracy = tf.keras.metrics.Accuracy()

      def dataset_fn(_):
        return test_utils_obj.dataset_fn(feature_mapper, label_mapper)

      @tf.function
      def train_step(iterator):
        """The step function for one training step."""

        def step_fn(inputs):
          """The computation to run on each replica."""
          features, labels = inputs
          with tf.GradientTape() as tape:
            pred = model(features, training=True)
            loss = tf.keras.losses.binary_crossentropy(labels, pred)
            loss = tf.nn.compute_average_loss(loss)
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

          actual_pred = tf.cast(tf.math.greater(pred, 0.5), tf.dtypes.int64)
          accuracy.update_state(labels, actual_pred)

        distribution.run(step_fn, args=(next(iterator),))

      distributed_dataset = distribution.distribute_datasets_from_function(
          dataset_fn)
      distributed_iterator = iter(distributed_dataset)
      num_epochs = 4
      num_steps = 7
      for _ in range(num_epochs):
        accuracy.reset_states()
        for _ in range(num_steps):
          train_step(distributed_iterator)

      self.assertGreater(accuracy.result().numpy(), 0.5)
      self.assertEqual(optimizer.iterations.numpy(), num_epochs * num_steps)

    # Test save/load/serving the trained model.
    test_utils_obj.test_save_load_serving_model(
        model, feature_mapper, test_utils_obj.define_reverse_lookup_layer())


if __name__ == "__main__":
  tf.test.main()
