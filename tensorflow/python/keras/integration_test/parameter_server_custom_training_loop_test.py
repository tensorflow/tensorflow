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
"""Test to demonstrate custom training loop with ParameterServerStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import multiprocessing
from absl import logging
import portpicker
import tensorflow as tf

NUM_EPOCHS = 10
NUM_STEPS = 100
STEPS_PER_EXECUTION = 10


class ParameterServerCustomTrainingLoopTest(tf.test.TestCase):
  """Test to demonstrate custom training loop with ParameterServerStrategy."""

  def create_in_process_cluster(self, num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver."""
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

    cluster_dict = {}
    cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
    if num_ps > 0:
      cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

    cluster_spec = tf.train.ClusterSpec(cluster_dict)

    # Workers need some inter_ops threads to work properly.
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
      worker_config.inter_op_parallelism_threads = num_workers + 1

    for i in range(num_workers):
      tf.distribute.Server(
          cluster_spec,
          job_name="worker",
          task_index=i,
          config=worker_config,
          protocol="grpc")

    for i in range(num_ps):
      tf.distribute.Server(
          cluster_spec, job_name="ps", task_index=i, protocol="grpc")

    return cluster_spec

  def setUp(self):
    super(ParameterServerCustomTrainingLoopTest, self).setUp()

    cluster_spec = self.create_in_process_cluster(num_workers=3, num_ps=2)
    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc")
    self.strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    self.coordinator = (
        tf.distribute.experimental.coordinator.ClusterCoordinator(
            self.strategy))

  def testCustomTrainingLoop(self):

    coordinator, strategy = self.coordinator, self.strategy

    def per_worker_dataset_fn():

      def dataset_fn(_):
        return tf.data.Dataset.from_tensor_slices((tf.random.uniform(
            (6, 10)), tf.random.uniform((6, 10)))).batch(2).repeat()

      return strategy.distribute_datasets_from_function(dataset_fn)

    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    with strategy.scope():
      model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
      optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
      train_accuracy = tf.keras.metrics.CategoricalAccuracy(
          name="train_accuracy")

    @tf.function
    def worker_train_fn(iterator):

      def replica_fn(inputs):
        """Training loop function."""
        batch_data, labels = inputs
        with tf.GradientTape() as tape:
          predictions = model(batch_data, training=True)
          loss = tf.keras.losses.CategoricalCrossentropy(
              reduction=tf.keras.losses.Reduction.NONE)(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)

      for _ in tf.range(STEPS_PER_EXECUTION):
        strategy.run(replica_fn, args=(next(iterator),))

    for epoch in range(NUM_EPOCHS):

      distributed_iterator = iter(per_worker_dataset)

      for step in range(0, NUM_STEPS, STEPS_PER_EXECUTION):
        coordinator.schedule(worker_train_fn, args=(distributed_iterator,))
        logging.info("Epoch %d, step %d scheduled.", epoch, step)

      logging.info("Now joining at epoch %d.", epoch)
      coordinator.join()
      logging.info(
          "Finished joining at epoch %d. Training accuracy: %f. "
          "Total iterations: %d", epoch, train_accuracy.result(),
          optimizer.iterations.value())

      if epoch < NUM_EPOCHS - 1:
        train_accuracy.reset_states()


if __name__ == "__main__":
  tf.__internal__.distribute.multi_process_runner.test_main()
