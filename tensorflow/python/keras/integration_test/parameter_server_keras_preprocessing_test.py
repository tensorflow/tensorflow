# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for ClusterCoordinator and Keras models."""

import multiprocessing
import os
import random
import tempfile
from absl.testing import parameterized
import numpy as np
import portpicker
import tensorflow as tf


# These vocabularies usually come from TFT or a Beam pipeline.
FEATURE_VOCAB = [
    "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong",
    "wonder_woman"
]
LABEL_VOCAB = ["yes", "no"]


def create_in_process_cluster(num_workers, num_ps):
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


class KPLTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(KPLTest, self).setUp()

    cluster_spec = create_in_process_cluster(num_workers=3, num_ps=2)
    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc")
    self.strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    self.coordinator = (
        tf.distribute.experimental.coordinator.ClusterCoordinator(
            self.strategy))

  def define_kpls_for_training(self, use_adapt):
    # Define KPLs under strategy's scope. Right now, if they have look up
    # tables, they will be created on the client. Their variables will be
    # created on PS. Ideally they should be cached on each worker since they
    # will not be changed in a training step.
    if use_adapt:
      feature_lookup_layer = (
          tf.keras.layers.experimental.preprocessing.StringLookup(
              num_oov_indices=1))
      feature_lookup_layer.adapt(FEATURE_VOCAB)
      label_lookup_layer = (
          tf.keras.layers.experimental.preprocessing.StringLookup(
              num_oov_indices=0, mask_token=None))
      label_lookup_layer.adapt(LABEL_VOCAB)
    else:
      # Do vocab shuffling.
      shuffled_vocab = FEATURE_VOCAB.copy()
      random.shuffle(shuffled_vocab)
      feature_lookup_layer = (
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=shuffled_vocab, num_oov_indices=1))
      label_lookup_layer = (
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=LABEL_VOCAB, num_oov_indices=0, mask_token=None))

    raw_feature_input = tf.keras.Input(
        shape=(3,), dtype=tf.string, name="feature", ragged=True)
    feature_id_input = feature_lookup_layer(raw_feature_input)

    # Model creates variables as well.
    feature_ps = tf.keras.Model({"features": raw_feature_input},
                                feature_id_input)

    raw_label_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="label")
    label_id_input = label_lookup_layer(raw_label_input)
    label_ps = tf.keras.Model({"label": raw_label_input}, label_id_input)

    return feature_ps, label_ps

  def define_reverse_lookup_layer(self):
    # Only needed for serving.
    label_inverse_lookup_layer = (
        tf.keras.layers.experimental.preprocessing.StringLookup(
            num_oov_indices=0,
            mask_token=None,
            vocabulary=LABEL_VOCAB,
            invert=True))
    return label_inverse_lookup_layer

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          mode=["eager"], use_adapt=[True, False]))
  def testTrainAndServe(self, use_adapt):

    with self.coordinator.strategy.scope():

      feature_ps, label_ps = self.define_kpls_for_training(use_adapt)

      def dataset_fn():

        def feature_and_label_gen():
          while True:
            features = random.sample(FEATURE_VOCAB, 3)
            label = ["yes"] if "avenger" in features else ["no"]
            yield {"features": features, "label": label}

        # The dataset will be created on the coordinator.
        raw_dataset = tf.data.Dataset.from_generator(
            feature_and_label_gen,
            output_signature={
                "features": tf.TensorSpec([3], tf.string),
                "label": tf.TensorSpec([1], tf.string)
            }).shuffle(100).batch(32)

        train_dataset = raw_dataset.map(lambda x: (  # pylint: disable=g-long-lambda
            {
                "features": feature_ps(x["features"])
            }, label_ps(x["label"])))
        return train_dataset

      # Create the model. The input needs to be compatible with KPLs.
      model_input = tf.keras.Input(
          shape=(3,), dtype=tf.int64, name="model_input")

      # input_dim includes a mask token and an oov token.
      emb_output = tf.keras.layers.Embedding(
          input_dim=len(FEATURE_VOCAB) + 2, output_dim=20)(
              model_input)
      emb_output = tf.reduce_mean(emb_output, axis=1)
      dense_output = tf.keras.layers.Dense(
          units=1, activation="sigmoid")(
              emb_output)
      model = tf.keras.Model({"features": model_input}, dense_output)

      optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
      accuracy = tf.keras.metrics.Accuracy()

    @tf.function
    def worker_fn(iterator):

      def replica_fn(iterator):
        batch_data, labels = next(iterator)
        with tf.GradientTape() as tape:
          pred = model(batch_data, training=True)
          loss = tf.nn.compute_average_loss(
              tf.keras.losses.BinaryCrossentropy(
                  reduction=tf.keras.losses.Reduction.NONE)(labels, pred))
          gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
        accuracy.update_state(labels, actual_pred)

      self.coordinator.strategy.run(replica_fn, args=(iterator,))

    distributed_dataset = self.coordinator.create_per_worker_dataset(dataset_fn)
    distributed_iterator = iter(distributed_dataset)
    for _ in range(4):
      accuracy.reset_state()
      for _ in range(7):
        self.coordinator.schedule(worker_fn, args=(distributed_iterator,))
      self.coordinator.join()
    self.assertGreater(accuracy.result().numpy(), 0.5)

    # Create a saved model.
    model.feature_ps = feature_ps
    model.label_ps = label_ps
    model.label_inverse_lookup_layer = self.define_reverse_lookup_layer()

    def create_serving_signature(model):

      @tf.function
      def serve_fn(raw_features):
        raw_features = tf.compat.v1.expand_dims(raw_features, axis=0)
        transformed_features = model.feature_ps(raw_features)
        outputs = model(transformed_features)
        outputs = tf.compat.v1.squeeze(outputs, axis=0)
        outputs = tf.cast(tf.greater(outputs, 0.5), tf.int64)
        decoded_outputs = model.label_inverse_lookup_layer(outputs)
        return tf.compat.v1.squeeze(decoded_outputs, axis=0)

      # serving does NOT have batch dimension
      return serve_fn.get_concrete_function(
          tf.TensorSpec(shape=(3), dtype=tf.string, name="example"))

    serving_fn = create_serving_signature(model)

    saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    model.save(saved_model_dir, signatures={"serving_default": serving_fn})

    # Test the saved_model.
    loaded_serving_fn = tf.keras.models.load_model(
        saved_model_dir).signatures["serving_default"]

    # check the result w/ and w/o avenger.
    prediction0 = loaded_serving_fn(
        tf.constant(["avenger", "ironman", "avenger"]))["output_0"]
    self.assertIn(prediction0, ("yes", "no"))

    prediction1 = loaded_serving_fn(
        tf.constant(["ironman", "ironman", "unkonwn"]))["output_0"]
    self.assertIn(prediction1, ("yes", "no"))


class KPLCreatedInDatasetsFromFunctionTest(tf.test.TestCase,
                                           parameterized.TestCase):

  def setUp(self):
    super(KPLCreatedInDatasetsFromFunctionTest, self).setUp()

    cluster_spec = create_in_process_cluster(num_workers=3, num_ps=2)
    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc")
    self.strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    self.coordinator = (
        tf.distribute.experimental.coordinator.ClusterCoordinator(
            self.strategy))

  def testKPLCreatedInDatasetsFromFunction(self):

    filepath = os.path.join(self.get_temp_dir(), "vocab")
    with open(filepath, "w") as f:
      f.write("\n".join(["earth", "wind", "and", "fire"]))

    def per_worker_dataset_fn():

      def dataset_fn(input_context):
        del input_context
        lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(
            num_oov_indices=1, vocabulary=filepath)
        x = np.array([["earth", "wind", "and", "fire"],
                      ["fire", "and", "earth", "michigan"]])
        y = np.array([0, 1])
        map_fn = lambda x, y: (lookup_layer(x), y)
        return tf.data.Dataset.from_tensor_slices(
            (x, y)).shuffle(10).repeat().batch(2).map(map_fn)

      return self.coordinator.strategy.distribute_datasets_from_function(
          dataset_fn)

    per_worker_distribute_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iter = iter(per_worker_distribute_dataset)

    @tf.function
    def worker_fn(iterator):

      def replica_fn(data):
        return data

      return self.coordinator.strategy.run(replica_fn, args=(next(iterator),))

    result = []
    for _ in range(10):
      result.append(
          self.coordinator.schedule(worker_fn, args=(per_worker_iter,)))

    self.coordinator.join()


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
