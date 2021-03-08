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
"""Tests for TPUStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tempfile

from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")

# These vocabularies usually come from TFT or a Beam pipeline.
FEATURE_VOCAB = [
    "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong",
    "wonder_woman"
]
LABEL_VOCAB = ["yes", "no"]


def get_tpu_cluster_resolver():
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy():
  resolver = get_tpu_cluster_resolver()
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  return tf.distribute.experimental.TPUStrategy(resolver)


class TpuStrategyTest(tf.test.TestCase):

  def define_kpls_for_training(self, use_adapt):
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
      feature_lookup_layer = (
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=FEATURE_VOCAB, num_oov_indices=1))
      label_lookup_layer = (
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=LABEL_VOCAB, num_oov_indices=0, mask_token=None))

    raw_feature_input = tf.keras.layers.Input(
        shape=(3,), dtype=tf.dtypes.string, name="feature", ragged=True)
    feature_id_input = feature_lookup_layer(raw_feature_input)
    feature_mapper = tf.keras.Model({"features": raw_feature_input},
                                    feature_id_input)

    raw_label_input = tf.keras.layers.Input(
        shape=(1,), dtype=tf.dtypes.string, name="label")
    label_id_input = label_lookup_layer(raw_label_input)
    label_mapper = tf.keras.Model({"label": raw_label_input}, label_id_input)

    return feature_mapper, label_mapper

  def define_inverse_lookup_layer(self):
    # Only needed for serving.
    label_inverse_lookup_layer = (
        tf.keras.layers.experimental.preprocessing.StringLookup(
            num_oov_indices=0,
            mask_token=None,
            vocabulary=LABEL_VOCAB,
            invert=True))
    return label_inverse_lookup_layer

  def test_keras_metric_outside_strategy_scope_per_replica(self):
    strategy = get_tpu_strategy()
    metric = tf.keras.metrics.Mean("test_metric", dtype=tf.float32)

    dataset = tf.data.Dataset.range(strategy.num_replicas_in_sync * 2).batch(2)
    dataset = strategy.experimental_distribute_dataset(dataset)

    @tf.function
    def step_fn(i):
      metric.update_state(i)

    with self.assertRaisesRegex(
        ValueError, "Trying to run metric.update_state "
        "in replica context"):
      with strategy.scope():
        for i in dataset:
          strategy.run(step_fn, args=(i,))

  def test_train_and_serve(self):
    strategy = get_tpu_strategy()
    use_adapt = False

    with strategy.scope():
      feature_mapper, label_mapper = self.define_kpls_for_training(use_adapt)

      def dataset_fn(_):

        def feature_and_label_gen():
          # Generator of dataset.
          while True:
            features = random.sample(FEATURE_VOCAB, 3)
            label = ["yes"] if "avenger" in features else ["no"]
            yield {"features": features, "label": label}

        raw_dataset = tf.data.Dataset.from_generator(
            feature_and_label_gen,
            output_signature={
                "features": tf.TensorSpec([3], tf.dtypes.string),
                "label": tf.TensorSpec([1], tf.dtypes.string)
            }).shuffle(100).batch(32)

        train_dataset = raw_dataset.map(lambda x: (  # pylint: disable=g-long-lambda
            {
                "features": feature_mapper(x["features"])
            }, label_mapper(x["label"])))
        return train_dataset

      # Create the model. The input needs to be compatible with KPLs.
      model_input = tf.keras.layers.Input(
          shape=(3,), dtype=tf.dtypes.int64, name="model_input")

      # input_dim includes a mask token and an oov token.
      emb_output = tf.keras.layers.Embedding(
          input_dim=len(FEATURE_VOCAB) + 2, output_dim=20)(
              model_input)
      emb_output = tf.math.reduce_mean(emb_output, axis=1)
      dense_output = tf.keras.layers.Dense(
          units=1, activation="sigmoid")(
              emb_output)
      model = tf.keras.Model({"features": model_input}, dense_output)

      optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
      accuracy = tf.keras.metrics.Accuracy()

      @tf.function
      def train_step(iterator):
        """The step function for one training step."""

        def step_fn(inputs):
          """The computation to run on each TPU device."""
          features, labels = inputs
          with tf.GradientTape() as tape:
            pred = model(features, training=True)
            loss = tf.keras.losses.binary_crossentropy(labels, pred)
            loss = tf.nn.compute_average_loss(loss)
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

          actual_pred = tf.cast(tf.math.greater(pred, 0.5), tf.dtypes.int64)
          accuracy.update_state(labels, actual_pred)

        strategy.run(step_fn, args=(next(iterator),))

      distributed_dataset = strategy.distribute_datasets_from_function(
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

      # Create a saved model.
      model.feature_mapper = feature_mapper
      model.label_mapper = label_mapper
      model.label_inverse_lookup_layer = self.define_inverse_lookup_layer()

      def create_serving_signature(model):

        @tf.function
        def serve_fn(raw_features):
          raw_features = tf.expand_dims(raw_features, axis=0)
          transformed_features = model.feature_mapper(raw_features)
          outputs = model(transformed_features)
          outputs = tf.squeeze(outputs, axis=0)
          outputs = tf.cast(tf.math.greater(outputs, 0.5), tf.dtypes.int64)
          decoded_outputs = model.label_inverse_lookup_layer(outputs)
          return tf.squeeze(decoded_outputs, axis=0)

        # Serving does NOT have batch dimension
        return serve_fn.get_concrete_function(
            tf.TensorSpec(shape=(3), dtype=tf.dtypes.string, name="example"))

      serving_fn = create_serving_signature(model)

      saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
      tf.saved_model.save(
          model, saved_model_dir, signatures={"serving_default": serving_fn})

    # Test the saved_model.
    loaded_serving_fn = tf.keras.models.load_model(
        saved_model_dir).signatures["serving_default"]

    # Check model calling with serving signature.
    prediction1 = loaded_serving_fn(
        tf.constant(["avenger", "ironman", "avenger"]))["output_0"]
    self.assertIn(prediction1, ("yes", "no"))

    prediction2 = loaded_serving_fn(
        tf.constant(["ironman", "ironman", "unkonwn"]))["output_0"]
    self.assertIn(prediction2, ("yes", "no"))


if __name__ == "__main__":
  tf.test.main()
