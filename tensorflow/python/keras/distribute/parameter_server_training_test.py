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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import callbacks as callbacks_lib
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.server_lib import ClusterSpec


# These vocabularies usually come from TFT or a Beam pipeline.
FEATURE_VOCAB = [
    "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong",
    "wonder_woman"
]
LABEL_VOCAB = ["yes", "no"]


def make_cluster(num_workers, num_ps):
  cluster_def = multi_worker_test_base.create_in_process_cluster(
      num_workers=num_workers, num_ps=num_ps, rpc_layer="grpc")
  cluster_def["chief"] = [
      "localhost:%d" % multi_worker_test_base.pick_unused_port()
  ]
  return SimpleClusterResolver(ClusterSpec(cluster_def), rpc_layer="grpc")


def make_coordinator(num_workers, num_ps, variable_partitioner=None):
  return coordinator_lib.ClusterCoordinator(
      parameter_server_strategy_v2.ParameterServerStrategyV2(
          make_cluster(num_workers, num_ps),
          variable_partitioner=variable_partitioner))


# TODO(yuefengz): move this to keras/integration_tests.
class KPLTest(test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(KPLTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=3, num_ps=2)

  def define_kpls_for_training(self, use_adapt):
    # Define KPLs under strategy's scope. Right now, if they have look up
    # tables, they will be created on the client. Their variables will be
    # created on PS. Ideally they should be cached on each worker since they
    # will not be changed in a training step.
    if use_adapt:
      feature_lookup_layer = string_lookup.StringLookup(num_oov_indices=1)
      feature_lookup_layer.adapt(FEATURE_VOCAB)
      label_lookup_layer = string_lookup.StringLookup(
          num_oov_indices=0, mask_token=None)
      label_lookup_layer.adapt(LABEL_VOCAB)
    else:
      # Do vocab shuffling.
      shuffled_vocab = FEATURE_VOCAB.copy()
      random.shuffle(shuffled_vocab)
      feature_lookup_layer = string_lookup.StringLookup(
          vocabulary=shuffled_vocab, num_oov_indices=1)
      label_lookup_layer = string_lookup.StringLookup(
          vocabulary=LABEL_VOCAB, num_oov_indices=0, mask_token=None)

    raw_feature_input = keras.layers.Input(
        shape=(3,), dtype=dtypes.string, name="feature", ragged=True)
    feature_id_input = feature_lookup_layer(raw_feature_input)

    # Model creates variables as well.
    feature_ps = keras.Model({"features": raw_feature_input}, feature_id_input)

    raw_label_input = keras.layers.Input(
        shape=(1,), dtype=dtypes.string, name="label")
    label_id_input = label_lookup_layer(raw_label_input)
    label_ps = keras.Model({"label": raw_label_input}, label_id_input)

    return feature_ps, label_ps

  def define_reverse_lookup_layer(self):
    # Only needed for serving.
    label_inverse_lookup_layer = string_lookup.StringLookup(
        num_oov_indices=0, mask_token=None, vocabulary=LABEL_VOCAB, invert=True)
    return label_inverse_lookup_layer

  @combinations.generate(
      combinations.combine(mode=["eager"], use_adapt=[True, False]))
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
        raw_dataset = dataset_ops.Dataset.from_generator(
            feature_and_label_gen,
            output_signature={
                "features": tensor_spec.TensorSpec([3], dtypes.string),
                "label": tensor_spec.TensorSpec([1], dtypes.string)
            }).shuffle(100).batch(32)

        train_dataset = raw_dataset.map(lambda x: (  # pylint: disable=g-long-lambda
            {
                "features": feature_ps(x["features"])
            }, label_ps(x["label"])))
        return train_dataset

      # Create the model. The input needs to be compatible with KPLs.
      model_input = keras.layers.Input(
          shape=(3,), dtype=dtypes.int64, name="model_input")

      # input_dim includes a mask token and an oov token.
      emb_output = keras.layers.Embedding(
          input_dim=len(FEATURE_VOCAB) + 2, output_dim=20)(
              model_input)
      emb_output = math_ops.reduce_mean(emb_output, axis=1)
      dense_output = keras.layers.Dense(
          units=1, activation="sigmoid")(
              emb_output)
      model = keras.Model({"features": model_input}, dense_output)

      optimizer = rmsprop.RMSprop(learning_rate=0.1)
      accuracy = keras.metrics.Accuracy()

    @def_function.function
    def worker_fn(iterator):

      def replica_fn(iterator):
        batch_data, labels = next(iterator)
        with backprop.GradientTape() as tape:
          pred = model(batch_data, training=True)
          loss = nn.compute_average_loss(
              keras.losses.BinaryCrossentropy(
                  reduction=losses_utils.ReductionV2.NONE)(labels, pred))
          gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        actual_pred = math_ops.cast(math_ops.greater(pred, 0.5), dtypes.int64)
        accuracy.update_state(labels, actual_pred)

      self.coordinator.strategy.run(replica_fn, args=(iterator,))

    distributed_dataset = self.coordinator.create_per_worker_dataset(dataset_fn)
    distributed_iterator = iter(distributed_dataset)
    for _ in range(4):
      accuracy.reset_states()
      for _ in range(7):
        self.coordinator.schedule(worker_fn, args=(distributed_iterator,))
      self.coordinator.join()
    self.assertGreater(accuracy.result().numpy(), 0.5)

    # Create a saved model.
    model.feature_ps = feature_ps
    model.label_ps = label_ps
    model.label_inverse_lookup_layer = self.define_reverse_lookup_layer()

    def create_serving_signature(model):

      @def_function.function
      def serve_fn(raw_features):
        raw_features = array_ops.expand_dims(raw_features, axis=0)
        transformed_features = model.feature_ps(raw_features)
        outputs = model(transformed_features)
        outputs = array_ops.squeeze(outputs, axis=0)
        outputs = math_ops.cast(math_ops.greater(outputs, 0.5), dtypes.int64)
        decoded_outputs = model.label_inverse_lookup_layer(outputs)
        return array_ops.squeeze(decoded_outputs, axis=0)

      # serving does NOT have batch dimension
      return serve_fn.get_concrete_function(
          tensor_spec.TensorSpec(
              shape=(3), dtype=dtypes.string, name="example"))

    serving_fn = create_serving_signature(model)

    saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    model.save(saved_model_dir, signatures={"serving_default": serving_fn})

    # Test the saved_model.
    loaded_serving_fn = keras.saving.save.load_model(
        saved_model_dir).signatures["serving_default"]

    # check the result w/ and w/o avenger.
    prediction0 = loaded_serving_fn(
        constant_op.constant(["avenger", "ironman", "avenger"]))["output_0"]
    self.assertIn(prediction0, ("yes", "no"))

    prediction1 = loaded_serving_fn(
        constant_op.constant(["ironman", "ironman", "unkonwn"]))["output_0"]
    self.assertIn(prediction1, ("yes", "no"))


class ModelFitTest(test.TestCase, parameterized.TestCase):

  def _model_compile(self,
                     steps_per_execution=1,
                     run_eagerly=False,
                     with_normalization_layer=False):

    class ResultAssertingCallback(callbacks_lib.Callback):

      def __init__(self):
        self._prev_epoch = -1

      def on_epoch_end(self, epoch, logs=None):
        logging.info("testModelFit: epoch=%r, logs=%r", epoch, logs)
        if epoch <= self._prev_epoch:
          raise RuntimeError("Epoch is supposed to be larger than previous.")
        self._prev_epoch = epoch
        if (logs.get("loss", None) is None or
            not isinstance(logs["loss"], np.floating)):
          raise RuntimeError("loss is supposed to be in the logs and float.")

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        make_cluster(3, 2),
        variable_partitioner=sharded_variable.FixedShardsPartitioner(2))
    with strategy.scope():
      model = sequential.Sequential([core_layers.Dense(10)])
      if with_normalization_layer:
        norm = keras.layers.BatchNormalization(
            axis=-1, input_shape=(4, 4, 3), momentum=0.8)
        model.add(norm)

    model.compile(
        gradient_descent.SGD(),
        loss="mse",
        steps_per_execution=steps_per_execution,
        run_eagerly=run_eagerly)
    return model, [ResultAssertingCallback()]

  def _model_fit(self,
                 steps_per_execution=1,
                 validation_data=None,
                 x=None,
                 steps_per_epoch=10,
                 run_eagerly=False,
                 with_normalization_layer=False):
    model, callbacks = self._model_compile(steps_per_execution, run_eagerly,
                                           with_normalization_layer)

    def dataset_fn(input_context):
      del input_context
      x = random_ops.random_uniform((10, 10))
      y = random_ops.random_uniform((10,))
      return dataset_ops.DatasetV2.from_tensor_slices(
          (x, y)).shuffle(10).repeat().batch(2)

    x = x or dataset_creator.DatasetCreator(dataset_fn)

    model.fit(
        x,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        callbacks=callbacks,
        validation_data=validation_data)
    return model

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelFit(self):
    model = self._model_fit()
    self.assertEqual(model.optimizer.iterations, 100)
    return model

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelFitWithNormalizationLayer(self):
    model = self._model_fit(with_normalization_layer=True)
    self.assertEqual(model.optimizer.iterations, 100)

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelFitWithStepsPerExecution(self):
    model = self._model_fit(steps_per_execution=10)
    self.assertEqual(model.optimizer.iterations, 100)

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelFitWithNoStepsPerEpoch(self):
    with self.assertRaisesRegex(
        ValueError, "`steps_per_epoch` must be specified with "
        "`ParameterServerStrategy`."):
      self._model_fit(steps_per_epoch=None)

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelFitWithRunEagerly(self):
    with self.assertRaisesRegex(
        ValueError, "When using `Model` with `ParameterServerStrategy`, "
        "`run_eagerly` is not supported."):
      self._model_fit(run_eagerly=True)

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelFitWithValidationData(self):
    with self.assertRaisesRegex(
        NotImplementedError, "Evaluation in `model.fit` with "
        "`ParameterServerStrategy` is not yet supported."):
      self._model_fit(
          validation_data=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelFitWithDatasetInstance(self):
    with self.assertRaisesRegex(
        NotImplementedError, "Only `DatasetCreator` input is supported in "
        "`ParameterServerStrategy` at this time."):
      self._model_fit(x=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelEvaluate(self):
    model, _ = self._model_compile()
    with self.assertRaisesRegex(
        NotImplementedError, "`model.evaluate` is not yet supported with "
        "`ParameterServerStrategy`."):
      model.evaluate(x=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testModelPredict(self):
    model, _ = self._model_compile()
    with self.assertRaisesRegex(
        NotImplementedError, "`model.predict` is not yet supported with "
        "`ParameterServerStrategy`."):
      model.predict(x=dataset_ops.DatasetV2.from_tensor_slices([1, 1]))

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testClusterCoordinatorSingleInstance(self):
    model = self._model_fit()
    strategy = model.distribute_strategy
    self.assertIs(strategy._cluster_coordinator,
                  coordinator_lib.ClusterCoordinator(strategy))


class ShardedVariableTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        make_cluster(3, 2),
        variable_partitioner=sharded_variable.FixedShardsPartitioner(2))

  def test_keras_layer_setattr(self):

    class Layer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        self.w = variables_lib.Variable([0, 1])
        self.b = variables_lib.Variable([2, 3], trainable=False)

    with self.strategy.scope():
      layer = Layer()

    self.assertLen(layer.trainable_weights, 2)
    self.assertEqual(layer.trainable_weights[0], [0])
    self.assertEqual(layer.trainable_weights[1], [1])
    self.assertLen(layer.non_trainable_weights, 2)
    self.assertEqual(layer.non_trainable_weights[0], [2])
    self.assertEqual(layer.non_trainable_weights[1], [3])
    self.assertAllEqual(layer.weights,
                        layer.trainable_weights + layer.non_trainable_weights)
    self.assertAllEqual(layer.trainable_weights, layer.trainable_variables)
    self.assertAllEqual(layer.weights, layer.variables)

    checkpoint_deps = set(dep.ref for dep in layer._checkpoint_dependencies)
    self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))

  def test_keras_layer_add_weight(self):

    class Layer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        self.w = self.add_weight(
            shape=(2,),
            initializer=lambda shape, dtype: constant_op.constant([0., 1.],),
            trainable=True)
        self.b = self.add_weight(
            shape=(2,),
            initializer=lambda shape, dtype: constant_op.constant([2., 3.]),
            trainable=False)

    with self.strategy.scope():
      layer = Layer()

    self.assertLen(layer.trainable_weights, 2)
    self.assertEqual(layer.trainable_weights[0], [0.])
    self.assertEqual(layer.trainable_weights[1], [1.])
    self.assertLen(layer.non_trainable_weights, 2)
    self.assertEqual(layer.non_trainable_weights[0], [2.])
    self.assertEqual(layer.non_trainable_weights[1], [3.])
    self.assertAllEqual(layer.weights,
                        layer.trainable_weights + layer.non_trainable_weights)
    self.assertAllEqual(layer.trainable_weights, layer.trainable_variables)
    self.assertAllEqual(layer.weights, layer.variables)

    checkpoint_deps = set(dep.ref for dep in layer._checkpoint_dependencies)
    self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
