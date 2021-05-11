# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `DatasetCreator` with `Model.fit` across usages and strategies."""

import os
from absl.testing import parameterized
import numpy as np
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras import callbacks as callbacks_lib
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class DatasetCreatorModelFitTestBase(test.TestCase, parameterized.TestCase):
  """The base class for DatasetCreator with Model.fit tests."""

  def _get_dataset_fn(self, use_lookup_layer):

    if use_lookup_layer:

      filepath = os.path.join(self.get_temp_dir(), "vocab")
      with open(filepath, "w") as f:
        f.write("\n".join(["earth", "wind", "and", "fire"]))

      def dataset_fn(input_context):
        del input_context
        lookup_layer = string_lookup.StringLookup(
            num_oov_indices=1, vocabulary=filepath)
        x = np.array([["earth", "wind", "and", "fire"],
                      ["fire", "and", "earth", "michigan"]])
        y = np.array([0, 1])
        map_fn = lambda x, y: (lookup_layer(x), y)
        return dataset_ops.DatasetV2.from_tensor_slices(
            (x, y)).shuffle(10).repeat().batch(2).map(map_fn)

    else:

      def dataset_fn(input_context):
        del input_context
        x = random_ops.random_uniform((10, 10))
        y = random_ops.random_uniform((10,))
        return dataset_ops.DatasetV2.from_tensor_slices(
            (x, y)).shuffle(10).repeat().batch(2)

    return dataset_fn

  def _model_compile(self,
                     strategy,
                     steps_per_execution=1,
                     run_eagerly=False,
                     with_normalization_layer=False,
                     use_lookup_layer=False):

    class ResultAssertingCallback(callbacks_lib.Callback):
      """A callback that asserts the result of the tests."""

      def __init__(self):
        self._prev_epoch = -1

      def on_epoch_end(self, epoch, logs=None):
        logging.info("testModelFit: epoch=%r, logs=%r", epoch, logs)
        if epoch <= self._prev_epoch:
          raise RuntimeError("Epoch is supposed to be larger than previous.")
        self._prev_epoch = epoch
        is_loss_float = (
            logs.get("loss", None) is not None and
            isinstance(logs["loss"], (float, np.floating)))
        if not is_loss_float:
          raise RuntimeError("loss is supposed to be in the logs and float.")

      def on_train_end(self, logs=None):
        if self._prev_epoch != 9:
          raise RuntimeError("Unexpected last epoch: {}".format(
              self._prev_epoch))

    with strategy.scope():
      model = sequential.Sequential([core_layers.Dense(10)])
      if with_normalization_layer:
        norm = keras.layers.BatchNormalization(
            axis=-1, input_shape=(4, 4, 3), momentum=0.8)
        model.add(norm)
      model.add(core_layers.Dense(1, activation="sigmoid"))
      self._metric = keras.metrics.Accuracy()

    model.compile(
        gradient_descent.SGD(),
        loss="binary_crossentropy",
        metrics=[self._metric],
        steps_per_execution=steps_per_execution,
        run_eagerly=run_eagerly)
    return model, [ResultAssertingCallback()]

  def _model_fit(self,
                 strategy,
                 steps_per_execution=1,
                 validation_data=None,
                 x=None,
                 steps_per_epoch=10,
                 run_eagerly=False,
                 with_normalization_layer=False,
                 callbacks=None,
                 use_lookup_layer=False):
    if callbacks is None:
      callbacks = []

    model, default_callbacks = self._model_compile(strategy,
                                                   steps_per_execution,
                                                   run_eagerly,
                                                   with_normalization_layer,
                                                   use_lookup_layer)
    callbacks += default_callbacks

    x = x or dataset_creator.DatasetCreator(
        self._get_dataset_fn(use_lookup_layer))
    validation_data = (
        validation_data or
        dataset_creator.DatasetCreator(self._get_dataset_fn(use_lookup_layer)))

    model.fit(
        x,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=steps_per_epoch)
    return model

  def _model_evaluate(self,
                      strategy,
                      steps_per_execution=1,
                      validation_data=None,
                      steps=10,
                      run_eagerly=False,
                      with_normalization_layer=False,
                      callbacks=None):
    if callbacks is None:
      callbacks = []

    model, default_callbacks = self._model_compile(strategy,
                                                   steps_per_execution,
                                                   run_eagerly,
                                                   with_normalization_layer)
    callbacks += default_callbacks

    def dataset_fn(input_context):
      del input_context
      x = random_ops.random_uniform((10, 10))
      y = random_ops.random_uniform((10, 1))
      return dataset_ops.DatasetV2.from_tensor_slices(
          (x, y)).shuffle(10).repeat().batch(8)

    validation_data = (
        validation_data or dataset_creator.DatasetCreator(dataset_fn))
    model.evaluate(x=validation_data, steps=steps, callbacks=callbacks)
    return model
