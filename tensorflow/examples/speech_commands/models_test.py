# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for speech commands models."""

import tensorflow as tf

from tensorflow.examples.speech_commands import models
from tensorflow.python.platform import test


class ModelsTest(test.TestCase):

  def _modelSettings(self):
    return models.prepare_model_settings(
        label_count=10,
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=20,
        window_stride_ms=10,
        feature_bin_count=40,
        preprocess="mfcc")

  def _checkModel(self, model_settings, architecture):
    model = models.create_model(model_settings, architecture)
    self.assertIsInstance(model, tf.keras.Model)
    fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
    logits = model(fingerprint_input, training=False)
    self.assertEqual([1, model_settings["label_count"]], logits.shape.as_list())
    return model

  def testPrepareModelSettings(self):
    self.assertIsNotNone(
        models.prepare_model_settings(
            label_count=10,
            sample_rate=16000,
            clip_duration_ms=1000,
            window_size_ms=20,
            window_stride_ms=10,
            feature_bin_count=40,
            preprocess="mfcc"))

  def testCreateModelConv(self):
    self._checkModel(self._modelSettings(), "conv")

  def testCreateModelLowLatencyConv(self):
    self._checkModel(self._modelSettings(), "low_latency_conv")

  def testCreateModelLowLatencyConvHasNoExtraActivations(self):
    # create_low_latency_conv_model's three fully-connected layers are
    # intentionally linear (only the first conv layer has a ReLU) -- make
    # sure a regression doesn't sneak an activation back onto them.
    model = self._checkModel(self._modelSettings(), "low_latency_conv")
    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]
    # The final Dense layer produces logits for label_count classes; the two
    # hidden Dense(128) layers before it must have no activation.
    hidden_dense_layers = [
        layer for layer in dense_layers if layer.units == 128
    ]
    self.assertLen(hidden_dense_layers, 2)
    for layer in hidden_dense_layers:
      self.assertEqual(tf.keras.activations.linear, layer.activation)

  def testCreateModelFullyConnected(self):
    self._checkModel(self._modelSettings(), "single_fc")

  def testCreateModelLowLatencySvdf(self):
    self._checkModel(self._modelSettings(), "low_latency_svdf")

  def testCreateModelTinyConv(self):
    self._checkModel(self._modelSettings(), "tiny_conv")

  def testCreateModelTinyEmbeddingConv(self):
    self._checkModel(self._modelSettings(), "tiny_embedding_conv")

  def testCreateModelBadArchitecture(self):
    model_settings = self._modelSettings()
    with self.assertRaises(Exception) as e:
      models.create_model(model_settings, "bad_architecture")
    self.assertIn("not recognized", str(e.exception))


if __name__ == "__main__":
  test.main()
