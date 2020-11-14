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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.speech_commands import models
from tensorflow.python.framework import test_util
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

  @test_util.run_deprecated_v1
  def testCreateModelConvTraining(self):
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_rate = models.create_model(
          fingerprint_input, model_settings, "conv", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_rate)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_rate.name))

  @test_util.run_deprecated_v1
  def testCreateModelConvInference(self):
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits = models.create_model(fingerprint_input, model_settings, "conv",
                                   False)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))

  @test_util.run_deprecated_v1
  def testCreateModelLowLatencyConvTraining(self):
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_rate = models.create_model(
          fingerprint_input, model_settings, "low_latency_conv", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_rate)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_rate.name))

  @test_util.run_deprecated_v1
  def testCreateModelFullyConnectedTraining(self):
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_rate = models.create_model(
          fingerprint_input, model_settings, "single_fc", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_rate)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_rate.name))

  def testCreateModelBadArchitecture(self):
    model_settings = self._modelSettings()
    with self.cached_session():
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      with self.assertRaises(Exception) as e:
        models.create_model(fingerprint_input, model_settings,
                            "bad_architecture", True)
      self.assertIn("not recognized", str(e.exception))

  @test_util.run_deprecated_v1
  def testCreateModelTinyConvTraining(self):
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_rate = models.create_model(
          fingerprint_input, model_settings, "tiny_conv", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_rate)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_rate.name))


if __name__ == "__main__":
  test.main()
