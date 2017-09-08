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
from tensorflow.python.platform import test


class ModelsTest(test.TestCase):

  def testPrepareModelSettings(self):
    self.assertIsNotNone(
        models.prepare_model_settings(10, 16000, 1000, 20, 10, 40))

  def testCreateModelConvTraining(self):
    model_settings = models.prepare_model_settings(10, 16000, 1000, 20, 10, 40)
    with self.test_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_prob = models.create_model(fingerprint_input,
                                                 model_settings, "conv", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_prob)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_prob.name))

  def testCreateModelConvInference(self):
    model_settings = models.prepare_model_settings(10, 16000, 1000, 20, 10, 40)
    with self.test_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits = models.create_model(fingerprint_input, model_settings, "conv",
                                   False)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))

  def testCreateModelLowLatencyConvTraining(self):
    model_settings = models.prepare_model_settings(10, 16000, 1000, 20, 10, 40)
    with self.test_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_prob = models.create_model(
          fingerprint_input, model_settings, "low_latency_conv", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_prob)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_prob.name))

  def testCreateModelFullyConnectedTraining(self):
    model_settings = models.prepare_model_settings(10, 16000, 1000, 20, 10, 40)
    with self.test_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_prob = models.create_model(
          fingerprint_input, model_settings, "single_fc", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_prob)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_prob.name))

  def testCreateModelBadArchitecture(self):
    model_settings = models.prepare_model_settings(10, 16000, 1000, 20, 10, 40)
    with self.test_session():
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      with self.assertRaises(Exception) as e:
        models.create_model(fingerprint_input, model_settings,
                            "bad_architecture", True)
      self.assertTrue("not recognized" in str(e.exception))


if __name__ == "__main__":
  test.main()
