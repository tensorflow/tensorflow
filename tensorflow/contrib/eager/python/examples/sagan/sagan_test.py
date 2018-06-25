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
"""Tests for self-attention generative adversarial network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.sagan import config as config_
from tensorflow.contrib.eager.python.examples.sagan import sagan
tfe = tf.contrib.eager


class SAGANTest(tf.test.TestCase):

  def setUp(self):
    super(SAGANTest, self).setUp()
    config = config_.get_hparams_mock()
    self.noise_shape = (config.batch_size, config.latent_dim)
    self.logits_shape = (config.batch_size, 1)
    self.images_shape = (config.batch_size,) + config.image_shape

    self.model = sagan.SAGAN(config=config)
    self.noise = tf.random_normal(shape=self.noise_shape)
    self.real_images = tf.random_normal(shape=self.images_shape)
    self.config = config

  def tearDown(self):
    del self.model
    del self.noise
    del self.real_images
    super(SAGANTest, self).tearDown()

  def test_generator_call(self):
    """Test `generator.__call__` function."""
    fake_images = self.model.generator(self.noise, training=False)
    self.assertEqual(fake_images.shape, self.images_shape)

  def test_generator_call_defun(self):
    """Test `generator.__call__` function with defun."""
    call_ = tfe.defun(self.model.generator.__call__)
    fake_images = call_(self.noise, training=False)
    self.assertEqual(fake_images.shape, self.images_shape)

  def test_discriminator_call(self):
    """Test `discriminator.__call__` function."""
    real_logits = self.model.discriminator(self.real_images)
    self.assertEqual(real_logits.shape, self.logits_shape)

  def test_discriminator_call_defun(self):
    """Test `discriminator.__call__` function with defun."""
    call_ = tfe.defun(self.model.discriminator.__call__)
    real_logits = call_(self.real_images)
    self.assertEqual(real_logits.shape, self.logits_shape)

  def test_compute_loss_and_grads(self):
    """Test `compute_loss_and_grads` function."""
    g_loss, d_loss, g_grads, d_grads = self.model.compute_loss_and_grads(
        self.real_images, self.noise, training=False)
    self.assertEqual(g_loss.shape, ())
    self.assertEqual(d_loss.shape, ())
    self.assertTrue(isinstance(g_grads, list))
    self.assertTrue(isinstance(d_grads, list))
    g_vars = self.model.generator.trainable_variables
    d_vars = self.model.discriminator.trainable_variables

    self.assertEqual(len(g_grads), len(g_vars))
    self.assertEqual(len(d_grads), len(d_vars))

  def test_compute_loss_and_grads_defun(self):
    """Test `compute_loss_and_grads` function with defun."""
    compute_loss_and_grads = tfe.defun(self.model.compute_loss_and_grads)
    g_loss, d_loss, g_grads, d_grads = compute_loss_and_grads(
        self.real_images, self.noise, training=False)
    self.assertEqual(g_loss.shape, ())
    self.assertEqual(d_loss.shape, ())
    self.assertTrue(isinstance(g_grads, list))
    self.assertTrue(isinstance(d_grads, list))
    g_vars = self.model.generator.trainable_variables
    d_vars = self.model.discriminator.trainable_variables

    self.assertEqual(len(g_grads), len(g_vars))
    self.assertEqual(len(d_grads), len(d_vars))


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
