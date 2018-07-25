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
"""Self-attention generative adversarial with eager execution.

Code for main model.

Reference [Self-Attention Generative Adversarial
Networks](https://arxiv.org/pdf/1805.08318.pdf)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python.examples.sagan import ops
tfe = tf.contrib.eager


class SelfAttentionModule(tf.keras.Model):
  """Self-attention module composed of convolutional layers."""

  def __init__(self,
               attention_features,
               original_features,
               data_format="channels_first"):
    """Initialize the module.

    Args:
      attention_features: Number of filters for the attention computation.
      original_features: Number of filters of the original Tensor.
      data_format: Either 'channels_first' or 'channels_last'
    """
    super(SelfAttentionModule, self).__init__()
    self.data_format = data_format
    # Matrix multiplication implemented as 2D Convolution
    self.f = tf.keras.layers.Conv2D(
        filters=attention_features,
        kernel_size=1,
        strides=(1, 1),
        data_format=data_format)
    self.g = tf.keras.layers.Conv2D(
        filters=attention_features,
        kernel_size=1,
        strides=(1, 1),
        data_format=data_format)
    self.h = tf.keras.layers.Conv2D(
        filters=original_features,
        kernel_size=1,
        strides=(1, 1),
        data_format=data_format)
    self.scale = tf.Variable(0., trainable=True)

  def call(self, x):
    f = self.f(x)
    g = self.g(x)
    h = self.h(x)

    f_flatten = ops.flatten_hw(f, data_format=self.data_format)
    g_flatten = ops.flatten_hw(g, data_format=self.data_format)
    h_flatten = ops.flatten_hw(h, data_format=self.data_format)

    s = tf.matmul(g_flatten, f_flatten, transpose_b=True)
    b = tf.nn.softmax(s, axis=-1)
    o = tf.matmul(b, h_flatten)
    y = self.scale * tf.reshape(o, tf.shape(x)) + x

    return y

  def compute_output_shape(self, input_shape):
    return input_shape


class SAGAN(tf.contrib.checkpoint.Checkpointable):
  """Self-attention generative adversarial network."""

  def __init__(self, config):
    """Initialize the model.

    Args:
      config: tf.contrib.training.HParams object; specifies hyperparameters
    """
    super(SAGAN, self).__init__()
    self.config = config
    self.generator = self._construct_generator()
    self.discriminator = self._construct_discriminator()

  def _construct_generator(self):
    """Construct generator."""
    # TODO(lxuechen): Add spectral normalization for WGAN
    axis = 1 if self.config.data_format == "channels_first" else 3

    generator = tf.keras.Sequential()
    generator.add(
        tf.keras.layers.InputLayer(input_shape=(self.config.latent_dim,)))
    generator.add(
        tf.keras.layers.Dense(
            units=np.prod(self.config.g_init_shape), activation=tf.nn.relu))

    if self.config.data_format == "channels_first":
      c, h, w = self.config.g_init_shape
    else:
      h, w, c = self.config.g_init_shape

    # Reshape to NHWC/NCHW
    generator.add(
        ops.BroadenHW(h=h, w=w, c=c, data_format=self.config.data_format))

    filters_list = [c // 2**p for p in range(1, self.config.num_upsamples + 1)]
    filters_list[-1] = 3  # Standard RGB images

    for filters in filters_list[:len(filters_list) // 2]:
      generator.add(
          tf.keras.layers.Conv2DTranspose(
              filters=filters,
              kernel_size=4,
              strides=(2, 2),
              use_bias=False,
              padding="SAME",
              data_format=self.config.data_format))
      generator.add(tf.keras.layers.BatchNormalization(axis=axis))
      generator.add(tf.keras.layers.Activation("relu"))

    # pylint: disable=undefined-loop-variable
    generator.add(
        SelfAttentionModule(
            original_features=filters,
            attention_features=filters // 8,
            data_format=self.config.data_format))
    # pylint: enable=undefined-loop-variable

    for filters in filters_list[len(filters_list) // 2:]:
      generator.add(
          tf.keras.layers.Conv2DTranspose(
              filters=filters,
              kernel_size=4,
              strides=(2, 2),
              use_bias=False,
              padding="SAME",
              data_format=self.config.data_format))
      if filters == 3:
        # Assume Image rescaled to [-1, 1]
        generator.add(tf.keras.layers.Activation("tanh"))
      else:
        generator.add(tf.keras.layers.BatchNormalization(axis=axis))
        generator.add(tf.keras.layers.Activation("relu"))

    return generator

  def _construct_discriminator(self):
    """Construct discriminator."""
    # TODO(lxuechen): Add spectral normalization for WGAN
    discriminator = tf.keras.Sequential()
    discriminator.add(
        tf.keras.layers.InputLayer(input_shape=self.config.image_shape))

    filters_list = [
        self.config.d_init_filters * 2**p
        for p in range(self.config.num_upsamples)
    ]

    for filters in filters_list[:(len(filters_list) + 1) // 2]:
      discriminator.add(
          tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=4,
              strides=(2, 2),
              padding="SAME",
              data_format=self.config.data_format))
      discriminator.add(tf.keras.layers.LeakyReLU(alpha=.1))

    # pylint: disable=undefined-loop-variable
    discriminator.add(
        SelfAttentionModule(
            original_features=filters,
            attention_features=filters // 8,
            data_format=self.config.data_format))
    # pylint: enable=undefined-loop-variable

    for filters in filters_list[(len(filters_list) + 1) // 2:]:
      discriminator.add(
          tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=4,
              strides=(2, 2),
              padding="SAME",
              data_format=self.config.data_format))
      discriminator.add(tf.keras.layers.LeakyReLU(alpha=.1))

    discriminator.add(tf.keras.layers.Flatten())
    discriminator.add(tf.keras.layers.Dense(units=1))

    return discriminator

  def compute_loss_and_grads(self, real_images, noise, training=True):
    """Compute loss and gradients for both generator and discriminator."""
    # TODO(lxuechen): Add gradient penalty for discriminator
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
      real_logits = self.discriminator(real_images, training=training)

      fake_images = self.generator.call(noise, training=training)
      fake_logits = self.discriminator.call(fake_images)

      g_loss = self.compute_g_loss(fake_logits)
      d_loss = self.compute_d_loss(fake_logits, real_logits)

    g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

    return g_loss, d_loss, g_grads, d_grads

  def compute_g_loss(self, fake_logits):
    return -tf.reduce_mean(fake_logits)  # Hinge loss

  def compute_d_loss(self, fake_logits, real_logits):
    # Hinge loss
    real_loss = tf.reduce_mean(tf.nn.relu(1. - real_logits))
    fake_loss = tf.reduce_mean(tf.nn.relu(1. + fake_logits))
    return real_loss + fake_loss
