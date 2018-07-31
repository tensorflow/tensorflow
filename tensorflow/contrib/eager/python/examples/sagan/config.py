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

Configuration in format of tf.contrib.training.HParams.
Supports default 128x128 ImageNet.

Reference [Self-Attention Generative Adversarial
Networks](https://arxiv.org/pdf/1805.08318.pdf)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tfe = tf.contrib.eager


def get_hparams_imagenet():
  """Configurations to train SAGAN on 128x128 ImageNet dataset."""
  config = tf.contrib.training.HParams()
  if tf.test.is_gpu_available():
    config.add_hparam("image_shape", (3, 128, 128))
    config.add_hparam("data_format", "channels_first")
    config.add_hparam("g_init_shape", (512, 4, 4))
  else:
    config.add_hparam("image_shape", (128, 128, 3))
    config.add_hparam("data_format", "channels_first")
    config.add_hparam("g_init_shape", (4, 4, 512))

  config.add_hparam("latent_dim", 128)
  config.add_hparam("update_g_once_every", 1)
  config.add_hparam("batch_size", 64)
  config.add_hparam("d_init_filters", 32)
  config.add_hparam("num_upsamples", 5)
  # (512, 4, 4) -> (3, 128, 128)
  return config


def get_hparams_mock():
  """Configurations of smaller networks for testing."""
  config = tf.contrib.training.HParams()
  if tf.test.is_gpu_available():
    config.add_hparam("image_shape", (3, 16, 16))
    config.add_hparam("data_format", "channels_first")
    config.add_hparam("g_init_shape", (32, 2, 2))
  else:
    config.add_hparam("image_shape", (16, 16, 3))
    config.add_hparam("data_format", "channels_last")
    config.add_hparam("g_init_shape", (2, 2, 32))

  config.add_hparam("latent_dim", 16)
  config.add_hparam("update_g_once_every", 1)
  config.add_hparam("batch_size", 2)
  config.add_hparam("d_init_filters", 4)
  config.add_hparam("num_upsamples", 3)
  # (32, 2, 2) -> (3, 16, 16)
  return config
