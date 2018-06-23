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
"""Reversible residual network compatible with eager execution.

Configuration in format of tf.contrib.training.HParams.
Supports CIFAR-10, CIFAR-100, and ImageNet datasets.

Reference [The Reversible Residual Network: Backpropagation
Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tfe = tf.contrib.eager


def get_hparams_cifar_38():
  """RevNet-38 configurations for CIFAR-10/CIFAR-100."""

  config = tf.contrib.training.HParams()
  config.add_hparam("init_filters", 32)
  config.add_hparam("init_kernel", 3)
  config.add_hparam("init_stride", 1)
  config.add_hparam("n_classes", 10)
  config.add_hparam("n_rev_blocks", 3)
  config.add_hparam("n_res", [3, 3, 3])
  config.add_hparam("filters", [32, 64, 112])
  config.add_hparam("strides", [1, 2, 2])
  config.add_hparam("batch_size", 100)
  config.add_hparam("bottleneck", False)
  config.add_hparam("fused", True)
  config.add_hparam("init_max_pool", False)
  if tfe.num_gpus() > 0:
    config.add_hparam("input_shape", (3, 32, 32))
    config.add_hparam("data_format", "channels_first")
  else:
    config.add_hparam("input_shape", (32, 32, 3))
    config.add_hparam("data_format", "channels_last")

  # Training details
  config.add_hparam("weight_decay", 2e-4)
  config.add_hparam("momentum", .9)
  config.add_hparam("lr_decay_steps", [40000, 60000])
  config.add_hparam("lr_list", [1e-1, 1e-2, 1e-3])
  config.add_hparam("max_train_iter", 80000)
  config.add_hparam("seed", 1234)
  config.add_hparam("shuffle", True)
  config.add_hparam("prefetch", True)
  config.add_hparam("log_every", 50)
  config.add_hparam("save_every", 50)
  config.add_hparam("dtype", tf.float32)
  config.add_hparam("eval_batch_size", 500)
  config.add_hparam("div255", True)
  config.add_hparam("iters_per_epoch", 50000 // config.batch_size)
  config.add_hparam("epochs", config.max_train_iter // config.iters_per_epoch)

  return config


def get_hparams_imagenet_56():
  """RevNet-56 configurations for ImageNet."""

  config = tf.contrib.training.HParams()
  config.add_hparam("init_filters", 128)
  config.add_hparam("init_kernel", 7)
  config.add_hparam("init_stride", 2)
  config.add_hparam("n_classes", 1000)
  config.add_hparam("n_rev_blocks", 4)
  config.add_hparam("n_res", [2, 2, 2, 2])
  config.add_hparam("filters", [128, 256, 512, 832])
  config.add_hparam("strides", [1, 2, 2, 2])
  config.add_hparam("batch_size", 16)
  config.add_hparam("bottleneck", True)
  config.add_hparam("fused", True)
  config.add_hparam("init_max_pool", True)
  if tf.test.is_gpu_available():
    config.add_hparam("input_shape", (3, 224, 224))
    config.add_hparam("data_format", "channels_first")
  else:
    config.add_hparam("input_shape", (224, 224, 3))
    config.add_hparam("data_format", "channels_last")

  # Training details
  config.add_hparam("weight_decay", 1e-4)
  config.add_hparam("momentum", .9)
  config.add_hparam("lr_decay_steps", [160000, 320000, 480000])
  config.add_hparam("lr_list", [1e-1, 1e-2, 1e-3, 1e-4])
  config.add_hparam("max_train_iter", 600000)
  config.add_hparam("seed", 1234)
  config.add_hparam("shuffle", True)
  config.add_hparam("prefetch", True)
  config.add_hparam("log_every", 50)
  config.add_hparam("save_every", 50)
  config.add_hparam("dtype", tf.float32)
  config.add_hparam("eval_batch_size", 500)
  config.add_hparam("div255", True)
  # TODO(lxuechen): Update this according to ImageNet data
  config.add_hparam("iters_per_epoch", 50000 // config.batch_size)
  config.add_hparam("epochs", config.max_train_iter // config.iters_per_epoch)

  if config.bottleneck:
    filters = [f * 4 for f in config.filters]
    config.filters = filters

  return config
