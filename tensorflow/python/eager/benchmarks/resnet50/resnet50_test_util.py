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
"""Tests and benchmarks for the ResNet50 model, executed eagerly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf


def device_and_data_format():
  if tf.config.list_physical_devices('GPU'):
    return ('/gpu:0', 'channels_first')
  return ('/cpu:0', 'channels_last')


def random_batch(batch_size, data_format, seed=None):
  """Create synthetic resnet50 images and labels for testing."""
  if seed:
    tf.random.set_seed(seed)
  shape = (3, 224, 224) if data_format == 'channels_first' else (224, 224, 3)
  shape = (batch_size,) + shape

  num_classes = 1000
  images = tf.random.uniform(shape)
  labels = tf.random.uniform([batch_size],
                             minval=0,
                             maxval=num_classes,
                             dtype=tf.int32)
  one_hot = tf.one_hot(labels, num_classes)

  return images, one_hot


def report(benchmark, label, start, num_iters, device, batch_size, data_format,
           num_replicas=1):
  avg_time = (time.time() - start) / num_iters
  dev = tf.DeviceSpec.from_string(device).device_type.lower()
  replica_str = '' if num_replicas == 1 else 'replicas_%d_' % num_replicas
  name = '%s_%s_batch_%d_%s%s' % (label, dev, batch_size,
                                  replica_str, data_format)
  extras = {'examples_per_sec': (num_replicas * batch_size) / avg_time}
  benchmark.report_benchmark(
      iters=num_iters, wall_time=avg_time, name=name, extras=extras)
