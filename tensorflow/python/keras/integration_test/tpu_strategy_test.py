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
"""Tests for TPUStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy():
  resolver = get_tpu_cluster_resolver()
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  return tf.distribute.experimental.TPUStrategy(resolver)


class TpuStrategyTest(tf.test.TestCase):

  def test_keras_metric_outside_strategy_scope_per_replica(self):
    strategy = get_tpu_strategy()
    metric = tf.keras.metrics.Mean("test_metric", dtype=tf.float32)

    dataset = tf.data.Dataset.range(strategy.num_replicas_in_sync * 2).batch(2)
    dataset = strategy.experimental_distribute_dataset(dataset)

    @tf.function
    def step_fn(i):
      metric.update_state(i)

    with self.assertRaisesRegex(ValueError, "Trying to run metric.update_state "
                                            "in replica context"):
      with strategy.scope():
        for i in dataset:
          strategy.run(step_fn, args=(i,))


if __name__ == "__main__":
  tf.test.main()
