# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A stand-alone example for tf.learn's random forest model on mnist."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import random_forest
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '', 'Base directory for output models.')
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

flags.DEFINE_integer('train_steps', 1000, 'Number of training steps.')
flags.DEFINE_string('batch_size', 1000,
                    'Number of examples in a training batch.')
flags.DEFINE_integer('num_trees', 100, 'Number of trees in the forest.')
flags.DEFINE_integer('max_nodes', 1000, 'Max total nodes in a single tree.')


def build_estimator(model_dir):
  """Build an estimator."""
  params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
      num_classes=10, num_features=784,
      num_trees=FLAGS.num_trees, max_nodes=FLAGS.max_nodes)
  return random_forest.TensorForestEstimator(params, model_dir=model_dir)


def train_and_eval():
  """Train and evaluate the model."""
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print('model directory = %s' % model_dir)

  estimator = build_estimator(model_dir)

  # TensorForest's LossMonitor allows training to terminate early if the
  # forest is no longer growing.
  early_stopping_rounds = 100
  check_every_n_steps = 100
  monitor = random_forest.LossMonitor(early_stopping_rounds,
                                      check_every_n_steps)

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

  estimator.fit(x=mnist.train.images, y=mnist.train.labels,
                batch_size=FLAGS.batch_size, monitors=[monitor])

  results = estimator.evaluate(x=mnist.test.images, y=mnist.test.labels,
                               batch_size=FLAGS.batch_size)
  for key in sorted(results):
    print('%s: %s' % (key, results[key]))


def main(_):
  train_and_eval()


if __name__ == '__main__':
  tf.app.run()
