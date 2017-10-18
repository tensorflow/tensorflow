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
r"""Demonstrates multiclass MNIST TF Boosted trees example.

  This example demonstrates how to run experiments with TF Boosted Trees on
  a MNIST dataset. We are using layer by layer boosting with diagonal hessian
  strategy for multiclass handling, and cross entropy loss.

  Example Usage:
  python tensorflow/contrib/boosted_trees/examples/mnist.py \
  --output_dir="/tmp/mnist" --depth=4 --learning_rate=0.3 --batch_size=60000  \
  --examples_per_layer=60000 --eval_batch_size=10000 --num_eval_steps=1 \
  --num_trees=10 --l2=1 --vmodule=training_ops=1

  When training is done, accuracy on eval data is reported. Point tensorboard
  to the directory for the run to see how the training progresses:

  tensorboard --logdir=/tmp/mnist

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.learn import learn_runner


def get_input_fn(dataset_split,
                 batch_size,
                 capacity=10000,
                 min_after_dequeue=3000):
  """Input function over MNIST data."""

  def _input_fn():
    """Prepare features and labels."""
    images_batch, labels_batch = tf.train.shuffle_batch(
        tensors=[dataset_split.images,
                 dataset_split.labels.astype(np.int32)],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True,
        num_threads=4)
    features_map = {"images": images_batch}
    return features_map, labels_batch

  return _input_fn


# Main config - creates a TF Boosted Trees Estimator based on flags.
def _get_tfbt(output_dir):
  """Configures TF Boosted Trees estimator based on flags."""
  learner_config = learner_pb2.LearnerConfig()

  num_classes = 10

  learner_config.learning_rate_tuner.fixed.learning_rate = FLAGS.learning_rate
  learner_config.num_classes = num_classes
  learner_config.regularization.l1 = 0.0
  learner_config.regularization.l2 = FLAGS.l2 / FLAGS.examples_per_layer
  learner_config.constraints.max_tree_depth = FLAGS.depth

  growing_mode = learner_pb2.LearnerConfig.LAYER_BY_LAYER
  learner_config.growing_mode = growing_mode
  run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)

  learner_config.multi_class_strategy = (
      learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)

  # Create a TF Boosted trees estimator that can take in custom loss.
  estimator = GradientBoostedDecisionTreeClassifier(
      learner_config=learner_config,
      n_classes=num_classes,
      examples_per_layer=FLAGS.examples_per_layer,
      model_dir=output_dir,
      num_trees=FLAGS.num_trees,
      center_bias=False,
      config=run_config)
  return estimator


def _make_experiment_fn(output_dir):
  """Creates experiment for gradient boosted decision trees."""
  data = tf.contrib.learn.datasets.mnist.load_mnist()
  train_input_fn = get_input_fn(data.train, FLAGS.batch_size)
  eval_input_fn = get_input_fn(data.validation, FLAGS.eval_batch_size)

  return tf.contrib.learn.Experiment(
      estimator=_get_tfbt(output_dir),
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=None,
      eval_steps=FLAGS.num_eval_steps,
      eval_metrics=None)


def main(unused_argv):
  learn_runner.run(
      experiment_fn=_make_experiment_fn,
      output_dir=FLAGS.output_dir,
      schedule="train_and_evaluate")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  # Define the list of flags that users can change.
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Choose the dir for the output.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=1000,
      help="The batch size for reading data.")
  parser.add_argument(
      "--eval_batch_size",
      type=int,
      default=1000,
      help="Size of the batch for eval.")
  parser.add_argument(
      "--num_eval_steps",
      type=int,
      default=1,
      help="The number of steps to run evaluation for.")
  # Flags for gradient boosted trees config.
  parser.add_argument(
      "--depth", type=int, default=4, help="Maximum depth of weak learners.")
  parser.add_argument(
      "--l2", type=float, default=1.0, help="l2 regularization per batch.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.1,
      help="Learning rate (shrinkage weight) with which each new tree is added."
  )
  parser.add_argument(
      "--examples_per_layer",
      type=int,
      default=1000,
      help="Number of examples to accumulate stats for per layer.")
  parser.add_argument(
      "--num_trees",
      type=int,
      default=None,
      required=True,
      help="Number of trees to grow before stopping.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
