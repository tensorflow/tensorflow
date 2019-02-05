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
r"""Regression on Boston housing data using DNNBoostedTreeCombinedRegressor.

  Example Usage:

  python tensorflow/contrib/boosted_trees/examples/boston_combined.py \
  --batch_size=404 --output_dir="/tmp/boston" \
  --dnn_hidden_units="8,4" --dnn_steps_to_train=1000 \
  --tree_depth=4 --tree_learning_rate=0.1 \
  --num_trees=100 --tree_l2=0.001 --num_eval_steps=1 \
  --vmodule=training_ops=1

  When training is done, mean squared error on eval data is reported.
  Point tensorboard to the directory for the run to see how the training
  progresses:

  tensorboard --logdir=/tmp/boston

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf

from tensorflow.contrib.boosted_trees.estimator_batch.dnn_tree_combined_estimator import DNNBoostedTreeCombinedRegressor
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils

_BOSTON_NUM_FEATURES = 13


def _get_estimator(output_dir, feature_cols):
  """Configures DNNBoostedTreeCombinedRegressor based on flags."""
  learner_config = learner_pb2.LearnerConfig()
  learner_config.learning_rate_tuner.fixed.learning_rate = (
      FLAGS.tree_learning_rate)
  learner_config.regularization.l1 = 0.0
  learner_config.regularization.l2 = FLAGS.tree_l2
  learner_config.constraints.max_tree_depth = FLAGS.tree_depth

  run_config = tf.contrib.learn.RunConfig(save_summary_steps=1)

  # Create a DNNBoostedTreeCombinedRegressor estimator.
  estimator = DNNBoostedTreeCombinedRegressor(
      dnn_hidden_units=[int(x) for x in FLAGS.dnn_hidden_units.split(",")],
      dnn_feature_columns=feature_cols,
      tree_learner_config=learner_config,
      num_trees=FLAGS.num_trees,
      # This should be the number of examples. For large datasets it can be
      # larger than the batch_size.
      tree_examples_per_layer=FLAGS.batch_size,
      model_dir=output_dir,
      config=run_config,
      dnn_input_layer_to_tree=True,
      dnn_steps_to_train=FLAGS.dnn_steps_to_train)
  return estimator


def _make_experiment_fn(output_dir):
  """Creates experiment for DNNBoostedTreeCombinedRegressor."""
  (x_train, y_train), (x_test,
                       y_test) = tf.keras.datasets.boston_housing.load_data()

  train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": x_train},
      y=y_train,
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True)
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": x_test}, y=y_test, num_epochs=1, shuffle=False)

  feature_columns = [
      feature_column.real_valued_column("x", dimension=_BOSTON_NUM_FEATURES)
  ]
  feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
      feature_columns)
  serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
  export_strategies = [
      saved_model_export_utils.make_export_strategy(serving_input_fn)]
  return tf.contrib.learn.Experiment(
      estimator=_get_estimator(output_dir, feature_columns),
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=None,
      eval_steps=FLAGS.num_eval_steps,
      eval_metrics=None,
      export_strategies=export_strategies)


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
      "--batch_size",
      type=int,
      default=1000,
      help="The batch size for reading data.")
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Choose the dir for the output.")
  parser.add_argument(
      "--num_eval_steps",
      type=int,
      default=1,
      help="The number of steps to run evaluation for.")
  # Flags for configuring DNNBoostedTreeCombinedRegressor.
  parser.add_argument(
      "--dnn_hidden_units",
      type=str,
      default="8,4",
      help="Hidden layers for DNN.")
  parser.add_argument(
      "--dnn_steps_to_train",
      type=int,
      default=1000,
      help="Number of steps to train DNN.")
  parser.add_argument(
      "--tree_depth", type=int, default=4, help="Maximum depth of trees.")
  parser.add_argument(
      "--tree_l2", type=float, default=1.0, help="l2 regularization per batch.")
  parser.add_argument(
      "--tree_learning_rate",
      type=float,
      default=0.1,
      help=("Learning rate (shrinkage weight) with which each "
            "new tree is added."))
  parser.add_argument(
      "--num_trees",
      type=int,
      default=None,
      required=True,
      help="Number of trees to grow before stopping.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
