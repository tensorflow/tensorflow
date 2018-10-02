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
r"""Demonstrates a regression on Boston housing data.

  This example demonstrates how to run experiments with TF Boosted Trees on
  a regression dataset. We split all the data into 20% test and 80% train,
  and are using l2 loss and l2 regularization.

  Example Usage:

  python tensorflow/contrib/boosted_trees/examples/boston.py \
  --batch_size=404 --output_dir="/tmp/boston" --depth=4 --learning_rate=0.1 \
  --num_eval_steps=1 --num_trees=500 --l2=0.001 \
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
import os
import sys
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch import custom_export_strategy
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeRegressor
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn import learn_runner
from tensorflow.python.util import compat

_BOSTON_NUM_FEATURES = 13


# Main config - creates a TF Boosted Trees Estimator based on flags.
def _get_tfbt(output_dir, feature_cols):
  """Configures TF Boosted Trees estimator based on flags."""
  learner_config = learner_pb2.LearnerConfig()
  learner_config.learning_rate_tuner.fixed.learning_rate = FLAGS.learning_rate
  learner_config.regularization.l1 = 0.0
  learner_config.regularization.l2 = FLAGS.l2
  learner_config.constraints.max_tree_depth = FLAGS.depth

  run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)

  # Create a TF Boosted trees regression estimator.
  estimator = GradientBoostedDecisionTreeRegressor(
      learner_config=learner_config,
      # This should be the number of examples. For large datasets it can be
      # larger than the batch_size.
      examples_per_layer=FLAGS.batch_size,
      feature_columns=feature_cols,
      label_dimension=1,
      model_dir=output_dir,
      num_trees=FLAGS.num_trees,
      center_bias=False,
      config=run_config)
  return estimator


def _convert_fn(dtec, sorted_feature_names, num_dense, num_sparse_float,
                num_sparse_int, export_dir, unused_eval_result):
  universal_format = custom_export_strategy.convert_to_universal_format(
      dtec, sorted_feature_names, num_dense, num_sparse_float, num_sparse_int)
  with tf.gfile.GFile(os.path.join(
      compat.as_bytes(export_dir), compat.as_bytes("tree_proto")), "w") as f:
    f.write(str(universal_format))


def _make_experiment_fn(output_dir):
  """Creates experiment for gradient boosted decision trees."""
  (x_train, y_train), (x_test,
                       y_test) = tf.keras.datasets.boston_housing.load_data()

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_train},
      y=y_train,
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_test}, y=y_test, num_epochs=1, shuffle=False)

  feature_columns = [
      feature_column.real_valued_column("x", dimension=_BOSTON_NUM_FEATURES)
  ]
  feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
      feature_columns)
  serving_input_fn = tf.contrib.learn.utils.build_parsing_serving_input_fn(
      feature_spec)
  # An export strategy that outputs the feature importance and also exports
  # the internal tree representation in another format.
  export_strategy = custom_export_strategy.make_custom_export_strategy(
      "exports",
      convert_fn=_convert_fn,
      feature_columns=feature_columns,
      export_input_fn=serving_input_fn)
  return tf.contrib.learn.Experiment(
      estimator=_get_tfbt(output_dir, feature_columns),
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=None,
      eval_steps=FLAGS.num_eval_steps,
      eval_metrics=None,
      export_strategies=[export_strategy])


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
      "--num_trees",
      type=int,
      default=None,
      required=True,
      help="Number of trees to grow before stopping.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
