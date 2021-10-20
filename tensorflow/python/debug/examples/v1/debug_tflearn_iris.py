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
"""Debug the tf-learn iris example, based on the tf-learn tutorial."""
import argparse
import sys
import tempfile

import tensorflow

from tensorflow.python import debug as tf_debug

tf = tensorflow.compat.v1

_IRIS_INPUT_DIM = 4


def main(_):
  # Generate some fake Iris data.
  # It is okay for this example because this example is about how to use the
  # debugger, not how to use machine learning to solve the Iris classification
  # problem.
  def training_input_fn():
    return ({
        "features": tf.random_normal([128, 4])
    }, tf.random_uniform([128], minval=0, maxval=3, dtype=tf.int32))

  def test_input_fn():
    return ({
        "features": tf.random_normal([32, 4])
    }, tf.random_uniform([32], minval=0, maxval=3, dtype=tf.int32))

  feature_columns = [tf.feature_column.numeric_column("features", shape=(4,))]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  model_dir = FLAGS.model_dir or tempfile.mkdtemp(prefix="debug_tflearn_iris_")

  classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[10, 20, 10],
      n_classes=3,
      model_dir=model_dir)

  if FLAGS.debug and FLAGS.tensorboard_debug_address:
    raise ValueError(
        "The --debug and --tensorboard_debug_address flags are mutually "
        "exclusive.")
  hooks = []
  if FLAGS.debug:
    config_file_path = (
        tempfile.mktemp(".tfdbg_config")
        if FLAGS.use_random_config_path else None)
    hooks.append(
        tf_debug.LocalCLIDebugHook(
            ui_type=FLAGS.ui_type,
            dump_root=FLAGS.dump_root,
            config_file_path=config_file_path))
  elif FLAGS.tensorboard_debug_address:
    hooks.append(tf_debug.TensorBoardDebugHook(FLAGS.tensorboard_debug_address))

  # Train model, using tfdbg hook.
  classifier.train(training_input_fn, steps=FLAGS.train_steps, hooks=hooks)

  # Evaluate accuracy, using tfdbg hook.
  accuracy_score = classifier.evaluate(
      test_input_fn, steps=FLAGS.eval_steps, hooks=hooks)["accuracy"]

  print("After training %d steps, Accuracy = %f" %
        (FLAGS.train_steps, accuracy_score))

  # Make predictions, using tfdbg hook.
  predict_results = classifier.predict(test_input_fn, hooks=hooks)
  print("A prediction result: %s" % next(predict_results))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--data_dir",
      type=str,
      default="/tmp/iris_data",
      help="Directory to save the training and test data in.")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Directory to save the trained model in.")
  parser.add_argument(
      "--train_steps",
      type=int,
      default=10,
      help="Number of steps to run training for.")
  parser.add_argument(
      "--eval_steps",
      type=int,
      default=1,
      help="Number of steps to run evaluation foir.")
  parser.add_argument(
      "--ui_type",
      type=str,
      default="curses",
      help="Command-line user interface type (curses | readline)")
  parser.add_argument(
      "--debug",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Use debugger to track down bad values during training. "
      "Mutually exclusive with the --tensorboard_debug_address flag.")
  parser.add_argument(
      "--dump_root",
      type=str,
      default="",
      help="Optional custom root directory for temporary debug dump data")
  parser.add_argument(
      "--use_random_config_path",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="""If set, set config file path to a random file in the temporary
      directory.""")
  parser.add_argument(
      "--tensorboard_debug_address",
      type=str,
      default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
