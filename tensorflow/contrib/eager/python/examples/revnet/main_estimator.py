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
"""Estimator workflow with RevNet train on CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import cifar_input
from tensorflow.contrib.eager.python.examples.revnet import main as main_
from tensorflow.contrib.eager.python.examples.revnet import revnet


def model_fn(features, labels, mode, params):
  """Function specifying the model that is required by the `tf.estimator` API.

  Args:
    features: Input images
    labels: Labels of images
    mode: One of `ModeKeys.TRAIN`, `ModeKeys.EVAL` or 'ModeKeys.PREDICT'
    params: A dictionary of extra parameter that might be passed

  Returns:
    An instance of `tf.estimator.EstimatorSpec`
  """

  inputs = features
  if isinstance(inputs, dict):
    inputs = features["image"]

  config = params["config"]
  model = revnet.RevNet(config=config)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant(
        global_step, config.lr_decay_steps, config.lr_list)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=config.momentum)
    logits, saved_hidden = model(inputs, training=True)
    grads, loss = model.compute_gradients(saved_hidden, labels, training=True)
    with tf.control_dependencies(model.get_updates_for(inputs)):
      train_op = optimizer.apply_gradients(
          zip(grads, model.trainable_variables), global_step=global_step)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  else:
    logits, _ = model(inputs, training=False)
    predictions = tf.argmax(logits, axis=1)
    probabilities = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.EVAL:
      loss = model.compute_loss(labels=labels, logits=logits)
      return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metric_ops={
              "accuracy":
                  tf.metrics.accuracy(labels=labels, predictions=predictions)
          })

    else:  # mode == tf.estimator.ModeKeys.PREDICT
      result = {
          "classes": predictions,
          "probabilities": probabilities,
      }

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          export_outputs={
              "classify": tf.estimator.export.PredictOutput(result)
          })


def get_input_fn(config, data_dir, split):
  """Get the input function that is required by the `tf.estimator` API.

  Args:
    config: Customized hyperparameters
    data_dir: Directory where the data is stored
    split: One of `train`, `validation`, `train_all`, and `test`

  Returns:
    Input function required by the `tf.estimator` API
  """

  data_dir = os.path.join(data_dir, config.dataset)
  # Fix split-dependent hyperparameters
  if split == "train_all" or split == "train":
    data_aug = True
    batch_size = config.batch_size
    epochs = config.epochs
    shuffle = True
    prefetch = config.batch_size
  else:
    data_aug = False
    batch_size = config.eval_batch_size
    epochs = 1
    shuffle = False
    prefetch = config.eval_batch_size

  def input_fn():
    """Input function required by the `tf.estimator.Estimator` API."""
    return cifar_input.get_ds_from_tfrecords(
        data_dir=data_dir,
        split=split,
        data_aug=data_aug,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
        prefetch=prefetch,
        data_format=config.data_format)

  return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # RevNet specific configuration
  config = main_.get_config(config_name=FLAGS.config, dataset=FLAGS.dataset)

  # Estimator specific configuration
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,  # Directory for storing checkpoints
      tf_random_seed=config.seed,
      save_summary_steps=config.log_every,
      save_checkpoints_steps=config.log_every,
      session_config=None,  # Using default
      keep_checkpoint_max=100,
      keep_checkpoint_every_n_hours=10000,  # Using default
      log_step_count_steps=config.log_every,
      train_distribute=None  # Default not use distribution strategy
  )

  # Construct estimator
  revnet_estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params={"config": config})

  # Construct input functions
  train_input_fn = get_input_fn(
      config=config, data_dir=FLAGS.data_dir, split="train_all")
  eval_input_fn = get_input_fn(
      config=config, data_dir=FLAGS.data_dir, split="test")

  # Train and evaluate estimator
  revnet_estimator.train(input_fn=train_input_fn)
  revnet_estimator.evaluate(input_fn=eval_input_fn)

  if FLAGS.export:
    input_shape = (None,) + config.input_shape
    inputs = tf.placeholder(tf.float32, shape=input_shape)
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "image": inputs
    })
    revnet_estimator.export_savedmodel(FLAGS.model_dir, input_fn)


if __name__ == "__main__":
  flags.DEFINE_string(
      "data_dir", default=None, help="Directory to load tfrecords")
  flags.DEFINE_string(
      "model_dir",
      default=None,
      help="[Optional] Directory to store the training information")
  flags.DEFINE_string(
      "dataset",
      default="cifar-10",
      help="[Optional] The dataset used; either `cifar-10` or `cifar-100`")
  flags.DEFINE_boolean(
      "export",
      default=False,
      help="[Optional] Export the model for serving if True")
  flags.DEFINE_string(
      "config",
      default="revnet-38",
      help="[Optional] Architecture of network. "
      "Other options include `revnet-110` and `revnet-164`")
  FLAGS = flags.FLAGS
  tf.app.run()
