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
"""Cloud TPU Estimator workflow with RevNet train on CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import cifar_input
from tensorflow.contrib.eager.python.examples.revnet import main as main_
from tensorflow.contrib.eager.python.examples.revnet import revnet
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator as estimator_


def model_fn(features, labels, mode, params):
  """Model function required by the `tf.contrib.tpu.TPUEstimator` API.

  Args:
    features: Input images
    labels: Labels of images
    mode: One of `ModeKeys.TRAIN`, `ModeKeys.EVAL` or 'ModeKeys.PREDICT'
    params: A dictionary of extra parameter that might be passed

  Returns:
    An instance of `tf.contrib.tpu.TPUEstimatorSpec`
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

    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    logits, saved_hidden = model(inputs, training=True)
    grads, loss = model.compute_gradients(saved_hidden, labels, training=True)
    train_op = optimizer.apply_gradients(
        zip(grads, model.trainable_variables), global_step=global_step)

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

  elif mode == tf.estimator.ModeKeys.EVAL:
    logits, _ = model(inputs, training=False)
    loss = model.compute_loss(labels=labels, logits=logits)

    def metric_fn(labels, logits):
      predictions = tf.argmax(logits, axis=1)
      accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
      return {
          "accuracy": accuracy,
      }

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))

  else:  # Predict or export
    logits, _ = model(inputs, training=False)
    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits),
    }

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            "classify": tf.estimator.export.PredictOutput(predictions)
        })


def get_input_fn(config, data_dir, split):
  """Get the input function required by the `tf.contrib.tpu.TPUEstimator` API.

  Args:
    config: Customized hyperparameters
    data_dir: Directory where the data is stored
    split: One of `train`, `validation`, `train_all`, and `test`

  Returns:
    Input function required by the `tf.contrib.tpu.TPUEstimator` API
  """

  data_dir = os.path.join(data_dir, config.dataset)
  # Fix split-dependent hyperparameters
  if split == "train_all" or split == "train":
    data_aug = True
    epochs = config.tpu_epochs
    shuffle = True
  else:
    data_aug = False
    epochs = 1
    shuffle = False

  def input_fn(params):
    """Input function required by the `tf.contrib.tpu.TPUEstimator` API."""
    batch_size = params["batch_size"]
    return cifar_input.get_ds_from_tfrecords(
        data_dir=data_dir,
        split=split,
        data_aug=data_aug,
        batch_size=batch_size,  # per-shard batch size
        epochs=epochs,
        shuffle=shuffle,
        prefetch=batch_size,  # per-shard batch size
        data_format=config.data_format)

  return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # RevNet specific configuration
  config = main_.get_config(config_name=FLAGS.config, dataset=FLAGS.dataset)

  if FLAGS.use_tpu:
    tf.logging.info("Using TPU.")
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  else:
    tpu_cluster_resolver = None

  # TPU specific configuration
  tpu_config = tf.contrib.tpu.TPUConfig(
      # Recommended to be set as number of global steps for next checkpoint
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.num_shards)

  # Estimator specific configuration
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tpu_config,
  )

  # Construct TPU Estimator
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=config.tpu_batch_size,
      eval_batch_size=config.tpu_eval_batch_size,
      config=run_config,
      params={"config": config})

  # Construct input functions
  train_input_fn = get_input_fn(
      config=config, data_dir=FLAGS.data_dir, split="train_all")
  eval_input_fn = get_input_fn(
      config=config, data_dir=FLAGS.data_dir, split="test")

  # Disabling a range within an else block currently doesn't work
  # due to https://github.com/PyCQA/pylint/issues/872
  # pylint: disable=protected-access
  if FLAGS.mode == "eval":
    # TPUEstimator.evaluate *requires* a steps argument.
    # Note that the number of examples used during evaluation is
    # --eval_steps * --batch_size.
    # So if you change --batch_size then change --eval_steps too.
    eval_steps = 10000 // config.tpu_eval_batch_size

    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info("Starting to evaluate.")
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info("Eval results: %s. Elapsed seconds: %d" %
                        (eval_results, elapsed_time))

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split("-")[1])
        if current_step >= config.max_train_iter:
          tf.logging.info(
              "Evaluation finished after training step %d" % current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            "Checkpoint %s no longer exists, skipping checkpoint" % ckpt)

  else:  # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    current_step = estimator_._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)
    tf.logging.info("Training for %d steps . Current"
                    " step %d." % (config.max_train_iter, current_step))

    start_timestamp = time.time()  # This time will include compilation time
    if FLAGS.mode == "train":
      estimator.train(input_fn=train_input_fn, max_steps=config.max_train_iter)
    else:
      eval_steps = 10000 // config.tpu_eval_batch_size
      assert FLAGS.mode == "train_and_eval"
      while current_step < config.max_train_iter:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              config.max_train_iter)
        estimator.train(input_fn=train_input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be consistently excluded modulo the batch size.
        tf.logging.info("Starting to evaluate.")
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn, steps=eval_steps)
        tf.logging.info("Eval results: %s" % eval_results)

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info("Finished training up to step %d. Elapsed seconds %d." %
                    (config.max_train_iter, elapsed_time))
  # pylint: enable=protected-access


if __name__ == "__main__":
  # Cloud TPU Cluster Resolver flags
  flags.DEFINE_string(
      "tpu",
      default=None,
      help="The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
      "url.")
  flags.DEFINE_string(
      "tpu_zone",
      default=None,
      help="[Optional] GCE zone where the Cloud TPU is located in. If not "
      "specified, we will attempt to automatically detect the GCE project from "
      "metadata.")
  flags.DEFINE_string(
      "gcp_project",
      default=None,
      help="[Optional] Project name for the Cloud TPU-enabled project. If not "
      "specified, we will attempt to automatically detect the GCE project from "
      "metadata.")

  # Model specific parameters
  flags.DEFINE_string(
      "data_dir", default=None, help="Directory to load tfrecords")
  flags.DEFINE_string(
      "model_dir",
      default=None,
      help="[Optional] Directory to store the model information")
  flags.DEFINE_string(
      "dataset",
      default="cifar-10",
      help="[Optional] The dataset used; either `cifar-10` or `cifar-100`")
  flags.DEFINE_string(
      "config",
      default="revnet-38",
      help="[Optional] Architecture of network. "
      "Other options include `revnet-110` and `revnet-164`")
  flags.DEFINE_boolean(
      "use_tpu", default=True, help="[Optional] Whether to use TPU")
  flags.DEFINE_integer(
      "num_shards", default=8, help="Number of shards (TPU chips).")
  flags.DEFINE_integer(
      "iterations_per_loop",
      default=100,
      help=(
          "Number of steps to run on TPU before feeding metrics to the CPU."
          " If the number of iterations in the loop would exceed the number of"
          " train steps, the loop will exit before reaching"
          " --iterations_per_loop. The larger this value is, the higher the"
          " utilization on the TPU."))
  flags.DEFINE_string(
      "mode",
      default="train_and_eval",
      help="[Optional] Mode to run: train, eval, train_and_eval")
  flags.DEFINE_integer(
      "eval_timeout", 60 * 60 * 24,
      "Maximum seconds between checkpoints before evaluation terminates.")
  flags.DEFINE_integer(
      "steps_per_eval",
      default=1000,
      help=(
          "Controls how often evaluation is performed. Since evaluation is"
          " fairly expensive, it is advised to evaluate as infrequently as"
          " possible (i.e. up to --train_steps, which evaluates the model only"
          " after finishing the entire training regime)."))
  FLAGS = flags.FLAGS
  tf.app.run()
