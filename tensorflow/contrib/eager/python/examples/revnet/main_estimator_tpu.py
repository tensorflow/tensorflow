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
"""Cloud TPU Estimator workflow with RevNet train on ImageNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import flags
import tensorflow as tf
from tensorflow.contrib import summary
from tensorflow.contrib.eager.python.examples.revnet import config as config_
from tensorflow.contrib.eager.python.examples.revnet import imagenet_input
from tensorflow.contrib.eager.python.examples.revnet import revnet
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def _host_call_fn(gs, loss, lr):
  """Training host call.

  Creates scalar summaries for training metrics.

  This function is executed on the CPU and should not directly reference
  any Tensors in the rest of the `model_fn`. To pass Tensors from the
  model to the `metric_fn`, provide as part of the `host_call`. See
  https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
  for more information.

  Arguments should match the list of `Tensor` objects passed as the second
  element in the tuple passed to `host_call`.

  Args:
    gs: `Tensor with shape `[batch]` for the global_step
    loss: `Tensor` with shape `[batch]` for the training loss.
    lr: `Tensor` with shape `[batch]` for the learning_rate.

  Returns:
    List of summary ops to run on the CPU host.
  """
  # Host call fns are executed FLAGS.iterations_per_loop times after one
  # TPU loop is finished, setting max_queue value to the same as number of
  # iterations will make the summary writer only flush the data to storage
  # once per loop.
  gs = gs[0]
  with summary.create_file_writer(
      FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
    with summary.always_record_summaries():
      summary.scalar("loss", loss[0], step=gs)
      summary.scalar("learning_rate", lr[0], step=gs)
      return summary.all_summary_ops()


def _metric_fn(labels, logits):
  """Evaluation metric function. Evaluates accuracy.

  This function is executed on the CPU and should not directly reference
  any Tensors in the rest of the `model_fn`. To pass Tensors from the model
  to the `metric_fn`, provide as part of the `eval_metrics`. See
  https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
  for more information.

  Arguments should match the list of `Tensor` objects passed as the second
  element in the tuple passed to `eval_metrics`.

  Args:
    labels: `Tensor` with shape `[batch]`.
    logits: `Tensor` with shape `[batch, num_classes]`.

  Returns:
    A dict of the metrics to return from evaluation.
  """
  predictions = tf.argmax(logits, axis=1)
  top_1_accuracy = tf.metrics.accuracy(labels, predictions)
  in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
  top_5_accuracy = tf.metrics.mean(in_top_5)

  return {
      "top_1_accuracy": top_1_accuracy,
      "top_5_accuracy": top_5_accuracy,
  }


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
  revnet_config = params["revnet_config"]
  model = revnet.RevNet(config=revnet_config)

  inputs = features
  if isinstance(inputs, dict):
    inputs = features["image"]

  if revnet_config.data_format == "channels_first":
    assert not FLAGS.transpose_input  # channels_first only for GPU
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
    inputs = tf.transpose(inputs, [3, 0, 1, 2])  # HWCN to NHWC

  # Normalize the image to zero mean and unit variance.
  inputs -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=inputs.dtype)
  inputs /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=inputs.dtype)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant(
        global_step, revnet_config.lr_decay_steps, revnet_config.lr_list)
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           revnet_config.momentum)
    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    logits, saved_hidden = model(inputs, training=True)
    grads, loss = model.compute_gradients(saved_hidden, labels, training=True)
    with tf.control_dependencies(model.get_updates_for(inputs)):
      train_op = optimizer.apply_gradients(
          zip(grads, model.trainable_variables), global_step=global_step)
    if not FLAGS.skip_host_call:
      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      lr_t = tf.reshape(learning_rate, [1])
      host_call = (_host_call_fn, [gs_t, loss_t, lr_t])

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, host_call=host_call)

  elif mode == tf.estimator.ModeKeys.EVAL:
    logits, _ = model(inputs, training=False)
    loss = model.compute_loss(labels=labels, logits=logits)

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(_metric_fn, [labels, logits]))

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


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # RevNet specific configuration
  revnet_config = {
      "revnet-56": config_.get_hparams_imagenet_56(),
      "revnet-104": config_.get_hparams_imagenet_104()
  }[FLAGS.revnet_config]

  if FLAGS.use_tpu:
    revnet_config.data_format = "channels_last"

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # Estimator specific configuration
  config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.
          PER_HOST_V2),
  )

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train, imagenet_eval = [
      imagenet_input.ImageNetInput(
          is_training=is_training,
          data_dir=FLAGS.data_dir,
          transpose_input=FLAGS.transpose_input,
          use_bfloat16=False) for is_training in [True, False]
  ]

  revnet_classifier = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=revnet_config.tpu_batch_size,
      eval_batch_size=revnet_config.tpu_eval_batch_size,
      config=config,
      export_to_tpu=False,
      params={"revnet_config": revnet_config})

  steps_per_epoch = revnet_config.tpu_iters_per_epoch
  eval_steps = revnet_config.tpu_eval_steps

  # pylint: disable=protected-access
  if FLAGS.mode == "eval":
    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info("Starting to evaluate.")
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = revnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info("Eval results: %s. Elapsed seconds: %d" %
                        (eval_results, elapsed_time))

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split("-")[1])
        if current_step >= revnet_config.max_train_iter:
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
    current_step = estimator._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)

    tf.logging.info(
        "Training for %d steps (%.2f epochs in total). Current"
        " step %d." % (revnet_config.max_train_iter,
                       revnet_config.max_train_iter / steps_per_epoch,
                       current_step))

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == "train":
      revnet_classifier.train(
          input_fn=imagenet_train.input_fn,
          max_steps=revnet_config.max_train_iter)

    else:
      assert FLAGS.mode == "train_and_eval"
      while current_step < revnet_config.max_train_iter:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              revnet_config.max_train_iter)
        revnet_classifier.train(
            input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info("Finished training up to step %d. Elapsed seconds %d." %
                        (next_checkpoint, int(time.time() - start_timestamp)))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info("Starting to evaluate.")
        eval_results = revnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn, steps=eval_steps)
        tf.logging.info("Eval results: %s" % eval_results)

        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info("Finished training up to step %d. Elapsed seconds %d." %
                        (revnet_config.max_train_iter, elapsed_time))

    if FLAGS.export_dir is not None:
      # The guide to serve an exported TensorFlow model is at:
      #    https://www.tensorflow.org/serving/serving_basic
      tf.logging.info("Starting to export model.")
      revnet_classifier.export_saved_model(
          export_dir_base=FLAGS.export_dir,
          serving_input_receiver_fn=imagenet_input.image_serving_input_fn)


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
      "revnet_config",
      default="revnet-56",
      help="[Optional] Architecture of network. "
      "Other options include `revnet-104`")
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
  flags.DEFINE_integer(
      "eval_timeout",
      default=None,
      help="Maximum seconds between checkpoints before evaluation terminates.")
  flags.DEFINE_integer(
      "steps_per_eval",
      default=5000,
      help=(
          "Controls how often evaluation is performed. Since evaluation is"
          " fairly expensive, it is advised to evaluate as infrequently as"
          " possible (i.e. up to --train_steps, which evaluates the model only"
          " after finishing the entire training regime)."))
  flags.DEFINE_bool(
      "transpose_input",
      default=True,
      help="Use TPU double transpose optimization")
  flags.DEFINE_string(
      "export_dir",
      default=None,
      help=("The directory where the exported SavedModel will be stored."))
  flags.DEFINE_bool(
      "skip_host_call",
      default=False,
      help=("Skip the host_call which is executed every training step. This is"
            " generally used for generating training summaries (train loss,"
            " learning rate, etc...). When --skip_host_call=false, there could"
            " be a performance drop if host_call function is slow and cannot"
            " keep up with the TPU-side computation."))
  flags.DEFINE_string(
      "mode",
      default="train_and_eval",
      help='One of {"train_and_eval", "train", "eval"}.')
  FLAGS = flags.FLAGS
  tf.app.run()
