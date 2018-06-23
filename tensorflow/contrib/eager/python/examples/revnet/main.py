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
"""Eager execution workflow with RevNet train on CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import cifar_input
from tensorflow.contrib.eager.python.examples.revnet import config as config_
from tensorflow.contrib.eager.python.examples.revnet import revnet
tfe = tf.contrib.eager


def main(_):
  """Eager execution workflow with RevNet trained on CIFAR-10."""
  if FLAGS.data_dir is None:
    raise ValueError("No supplied data directory")

  if not os.path.exists(FLAGS.data_dir):
    raise ValueError("Data directory {} does not exist".format(FLAGS.data_dir))

  tf.enable_eager_execution()
  config = config_.get_hparams_cifar_38()
  model = revnet.RevNet(config=config)

  ds_train = cifar_input.get_ds_from_tfrecords(
      data_dir=FLAGS.data_dir,
      split="train",
      data_aug=True,
      batch_size=config.batch_size,
      epochs=config.epochs,
      shuffle=config.shuffle,
      data_format=config.data_format,
      dtype=config.dtype,
      prefetch=config.prefetch)

  ds_validation = cifar_input.get_ds_from_tfrecords(
      data_dir=FLAGS.data_dir,
      split="validation",
      data_aug=False,
      batch_size=config.eval_batch_size,
      epochs=1,
      data_format=config.data_format,
      dtype=config.dtype,
      prefetch=config.prefetch)

  ds_test = cifar_input.get_ds_from_tfrecords(
      data_dir=FLAGS.data_dir,
      split="test",
      data_aug=False,
      batch_size=config.eval_batch_size,
      epochs=1,
      data_format=config.data_format,
      dtype=config.dtype,
      prefetch=config.prefetch)

  global_step = tfe.Variable(1, trainable=False)

  def learning_rate():  # TODO(lxuechen): Remove once cl/201089859 is in place
    return tf.train.piecewise_constant(global_step, config.lr_decay_steps,
                                       config.lr_list)

  optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
  checkpoint = tf.train.Checkpoint(
      optimizer=optimizer, model=model, optimizer_step=global_step)

  if FLAGS.train_dir:
    summary_writer = tf.contrib.summary.create_file_writer(FLAGS.train_dir)
    if FLAGS.restore:
      latest_path = tf.train.latest_checkpoint(FLAGS.train_dir)
      checkpoint.restore(latest_path)

  for x, y in ds_train:
    loss = train_one_iter(model, x, y, optimizer, global_step=global_step)

    if global_step % config.log_every == 0:
      it_validation = ds_validation.make_one_shot_iterator()
      it_test = ds_test.make_one_shot_iterator()
      acc_validation = evaluate(model, it_validation)
      acc_test = evaluate(model, it_test)
      print("Iter {}, "
            "train loss {}, "
            "validation accuracy {}, "
            "test accuracy {}".format(global_step.numpy(), loss, acc_validation,
                                      acc_test))

      if FLAGS.train_dir:
        with summary_writer.as_default():
          with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar("Validation accuracy", acc_validation)
            tf.contrib.summary.scalar("Test accuracy", acc_test)
            tf.contrib.summary.scalar("Training loss", loss)

    if global_step.numpy() % config.save_every == 0 and FLAGS.train_dir:
      checkpoint.save(file_prefix=FLAGS.train_dir + "ckpt")


def train_one_iter(model, inputs, labels, optimizer, global_step=None):
  """Train for one iteration."""
  grads, vars_, loss = model.compute_gradients(inputs, labels, training=True)
  optimizer.apply_gradients(zip(grads, vars_), global_step=global_step)

  return loss.numpy()


def evaluate(model, iterator):
  """Compute accuracy with the given dataset iterator."""
  accuracy = tfe.metrics.Accuracy()
  for x, y in iterator:
    logits, _ = model(x, training=False)
    accuracy(
        labels=tf.cast(y, tf.int64),
        predictions=tf.argmax(logits, axis=1, output_type=tf.int64))

  return accuracy.result().numpy()


if __name__ == "__main__":
  flags.DEFINE_string(
      "train_dir",
      default=None,
      help="[Optional] Directory to store the training information")
  flags.DEFINE_string(
      "data_dir", default=None, help="Directory to load tfrecords.")
  flags.DEFINE_boolean(
      "restore",
      default=True,
      help="[Optional] Restore the latest checkpoint from `train_dir` if True")
  FLAGS = flags.FLAGS
  tf.app.run(main)
