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
import sys

from absl import flags
import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import cifar_input
from tensorflow.contrib.eager.python.examples.revnet import config as config_
from tensorflow.contrib.eager.python.examples.revnet import revnet
tfe = tf.contrib.eager


def main(_):
  """Eager execution workflow with RevNet trained on CIFAR-10."""
  tf.enable_eager_execution()

  config = get_config(config_name=FLAGS.config, dataset=FLAGS.dataset)
  ds_train, ds_train_one_shot, ds_validation, ds_test = get_datasets(
      data_dir=FLAGS.data_dir, config=config)
  model = revnet.RevNet(config=config)
  global_step = tf.train.get_or_create_global_step()  # Ensure correct summary
  global_step.assign(1)
  learning_rate = tf.train.piecewise_constant(
      global_step, config.lr_decay_steps, config.lr_list)
  optimizer = tf.train.MomentumOptimizer(
      learning_rate, momentum=config.momentum)
  checkpointer = tf.train.Checkpoint(
      optimizer=optimizer, model=model, optimizer_step=global_step)

  if FLAGS.use_defun:
    model.call = tfe.defun(model.call)

  if FLAGS.train_dir:
    summary_writer = tf.contrib.summary.create_file_writer(FLAGS.train_dir)
    if FLAGS.restore:
      latest_path = tf.train.latest_checkpoint(FLAGS.train_dir)
      checkpointer.restore(latest_path)
      print("Restored latest checkpoint at path:\"{}\" "
            "with global_step: {}".format(latest_path, global_step.numpy()))
      sys.stdout.flush()

  for x, y in ds_train:
    train_one_iter(model, x, y, optimizer, global_step=global_step)

    if global_step.numpy() % config.log_every == 0:
      it_test = ds_test.make_one_shot_iterator()
      acc_test, loss_test = evaluate(model, it_test)

      if FLAGS.validate:
        it_train = ds_train_one_shot.make_one_shot_iterator()
        it_validation = ds_validation.make_one_shot_iterator()
        acc_train, loss_train = evaluate(model, it_train)
        acc_validation, loss_validation = evaluate(model, it_validation)
        print("Iter {}, "
              "training set accuracy {:.4f}, loss {:.4f}; "
              "validation set accuracy {:.4f}, loss {:.4f}; "
              "test accuracy {:.4f}, loss {:.4f}".format(
                  global_step.numpy(), acc_train, loss_train, acc_validation,
                  loss_validation, acc_test, loss_test))
      else:
        print("Iter {}, test accuracy {:.4f}, loss {:.4f}".format(
            global_step.numpy(), acc_test, loss_test))
      sys.stdout.flush()

      if FLAGS.train_dir:
        with summary_writer.as_default():
          with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar("Test accuracy", acc_test)
            tf.contrib.summary.scalar("Test loss", loss_test)
            if FLAGS.validate:
              tf.contrib.summary.scalar("Training accuracy", acc_train)
              tf.contrib.summary.scalar("Training loss", loss_train)
              tf.contrib.summary.scalar("Validation accuracy", acc_validation)
              tf.contrib.summary.scalar("Validation loss", loss_validation)

    if global_step.numpy() % config.save_every == 0 and FLAGS.train_dir:
      saved_path = checkpointer.save(
          file_prefix=os.path.join(FLAGS.train_dir, "ckpt"))
      print("Saved checkpoint at path: \"{}\" "
            "with global_step: {}".format(saved_path, global_step.numpy()))
      sys.stdout.flush()


def get_config(config_name="revnet-38", dataset="cifar-10"):
  """Return configuration."""
  print("Config: {}".format(config_name))
  sys.stdout.flush()
  config = {
      "revnet-38": config_.get_hparams_cifar_38(),
      "revnet-110": config_.get_hparams_cifar_110(),
      "revnet-164": config_.get_hparams_cifar_164(),
  }[config_name]

  if dataset == "cifar-10":
    config.add_hparam("n_classes", 10)
    config.add_hparam("dataset", "cifar-10")
  else:
    config.add_hparam("n_classes", 100)
    config.add_hparam("dataset", "cifar-100")

  return config


def get_datasets(data_dir, config):
  """Return dataset."""
  if data_dir is None:
    raise ValueError("No supplied data directory")
  if not os.path.exists(data_dir):
    raise ValueError("Data directory {} does not exist".format(data_dir))
  if config.dataset not in ["cifar-10", "cifar-100"]:
    raise ValueError("Unknown dataset {}".format(config.dataset))

  print("Training on {} dataset.".format(config.dataset))
  sys.stdout.flush()
  data_dir = os.path.join(data_dir, config.dataset)
  if FLAGS.validate:
    # 40k Training set
    ds_train = cifar_input.get_ds_from_tfrecords(
        data_dir=data_dir,
        split="train",
        data_aug=True,
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=config.shuffle,
        data_format=config.data_format,
        dtype=config.dtype,
        prefetch=config.batch_size)
    # 10k Training set
    ds_validation = cifar_input.get_ds_from_tfrecords(
        data_dir=data_dir,
        split="validation",
        data_aug=False,
        batch_size=config.eval_batch_size,
        epochs=1,
        shuffle=False,
        data_format=config.data_format,
        dtype=config.dtype,
        prefetch=config.eval_batch_size)
  else:
    # 50k Training set
    ds_train = cifar_input.get_ds_from_tfrecords(
        data_dir=data_dir,
        split="train_all",
        data_aug=True,
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=config.shuffle,
        data_format=config.data_format,
        dtype=config.dtype,
        prefetch=config.batch_size)
    ds_validation = None

  # Always compute loss and accuracy on whole test set
  ds_train_one_shot = cifar_input.get_ds_from_tfrecords(
      data_dir=data_dir,
      split="train_all",
      data_aug=False,
      batch_size=config.eval_batch_size,
      epochs=1,
      shuffle=False,
      data_format=config.data_format,
      dtype=config.dtype,
      prefetch=config.eval_batch_size)

  ds_test = cifar_input.get_ds_from_tfrecords(
      data_dir=data_dir,
      split="test",
      data_aug=False,
      batch_size=config.eval_batch_size,
      epochs=1,
      shuffle=False,
      data_format=config.data_format,
      dtype=config.dtype,
      prefetch=config.eval_batch_size)

  return ds_train, ds_train_one_shot, ds_validation, ds_test


def train_one_iter(model, inputs, labels, optimizer, global_step=None):
  """Train for one iteration."""
  grads, vars_, logits, loss = model.compute_gradients(
      inputs, labels, training=True)
  optimizer.apply_gradients(zip(grads, vars_), global_step=global_step)

  return logits, loss


def evaluate(model, iterator):
  """Compute accuracy with the given dataset iterator."""
  mean_loss = tfe.metrics.Mean()
  accuracy = tfe.metrics.Accuracy()
  for x, y in iterator:
    logits, _ = model(x, training=False)
    loss = model.compute_loss(logits=logits, labels=y)
    accuracy(
        labels=tf.cast(y, tf.int64),
        predictions=tf.argmax(logits, axis=1, output_type=tf.int64))
    mean_loss(loss)

  return accuracy.result().numpy(), mean_loss.result().numpy()


if __name__ == "__main__":
  flags.DEFINE_string(
      "data_dir", default=None, help="Directory to load tfrecords")
  flags.DEFINE_string(
      "train_dir",
      default=None,
      help="[Optional] Directory to store the training information")
  flags.DEFINE_boolean(
      "restore",
      default=False,
      help="[Optional] Restore the latest checkpoint from `train_dir` if True")
  flags.DEFINE_boolean(
      "validate",
      default=False,
      help="[Optional] Use the validation set or not for hyperparameter search")
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
      "use_defun",
      default=False,
      help="[Optional] Use `tfe.defun` to boost performance.")
  FLAGS = flags.FLAGS
  tf.app.run(main)
