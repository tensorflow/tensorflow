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
"""MNIST model training with TensorFlow eager execution.

See:
https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html

This program demonstrates training, export, and inference of a convolutional
neural network model with eager execution enabled.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

tfe = tf.contrib.eager

flags.DEFINE_integer(
    name='log_interval',
    default=10,
    help='batches between logging training status')

flags.DEFINE_float(name='learning_rate', default=0.01, help='Learning rate.')

flags.DEFINE_float(
    name='momentum', short_name='m', default=0.5, help='SGD momentum.')

flags.DEFINE_integer(
    name='batch_size',
    default=100,
    help='Batch size to use during training / eval')

flags.DEFINE_integer(
    name='train_epochs', default=10, help='Number of epochs to train')

flags.DEFINE_string(
    name='model_dir',
    default='/tmp/tensorflow/mnist',
    help='Where to save checkpoints, tensorboard summaries, etc.')

flags.DEFINE_bool(
    name='clean',
    default=False,
    help='Whether to clear model directory before training')

FLAGS = flags.FLAGS


def create_model():
  """Model to recognize digits in the MNIST dataset.

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
  But uses the tf.keras API.
  Returns:
    A tf.keras.Model.
  """
  # Assumes data_format == 'channel_last'.
  # See https://www.tensorflow.org/performance/performance_guide#data_formats

  input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D((2, 2), (2, 2), padding='same')
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  model = tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(2, 5, padding='same', activation=tf.nn.relu),
          max_pool,
          l.Conv2D(4, 5, padding='same', activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(32, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])
  # TODO(brianklee): Remove when @kaftan makes this happen by default.
  # TODO(brianklee): remove `autograph=True` when kwarg default is flipped.
  model.call = tfe.function(model.call, autograph=True)
  # Needs to have input_signature specified in order to be exported
  # since model.predict() is never called before saved_model.export()
  # TODO(brianklee): Update with input signature, depending on how the impl of
  # saved_model.restore() pans out.
  model.predict = tfe.function(model.predict, autograph=True)
  # ,input_signature=(tensor_spec.TensorSpec(shape=[28, 28, None], dtype=tf.float32),) # pylint: disable=line-too-long
  return model


def mnist_datasets():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # Numpy defaults to dtype=float64; TF defaults to float32. Stick with float32.
  x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
  y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  return train_dataset, test_dataset


def loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
  labels = tf.cast(labels, tf.int64)
  return tf.reduce_mean(
      tf.cast(tf.equal(predictions, labels), dtype=tf.float32))


# TODO(brianklee): Enable @tf.function on the training loop when zip, enumerate
# are supported by autograph.
def train(model, optimizer, dataset, step_counter, log_interval=None,
          num_steps=None):
  """Trains model on `dataset` using `optimizer`."""
  start = time.time()
  for (batch, (images, labels)) in enumerate(dataset):
    if num_steps is not None and batch > num_steps:
      break
    with tf.contrib.summary.record_summaries_every_n_global_steps(
        10, global_step=step_counter):
      # Record the operations used to compute the loss given the input,
      # so that the gradient of the loss with respect to the variables
      # can be computed.
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss(logits, labels)
        tf.contrib.summary.scalar('loss', loss_value)
        tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))
      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(
          zip(grads, model.variables), global_step=step_counter)
      if log_interval and batch % log_interval == 0:
        rate = log_interval / (time.time() - start)
        print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
        start = time.time()


def test(model, dataset):
  """Perform an evaluation of `model` on the examples from `dataset`."""
  avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
  accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

  for (images, labels) in dataset:
    logits = model(images, training=False)
    avg_loss(loss(logits, labels))
    accuracy(
        tf.argmax(logits, axis=1, output_type=tf.int64),
        tf.cast(labels, tf.int64))
  print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
        (avg_loss.result(), 100 * accuracy.result()))
  with tf.contrib.summary.always_record_summaries():
    tf.contrib.summary.scalar('loss', avg_loss.result())
    tf.contrib.summary.scalar('accuracy', accuracy.result())


def train_and_export(flags_obj):
  """Run MNIST training and eval loop in eager mode.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  # Load the datasets
  train_ds, test_ds = mnist_datasets()
  train_ds = train_ds.shuffle(60000).batch(flags_obj.batch_size)
  test_ds = test_ds.batch(flags_obj.batch_size)

  # Create the model and optimizer
  model = create_model()
  optimizer = tf.train.MomentumOptimizer(
      flags_obj.learning_rate, flags_obj.momentum)

  # See summaries with `tensorboard --logdir=<model_dir>`
  train_dir = os.path.join(flags_obj.model_dir, 'summaries', 'train')
  test_dir = os.path.join(flags_obj.model_dir, 'summaries', 'eval')
  summary_writer = tf.contrib.summary.create_file_writer(
      train_dir, flush_millis=10000)
  test_summary_writer = tf.contrib.summary.create_file_writer(
      test_dir, flush_millis=10000, name='test')

  # Create and restore checkpoint (if one exists on the path)
  checkpoint_dir = os.path.join(flags_obj.model_dir, 'checkpoints')
  checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
  step_counter = tf.train.get_or_create_global_step()
  checkpoint = tf.train.Checkpoint(
      model=model, optimizer=optimizer, step_counter=step_counter)
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  # Train and evaluate for a set number of epochs.
  for _ in range(flags_obj.train_epochs):
    start = time.time()
    with summary_writer.as_default():
      train(model, optimizer, train_ds, step_counter,
            flags_obj.log_interval, num_steps=1)
    end = time.time()
    print('\nTrain time for epoch #%d (%d total steps): %f' %
          (checkpoint.save_counter.numpy() + 1,
           step_counter.numpy(),
           end - start))
    with test_summary_writer.as_default():
      test(model, test_ds)
    checkpoint.save(checkpoint_prefix)

  # TODO(brianklee): Enable this functionality after @allenl implements this.
  # export_path = os.path.join(flags_obj.model_dir, 'export')
  # tf.saved_model.save(export_path, model)


def import_and_eval(flags_obj):
  export_path = os.path.join(flags_obj.model_dir, 'export')
  model = tf.saved_model.restore(export_path)
  _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_test = x_test / np.float32(255)
  y_predict = model(x_test)
  accuracy = compute_accuracy(y_predict, y_test)
  print('Model accuracy: {:0.2f}%'.format(accuracy.numpy() * 100))


def apply_clean(flags_obj):
  if flags_obj.clean and tf.gfile.Exists(flags_obj.model_dir):
    tf.logging.info('--clean flag set. Removing existing model dir: {}'.format(
        flags_obj.model_dir))
    tf.gfile.DeleteRecursively(flags_obj.model_dir)


def main(_):
  apply_clean(flags.FLAGS)
  train_and_export(flags.FLAGS)
  # TODO(brianklee): Enable this functionality after @allenl implements this.
  # import_and_eval(flags.FLAGS)


if __name__ == '__main__':
  app.run(main)
