# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Trains the MNIST network using preloaded data stored in a variable.

Run using bazel:

bazel run -c opt \
    <...>/tensorflow/examples/how_tos/reading_data:fully_connected_preloaded_var

or, if installed via pip:

cd tensorflow/examples/how_tos/reading_data
python fully_connected_preloaded_var.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', '/tmp/data',
                    'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


def run_training():
  """Train MNIST for a number of epochs."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    with tf.name_scope('input'):
      # Input data
      images_initializer = tf.placeholder(
          dtype=data_sets.train.images.dtype,
          shape=data_sets.train.images.shape)
      labels_initializer = tf.placeholder(
          dtype=data_sets.train.labels.dtype,
          shape=data_sets.train.labels.shape)
      input_images = tf.Variable(
          images_initializer, trainable=False, collections=[])
      input_labels = tf.Variable(
          labels_initializer, trainable=False, collections=[])

      image, label = tf.train.slice_input_producer(
          [input_images, input_labels], num_epochs=FLAGS.num_epochs)
      label = tf.cast(label, tf.int32)
      images, labels = tf.train.batch(
          [image, label], batch_size=FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images, FLAGS.hidden1, FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create the op for initializing variables.
    init_op = tf.initialize_all_variables()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(init_op)
    sess.run(input_images.initializer,
             feed_dict={images_initializer: data_sets.train.images})
    sess.run(input_labels.initializer,
             feed_dict={labels_initializer: data_sets.train.labels})

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # And then after everything is built, start the training loop.
    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()

        # Run one step of the model.
        _, loss_value = sess.run([train_op, loss])

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
          # Print status to stdout.
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                     duration))
          # Update the events file.
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)
          step += 1

        # Save a checkpoint periodically.
        if (step + 1) % 1000 == 0:
          print('Saving')
          saver.save(sess, FLAGS.train_dir, global_step=step)

        step += 1
    except tf.errors.OutOfRangeError:
      print('Saving')
      saver.save(sess, FLAGS.train_dir, global_step=step)
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
