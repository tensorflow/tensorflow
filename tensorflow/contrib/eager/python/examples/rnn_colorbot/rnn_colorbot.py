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
r"""TensorFlow Eager Execution Example: RNN Colorbot.

This example builds, trains, and evaluates a multi-layer RNN that can be
run with eager execution enabled. The RNN is trained to map color names to
their RGB values: it takes as input a one-hot encoded character sequence and
outputs a three-tuple (R, G, B) (scaled by 1/255).

For example, say we'd like the RNN Colorbot to generate the RGB values for the
color white. To represent our query in a form that the Colorbot could
understand, we would create a sequence of five 256-long vectors encoding the
ASCII values of the characters in "white". The first vector in our sequence
would be 0 everywhere except for the ord("w")-th position, where it would be
1, the second vector would be 0 everywhere except for the
ord("h")-th position, where it would be 1, and similarly for the remaining three
vectors. We refer to such indicator vectors as "one-hot encodings" of
characters. After consuming these vectors, a well-trained Colorbot would output
the three tuple (1, 1, 1), since the RGB values for white are (255, 255, 255).
We are of course free to ask the colorbot to generate colors for any string we'd
like, such as "steel gray," "tensorflow orange," or "green apple," though
your mileage may vary as your queries increase in creativity.

This example shows how to:
  1. read, process, (one-hot) encode, and pad text data via the
     Datasets API;
  2. build a trainable model;
  3. implement a multi-layer RNN using Python control flow
     constructs (e.g., a for loop);
  4. train a model using an iterative gradient-based method; and

The data used in this example is licensed under the Creative Commons
Attribution-ShareAlike License and is available at
  https://en.wikipedia.org/wiki/List_of_colors:_A-F
  https://en.wikipedia.org/wiki/List_of_colors:_G-M
  https://en.wikipedia.org/wiki/List_of_colors:_N-Z

This example was adapted from
  https://github.com/random-forests/tensorflow-workshop/tree/master/extras/colorbot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
import sys
import time

import six
import tensorflow as tf

from tensorflow.contrib.eager.python import tfe

try:
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  HAS_MATPLOTLIB = True
except ImportError:
  HAS_MATPLOTLIB = False


def parse(line):
  """Parse a line from the colors dataset."""

  # Each line of the dataset is comma-separated and formatted as
  #    color_name, r, g, b
  # so `items` is a list [color_name, r, g, b].
  items = tf.string_split([line], ",").values
  rgb = tf.string_to_number(items[1:], out_type=tf.float32) / 255.
  # Represent the color name as a one-hot encoded character sequence.
  color_name = items[0]
  chars = tf.one_hot(tf.decode_raw(color_name, tf.uint8), depth=256)
  # The sequence length is needed by our RNN.
  length = tf.cast(tf.shape(chars)[0], dtype=tf.int64)
  return rgb, chars, length


def load_dataset(data_dir, url, batch_size):
  """Loads the colors data at path into a PaddedDataset."""

  # Downloads data at url into data_dir/basename(url). The dataset has a header
  # row (color_name, r, g, b) followed by comma-separated lines.
  path = tf.contrib.learn.datasets.base.maybe_download(
      os.path.basename(url), data_dir, url)

  # This chain of commands loads our data by:
  #   1. skipping the header; (.skip(1))
  #   2. parsing the subsequent lines; (.map(parse))
  #   3. shuffling the data; (.shuffle(...))
  #   3. grouping the data into padded batches (.padded_batch(...)).
  dataset = tf.data.TextLineDataset(path).skip(1).map(parse).shuffle(
      buffer_size=10000).padded_batch(
          batch_size, padded_shapes=([None], [None, None], []))
  return dataset


# pylint: disable=not-callable
class RNNColorbot(tfe.Network):
  """Multi-layer (LSTM) RNN that regresses on real-valued vector labels.
  """

  def __init__(self, rnn_cell_sizes, label_dimension, keep_prob):
    """Constructs an RNNColorbot.

    Args:
      rnn_cell_sizes: list of integers denoting the size of each LSTM cell in
        the RNN; rnn_cell_sizes[i] is the size of the i-th layer cell
      label_dimension: the length of the labels on which to regress
      keep_prob: (1 - dropout probability); dropout is applied to the outputs of
        each LSTM layer
    """
    super(RNNColorbot, self).__init__(name="")
    self.label_dimension = label_dimension
    self.keep_prob = keep_prob

    # Note the calls to `track_layer` below; these calls register the layers as
    # network components that house trainable variables.
    self.cells = [
        self.track_layer(tf.nn.rnn_cell.BasicLSTMCell(size))
        for size in rnn_cell_sizes
    ]
    self.relu = self.track_layer(
        tf.layers.Dense(label_dimension, activation=tf.nn.relu, name="relu"))

  def call(self, chars, sequence_length, training=False):
    """Implements the RNN logic and prediction generation.

    Args:
      chars: a Tensor of dimension [batch_size, time_steps, 256] holding a
        batch of one-hot encoded color names
      sequence_length: a Tensor of dimension [batch_size] holding the length
        of each character sequence (i.e., color name)
      training: whether the invocation is happening during training

    Returns:
      A tensor of dimension [batch_size, label_dimension] that is produced by
      passing chars through a multi-layer RNN and applying a ReLU to the final
      hidden state.
    """
    # Transpose the first and second dimensions so that chars is of shape
    # [time_steps, batch_size, dimension].
    chars = tf.transpose(chars, [1, 0, 2])
    # The outer loop cycles through the layers of the RNN; the inner loop
    # executes the time steps for a particular layer.
    batch_size = int(chars.shape[1])
    for l in range(len(self.cells)):
      cell = self.cells[l]
      outputs = []
      state = cell.zero_state(batch_size, tf.float32)
      # Unstack the inputs to obtain a list of batches, one for each time step.
      chars = tf.unstack(chars, axis=0)
      for ch in chars:
        output, state = cell(ch, state)
        outputs.append(output)
      # The outputs of this layer are the inputs of the subsequent layer.
      chars = tf.stack(outputs, axis=0)
      if training:
        chars = tf.nn.dropout(chars, self.keep_prob)
    # Extract the correct output (i.e., hidden state) for each example. All the
    # character sequences in this batch were padded to the same fixed length so
    # that they could be easily fed through the above RNN loop. The
    # `sequence_length` vector tells us the true lengths of the character
    # sequences, letting us obtain for each sequence the hidden state that was
    # generated by its non-padding characters.
    batch_range = [i for i in range(batch_size)]
    indices = tf.stack([sequence_length - 1, batch_range], axis=1)
    hidden_states = tf.gather_nd(chars, indices)
    return self.relu(hidden_states)


def loss(labels, predictions):
  """Computes mean squared loss."""
  return tf.reduce_mean(tf.square(predictions - labels))


def test(model, eval_data):
  """Computes the average loss on eval_data, which should be a Dataset."""
  avg_loss = tfe.metrics.Mean("loss")
  for (labels, chars, sequence_length) in tfe.Iterator(eval_data):
    predictions = model(chars, sequence_length, training=False)
    avg_loss(loss(labels, predictions))
  print("eval/loss: %.6f\n" % avg_loss.result())
  with tf.contrib.summary.always_record_summaries():
    tf.contrib.summary.scalar("loss", avg_loss.result())


def train_one_epoch(model, optimizer, train_data, log_interval=10):
  """Trains model on train_data using optimizer."""

  tf.train.get_or_create_global_step()

  def model_loss(labels, chars, sequence_length):
    predictions = model(chars, sequence_length, training=True)
    loss_value = loss(labels, predictions)
    tf.contrib.summary.scalar("loss", loss_value)
    return loss_value

  for (batch, (labels, chars, sequence_length)) in enumerate(
      tfe.Iterator(train_data)):
    with tf.contrib.summary.record_summaries_every_n_global_steps(log_interval):
      batch_model_loss = functools.partial(model_loss, labels, chars,
                                           sequence_length)
      optimizer.minimize(
          batch_model_loss, global_step=tf.train.get_global_step())
      if log_interval and batch % log_interval == 0:
        print("train/batch #%d\tloss: %.6f" % (batch, batch_model_loss()))


SOURCE_TRAIN_URL = "https://raw.githubusercontent.com/random-forests/tensorflow-workshop/master/extras/colorbot/data/train.csv"
SOURCE_TEST_URL = "https://raw.githubusercontent.com/random-forests/tensorflow-workshop/master/extras/colorbot/data/test.csv"


def main(_):
  data_dir = os.path.join(FLAGS.dir, "data")
  train_data = load_dataset(
      data_dir=data_dir, url=SOURCE_TRAIN_URL, batch_size=FLAGS.batch_size)
  eval_data = load_dataset(
      data_dir=data_dir, url=SOURCE_TEST_URL, batch_size=FLAGS.batch_size)

  model = RNNColorbot(
      rnn_cell_sizes=FLAGS.rnn_cell_sizes,
      label_dimension=3,
      keep_prob=FLAGS.keep_probability)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

  if FLAGS.no_gpu or tfe.num_gpus() <= 0:
    print(tfe.num_gpus())
    device = "/cpu:0"
  else:
    device = "/gpu:0"
  print("Using device %s." % device)

  log_dir = os.path.join(FLAGS.dir, "summaries")
  tf.gfile.MakeDirs(log_dir)
  train_summary_writer = tf.contrib.summary.create_file_writer(
      os.path.join(log_dir, "train"), flush_millis=10000)
  test_summary_writer = tf.contrib.summary.create_file_writer(
      os.path.join(log_dir, "eval"), flush_millis=10000, name="eval")

  with tf.device(device):
    for epoch in range(FLAGS.num_epochs):
      start = time.time()
      with train_summary_writer.as_default():
        train_one_epoch(model, optimizer, train_data, FLAGS.log_interval)
      end = time.time()
      print("train/time for epoch #%d: %.2f" % (epoch, end - start))
      with test_summary_writer.as_default():
        test(model, eval_data)

  print("Colorbot is ready to generate colors!")
  while True:
    try:
      color_name = six.moves.input(
          "Give me a color name (or press enter to exit): ")
    except EOFError:
      return

    if not color_name:
      return

    _, chars, length = parse(color_name)
    with tf.device(device):
      (chars, length) = (tf.identity(chars), tf.identity(length))
      chars = tf.expand_dims(chars, 0)
      length = tf.expand_dims(length, 0)
      preds = tf.unstack(model(chars, length, training=False)[0])

    # Predictions cannot be negative, as they are generated by a ReLU layer;
    # they may, however, be greater than 1.
    clipped_preds = tuple(min(float(p), 1.0) for p in preds)
    rgb = tuple(int(p * 255) for p in clipped_preds)
    print("rgb:", rgb)
    data = [[clipped_preds]]
    if HAS_MATPLOTLIB:
      plt.imshow(data)
      plt.title(color_name)
      plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--dir",
      type=str,
      default="/tmp/rnn_colorbot/",
      help="Directory to download data files and save logs.")
  parser.add_argument(
      "--log_interval",
      type=int,
      default=10,
      metavar="N",
      help="Log training loss every log_interval batches.")
  parser.add_argument(
      "--num_epochs", type=int, default=20, help="Number of epochs to train.")
  parser.add_argument(
      "--rnn_cell_sizes",
      type=int,
      nargs="+",
      default=[256, 128],
      help="List of sizes for each layer of the RNN.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=64,
      help="Batch size for training and eval.")
  parser.add_argument(
      "--keep_probability",
      type=float,
      default=0.5,
      help="Keep probability for dropout between layers.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.01,
      help="Learning rate to be used during training.")
  parser.add_argument(
      "--no_gpu",
      action="store_true",
      default=False,
      help="Disables GPU usage even if a GPU is available.")

  FLAGS, unparsed = parser.parse_known_args()
  tfe.run(main=main, argv=[sys.argv[0]] + unparsed)
