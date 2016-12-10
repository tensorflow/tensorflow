# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A small library of functions dealing with LSTMs applied to images.

Tensors in this library generally have the shape (num_images, height, width,
depth).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib.ndlstm.python import lstm1d


def _shape(tensor):
  """Get the shape of a tensor as an int list."""
  return tensor.get_shape().as_list()


def images_to_sequence(tensor):
  """Convert a batch of images into a batch of sequences.

  Args:
    tensor: a (num_images, height, width, depth) tensor

  Returns:
    (width, num_images*height, depth) sequence tensor
  """

  num_image_batches, height, width, depth = _shape(tensor)
  transposed = tf.transpose(tensor, [2, 0, 1, 3])
  return tf.reshape(transposed, [width, num_image_batches * height, depth])


def sequence_to_images(tensor, num_image_batches):
  """Convert a batch of sequences into a batch of images.

  Args:
    tensor: (num_steps, num_batches, depth) sequence tensor
    num_image_batches: the number of image batches

  Returns:
    (num_images, height, width, depth) tensor
  """

  width, num_batches, depth = _shape(tensor)
  height = num_batches // num_image_batches
  reshaped = tf.reshape(tensor, [width, num_image_batches, height, depth])
  return tf.transpose(reshaped, [1, 2, 0, 3])


def horizontal_lstm(images, num_filters_out, scope=None):
  """Run an LSTM bidirectionally over all the rows of each image.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output depth
    scope: optional scope name

  Returns:
    (num_images, height, width, num_filters_out) tensor, where
    num_steps is width and new num_batches is num_image_batches * height
  """
  with tf.variable_scope(scope, "HorizontalLstm", [images]):
    batch_size, _, _, _ = _shape(images)
    sequence = images_to_sequence(images)
    with tf.variable_scope("lr"):
      hidden_sequence_lr = lstm1d.ndlstm_base(sequence, num_filters_out // 2)
    with tf.variable_scope("rl"):
      hidden_sequence_rl = (
          lstm1d.ndlstm_base(sequence,
                             num_filters_out - num_filters_out // 2,
                             reverse=1))
    output_sequence = tf.concat_v2([hidden_sequence_lr, hidden_sequence_rl], 2)
    output = sequence_to_images(output_sequence, batch_size)
    return output


def separable_lstm(images, num_filters_out, nhidden=None, scope=None):
  """Run bidirectional LSTMs first horizontally then vertically.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output layer depth
    nhidden: hidden layer depth
    scope: optional scope name

  Returns:
    (num_images, height, width, num_filters_out) tensor
  """
  with tf.variable_scope(scope, "SeparableLstm", [images]):
    if nhidden is None:
      nhidden = num_filters_out
    hidden = horizontal_lstm(images, nhidden)
    with tf.variable_scope("vertical"):
      transposed = tf.transpose(hidden, [0, 2, 1, 3])
      output_transposed = horizontal_lstm(transposed, num_filters_out)
    output = tf.transpose(output_transposed, [0, 2, 1, 3])
    return output


def reduce_to_sequence(images, num_filters_out, scope=None):
  """Reduce an image to a sequence by scanning an LSTM vertically.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output layer depth
    scope: optional scope name

  Returns:
    A (width, num_images, num_filters_out) sequence.
  """
  with tf.variable_scope(scope, "ReduceToSequence", [images]):
    batch_size, height, width, depth = _shape(images)
    transposed = tf.transpose(images, [1, 0, 2, 3])
    reshaped = tf.reshape(transposed, [height, batch_size * width, depth])
    reduced = lstm1d.sequence_to_final(reshaped, num_filters_out)
    output = tf.reshape(reduced, [batch_size, width, num_filters_out])
    return output


def reduce_to_final(images, num_filters_out, nhidden=None, scope=None):
  """Reduce an image to a final state by running two LSTMs.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output layer depth
    nhidden: hidden layer depth (defaults to num_filters_out)
    scope: optional scope name

  Returns:
    A (num_images, num_filters_out) batch.
  """
  with tf.variable_scope(scope, "ReduceToFinal", [images]):
    nhidden = nhidden or num_filters_out
    batch_size, height, width, depth = _shape(images)
    transposed = tf.transpose(images, [1, 0, 2, 3])
    reshaped = tf.reshape(transposed, [height, batch_size * width, depth])
    with tf.variable_scope("reduce1"):
      reduced = lstm1d.sequence_to_final(reshaped, nhidden)
      transposed_hidden = tf.reshape(reduced, [batch_size, width, nhidden])
      hidden = tf.transpose(transposed_hidden, [1, 0, 2])
    with tf.variable_scope("reduce2"):
      output = lstm1d.sequence_to_final(hidden, num_filters_out)
    return output
