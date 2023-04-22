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
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def _next_power_of_two(x):
  """Calculates the smallest enclosing power of two for an input.

  Args:
    x: Positive float or integer number.

  Returns:
    Next largest power of two integer.
  """
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count,
                           preprocess):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    feature_bin_count: Number of frequency bins to use for analysis.
    preprocess: How the spectrogram is processed to produce features.

  Returns:
    Dictionary containing common settings.

  Raises:
    ValueError: If the preprocessing mode isn't recognized.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if preprocess == 'average':
    fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
    average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
    fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
  elif preprocess == 'mfcc':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  elif preprocess == 'micro':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                     ' "average", or "micro")' % (preprocess))
  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                         is_training, runtime_settings)
  elif model_architecture == 'tiny_conv':
    return create_tiny_conv_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'tiny_embedding_conv':
    return create_tiny_embedding_conv_model(fingerprint_input, model_settings,
                                            is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, "low_latency_svdf",' +
                    ' "tiny_conv", or "tiny_embedding_conv"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
  saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):
  """Builds a model with a single hidden fully-connected layer.

  This is a very simple model with just one matmul and bias layer. As you'd
  expect, it doesn't produce very accurate results, but it is very fast and
  simple, so it's useful for sanity testing.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.compat.v1.get_variable(
      name='weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001),
      shape=[fingerprint_size, label_count])
  bias = tf.compat.v1.get_variable(name='bias',
                                   initializer=tf.compat.v1.zeros_initializer,
                                   shape=[label_count])
  logits = tf.matmul(fingerprint_input, weights) + bias
  if is_training:
    return logits, dropout_rate
  else:
    return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.compat.v1.get_variable(
      name='first_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.compat.v1.get_variable(
      name='first_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[first_filter_count])

  first_conv = tf.nn.conv2d(input=fingerprint_4d,
                            filters=first_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool2d(input=first_dropout,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.compat.v1.get_variable(
      name='second_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[
          second_filter_height, second_filter_width, first_filter_count,
          second_filter_count
      ])
  second_bias = tf.compat.v1.get_variable(
      name='second_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[second_filter_count])
  second_conv = tf.nn.conv2d(input=max_pool,
                             filters=second_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, rate=dropout_rate)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.compat.v1.get_variable(
      name='final_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[second_conv_element_count, label_count])
  final_fc_bias = tf.compat.v1.get_variable(
      name='final_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_rate
  else:
    return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
  """Builds a convolutional model with low compute requirements.

  This is roughly the network labeled as 'cnn-one-fstride4' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces slightly lower quality results than the 'conv' model, but needs
  fewer weight parameters and computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 1
  first_weights = tf.compat.v1.get_variable(
      name='first_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.compat.v1.get_variable(
      name='first_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[first_filter_count])
  first_conv = tf.nn.conv2d(
      input=fingerprint_4d,
      filters=first_weights,
      strides=[1, first_filter_stride_y, first_filter_stride_x, 1],
      padding='VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)
  else:
    first_dropout = first_relu
  first_conv_output_width = math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x)
  first_conv_output_height = math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y)
  first_conv_element_count = int(
      first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,
                                    [-1, first_conv_element_count])
  first_fc_output_channels = 128
  first_fc_weights = tf.compat.v1.get_variable(
      name='first_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[first_conv_element_count, first_fc_output_channels])
  first_fc_bias = tf.compat.v1.get_variable(
      name='first_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[first_fc_output_channels])
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, rate=dropout_rate)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 128
  second_fc_weights = tf.compat.v1.get_variable(
      name='second_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[first_fc_output_channels, second_fc_output_channels])
  second_fc_bias = tf.compat.v1.get_variable(
      name='second_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[second_fc_output_channels])
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, rate=dropout_rate)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.compat.v1.get_variable(
      name='final_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[second_fc_output_channels, label_count])
  final_fc_bias = tf.compat.v1.get_variable(
      name='final_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_rate
  else:
    return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
  """Builds an SVDF model with low compute requirements.

  This is based in the topology presented in the 'Compressing Deep Neural
  Networks using a Rank-Constrained Topology' paper:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
        [SVDF]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This model produces lower recognition accuracy than the 'conv' model above,
  but requires fewer weight parameters and, significantly fewer computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    The node is expected to produce a 2D Tensor of shape:
      [batch, model_settings['fingerprint_width'] *
              model_settings['spectrogram_length']]
    with the features corresponding to the same time slot arranged contiguously,
    and the oldest slot at index [:, 0], and newest at [:, -1].
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
      ValueError: If the inputs tensor is incorrectly shaped.
  """
  if is_training:
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')

  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']

  # Validation.
  input_shape = fingerprint_input.get_shape()
  if len(input_shape) != 2:
    raise ValueError('Inputs to `SVDF` should have rank == 2.')
  if input_shape[-1].value is None:
    raise ValueError('The last dimension of the input to `SVDF` '
                     'should be defined. Found `None`.')
  if input_shape[-1].value % input_frequency_size != 0:
    raise ValueError('The last dimension of the input to `SVDF` = {0} must be '
                     'a multiple of the frame size = {1}'.format(
                         input_shape.shape[-1].value, input_frequency_size))

  # Set number of units (i.e. nodes) and rank.
  rank = 2
  num_units = 1280
  # Number of filters: pairs of feature and time filters.
  num_filters = rank * num_units
  # Create the runtime memory: [num_filters, batch, input_time_size]
  batch = 1
  memory = tf.compat.v1.get_variable(
      initializer=tf.compat.v1.zeros_initializer,
      shape=[num_filters, batch, input_time_size],
      trainable=False,
      name='runtime-memory')
  first_time_flag = tf.compat.v1.get_variable(
      name='first_time_flag', dtype=tf.int32, initializer=1)
  # Determine the number of new frames in the input, such that we only operate
  # on those. For training we do not use the memory, and thus use all frames
  # provided in the input.
  # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
  if is_training:
    num_new_frames = input_time_size
  else:
    window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                           model_settings['sample_rate'])
    num_new_frames = tf.cond(
        pred=tf.equal(first_time_flag, 1),
        true_fn=lambda: input_time_size,
        false_fn=lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))  # pylint:disable=line-too-long
  first_time_flag = 0
  new_fingerprint_input = fingerprint_input[
      :, -num_new_frames*input_frequency_size:]
  # Expand to add input channels dimension.
  new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

  # Create the frequency filters.
  weights_frequency = tf.compat.v1.get_variable(
      name='weights_frequency',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[input_frequency_size, num_filters])
  # Expand to add input channels dimensions.
  # weights_frequency: [input_frequency_size, 1, num_filters]
  weights_frequency = tf.expand_dims(weights_frequency, 1)
  # Convolve the 1D feature filters sliding over the time dimension.
  # activations_time: [batch, num_new_frames, num_filters]
  activations_time = tf.nn.conv1d(input=new_fingerprint_input,
                                  filters=weights_frequency,
                                  stride=input_frequency_size,
                                  padding='VALID')
  # Rearrange such that we can perform the batched matmul.
  # activations_time: [num_filters, batch, num_new_frames]
  activations_time = tf.transpose(a=activations_time, perm=[2, 0, 1])

  # Runtime memory optimization.
  if not is_training:
    # We need to drop the activations corresponding to the oldest frames, and
    # then add those corresponding to the new frames.
    new_memory = memory[:, :, num_new_frames:]
    new_memory = tf.concat([new_memory, activations_time], 2)
    tf.compat.v1.assign(memory, new_memory)
    activations_time = new_memory

  # Create the time filters.
  weights_time = tf.compat.v1.get_variable(
      name='weights_time',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[num_filters, input_time_size])
  # Apply the time filter on the outputs of the feature filters.
  # weights_time: [num_filters, input_time_size, 1]
  # outputs: [num_filters, batch, 1]
  weights_time = tf.expand_dims(weights_time, 2)
  outputs = tf.matmul(activations_time, weights_time)
  # Split num_units and rank into separate dimensions (the remaining
  # dimension is the input_shape[0] -i.e. batch size). This also squeezes
  # the last dimension, since it's not used.
  # [num_filters, batch, 1] => [num_units, rank, batch]
  outputs = tf.reshape(outputs, [num_units, rank, -1])
  # Sum the rank outputs per unit => [num_units, batch].
  units_output = tf.reduce_sum(input_tensor=outputs, axis=1)
  # Transpose to shape [batch, num_units]
  units_output = tf.transpose(a=units_output)

  # Appy bias.
  bias = tf.compat.v1.get_variable(name='bias',
                                   initializer=tf.compat.v1.zeros_initializer,
                                   shape=[num_units])
  first_bias = tf.nn.bias_add(units_output, bias)

  # Relu.
  first_relu = tf.nn.relu(first_bias)

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)
  else:
    first_dropout = first_relu

  first_fc_output_channels = 256
  first_fc_weights = tf.compat.v1.get_variable(
      name='first_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[num_units, first_fc_output_channels])
  first_fc_bias = tf.compat.v1.get_variable(
      name='first_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[first_fc_output_channels])
  first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, rate=dropout_rate)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 256
  second_fc_weights = tf.compat.v1.get_variable(
      name='second_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[first_fc_output_channels, second_fc_output_channels])
  second_fc_bias = tf.compat.v1.get_variable(
      name='second_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[second_fc_output_channels])
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, rate=dropout_rate)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.compat.v1.get_variable(
      name='final_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[second_fc_output_channels, label_count])
  final_fc_bias = tf.compat.v1.get_variable(
      name='final_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_rate
  else:
    return final_fc


def create_tiny_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a convolutional model aimed at microcontrollers.

  Devices like DSPs and microcontrollers can have very small amounts of
  memory and limited processing power. This model is designed to use less
  than 20KB of working RAM, and fit within 32KB of read-only (flash) memory.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This doesn't produce particularly accurate results, but it's designed to be
  used as the first stage of a pipeline, running on a low-energy piece of
  hardware that can always be on, and then wake higher-power chips when a
  possible utterance has been found, so that more accurate analysis can be done.

  During training, a dropout node is introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 10
  first_filter_count = 8
  first_weights = tf.compat.v1.get_variable(
      name='first_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.compat.v1.get_variable(
      name='first_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[first_filter_count])
  first_conv_stride_x = 2
  first_conv_stride_y = 2
  first_conv = tf.nn.conv2d(
      input=fingerprint_4d, filters=first_weights,
      strides=[1, first_conv_stride_y, first_conv_stride_x, 1],
      padding='SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)
  else:
    first_dropout = first_relu
  first_dropout_shape = first_dropout.get_shape()
  first_dropout_output_width = first_dropout_shape[2]
  first_dropout_output_height = first_dropout_shape[1]
  first_dropout_element_count = int(
      first_dropout_output_width * first_dropout_output_height *
      first_filter_count)
  flattened_first_dropout = tf.reshape(first_dropout,
                                       [-1, first_dropout_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.compat.v1.get_variable(
      name='final_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[first_dropout_element_count, label_count])
  final_fc_bias = tf.compat.v1.get_variable(
      name='final_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[label_count])
  final_fc = (
      tf.matmul(flattened_first_dropout, final_fc_weights) + final_fc_bias)
  if is_training:
    return final_fc, dropout_rate
  else:
    return final_fc


def create_tiny_embedding_conv_model(fingerprint_input, model_settings,
                                     is_training):
  """Builds a convolutional model aimed at microcontrollers.

  Devices like DSPs and microcontrollers can have very small amounts of
  memory and limited processing power. This model is designed to use less
  than 20KB of working RAM, and fit within 32KB of read-only (flash) memory.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This doesn't produce particularly accurate results, but it's designed to be
  used as the first stage of a pipeline, running on a low-energy piece of
  hardware that can always be on, and then wake higher-power chips when a
  possible utterance has been found, so that more accurate analysis can be done.

  During training, a dropout node is introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])

  first_filter_width = 8
  first_filter_height = 10
  first_filter_count = 8
  first_weights = tf.compat.v1.get_variable(
      name='first_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.compat.v1.get_variable(
      name='first_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[first_filter_count])
  first_conv_stride_x = 2
  first_conv_stride_y = 2

  first_conv = tf.nn.conv2d(
      input=fingerprint_4d, filters=first_weights,
      strides=[1, first_conv_stride_y, first_conv_stride_x, 1],
      padding='SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)

  else:
    first_dropout = first_relu

  second_filter_width = 8
  second_filter_height = 10
  second_filter_count = 8
  second_weights = tf.compat.v1.get_variable(
      name='second_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[
          second_filter_height, second_filter_width, first_filter_count,
          second_filter_count
      ])
  second_bias = tf.compat.v1.get_variable(
      name='second_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[second_filter_count])
  second_conv_stride_x = 8
  second_conv_stride_y = 8
  second_conv = tf.nn.conv2d(
      input=first_dropout, filters=second_weights,
      strides=[1, second_conv_stride_y, second_conv_stride_x, 1],
      padding='SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, rate=dropout_rate)
  else:
    second_dropout = second_relu

  second_dropout_shape = second_dropout.get_shape()
  second_dropout_output_width = second_dropout_shape[2]
  second_dropout_output_height = second_dropout_shape[1]
  second_dropout_element_count = int(second_dropout_output_width *
                                     second_dropout_output_height *
                                     second_filter_count)
  flattened_second_dropout = tf.reshape(second_dropout,
                                        [-1, second_dropout_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.compat.v1.get_variable(
      name='final_fc_weights',
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
      shape=[second_dropout_element_count, label_count])
  final_fc_bias = tf.compat.v1.get_variable(
      name='final_fc_bias',
      initializer=tf.compat.v1.zeros_initializer,
      shape=[label_count])
  final_fc = (
      tf.matmul(flattened_second_dropout, final_fc_weights) + final_fc_bias)
  if is_training:
    return final_fc, dropout_rate
  else:
    return final_fc
