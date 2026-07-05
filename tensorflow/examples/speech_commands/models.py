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
"""Model definitions for simple speech recognition."""

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


def create_model(model_settings, model_architecture):
  """Builds a tf.keras model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. The returned model takes a batch of 1D
  features that describe the audio (of size
  model_settings['fingerprint_size']) and outputs logits over
  model_settings['label_count'] classes.

  Typically the input will be derived from a spectrogram that's been run
  through an MFCC, but in theory it can be any feature vector of the size
  specified in model_settings['fingerprint_size'].

  See the implementations below for the possible model architectures that can
  be requested.

  Args:
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.

  Returns:
    A tf.keras.Model (or Sequential) that outputs logits.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'single_fc':
    return create_single_fc_model(model_settings)
  elif model_architecture == 'conv':
    return create_conv_model(model_settings)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(model_settings)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(model_settings)
  elif model_architecture == 'tiny_conv':
    return create_tiny_conv_model(model_settings)
  elif model_architecture == 'tiny_embedding_conv':
    return create_tiny_embedding_conv_model(model_settings)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, "low_latency_svdf",' +
                    ' "tiny_conv", or "tiny_embedding_conv"')


def create_single_fc_model(model_settings):
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
    model_settings: Dictionary of information about the model.

  Returns:
    tf.keras.Model outputting logits.
  """
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  return tf.keras.Sequential([
      tf.keras.layers.Input(shape=(fingerprint_size,)),
      tf.keras.layers.Dense(
          units=label_count,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.001)),
  ])


def create_conv_model(model_settings):
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

  During training, dropout is applied after each relu.

  Args:
    model_settings: Dictionary of information about the model.

  Returns:
    tf.keras.Model outputting logits.
  """
  fingerprint_size = model_settings['fingerprint_size']
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  label_count = model_settings['label_count']

  return tf.keras.Sequential([
      tf.keras.layers.Input(shape=(fingerprint_size,)),
      tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)),
      tf.keras.layers.Conv2D(
          filters=64,
          kernel_size=(20, 8),
          strides=(1, 1),
          padding='same',
          activation='relu',
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.MaxPooling2D(
          pool_size=(2, 2), strides=(2, 2), padding='same'),
      tf.keras.layers.Conv2D(
          filters=64,
          kernel_size=(10, 4),
          strides=(1, 1),
          padding='same',
          activation='relu',
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          units=label_count,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
  ])


def create_low_latency_conv_model(model_settings):
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

  Note that unlike 'conv', only the first layer has a ReLU/dropout -- the
  three fully-connected layers that follow are intentionally linear, matching
  the original topology (see the graph above; there's no [Relu] between the
  [MatMul]/[BiasAdd] pairs).

  Args:
    model_settings: Dictionary of information about the model.

  Returns:
    tf.keras.Model outputting logits.
  """
  fingerprint_size = model_settings['fingerprint_size']
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  label_count = model_settings['label_count']

  return tf.keras.Sequential([
      tf.keras.layers.Input(shape=(fingerprint_size,)),
      tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)),
      tf.keras.layers.Conv2D(
          filters=186,
          kernel_size=(input_time_size, 8),
          strides=(1, 1),
          padding='valid',
          activation='relu',
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          units=128,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          units=128,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          units=label_count,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
  ])


class SVDFLayer(tf.keras.layers.Layer):
  """Keras layer encapsulating the SVDF matrix math used for training.

  This is based on the topology presented in the 'Compressing Deep Neural
  Networks using a Rank-Constrained Topology' paper:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

  NOTE: the original TF1 implementation (see create_low_latency_svdf_model in
  the pre-TF2-migration revision of this file, in git history) kept a
  persistent 'runtime-memory' variable so that a *deployed/streaming* graph
  only had to process newly-arrived audio frames on each call, instead of
  reprocessing the whole window every time. That's a compute optimization for
  low-power/streaming inference, not a correctness requirement -- this layer
  intentionally only implements the training-time path (always processes the
  full window), which is simpler and produces identical predictions, just
  without that on-device optimization. If a future streaming deployment needs
  the incremental-memory behavior back, port it from git history as a
  separate inference-only code path rather than complicating this layer.
  """

  def __init__(self, num_units, rank, input_time_size, input_frequency_size,
              **kwargs):
    super(SVDFLayer, self).__init__(**kwargs)
    self.num_units = num_units
    self.rank = rank
    self.num_filters = rank * num_units
    self.input_time_size = input_time_size
    self.input_frequency_size = input_frequency_size

  def build(self, input_shape):
    self.weights_frequency = self.add_weight(
        name='weights_frequency',
        shape=(self.input_frequency_size, 1, self.num_filters),
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        trainable=True)
    self.weights_time = self.add_weight(
        name='weights_time',
        shape=(self.num_filters, self.input_time_size, 1),
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        trainable=True)
    self.bias_weight = self.add_weight(
        name='bias', shape=(self.num_units,), initializer='zeros',
        trainable=True)

  def call(self, inputs):
    # Expand to add an input-channels dimension, then convolve the 1D
    # frequency filters, sliding over the time dimension.
    # activations_time: [batch, input_time_size, num_filters]
    x = tf.expand_dims(inputs, 2)
    x = tf.nn.conv1d(
        x, self.weights_frequency, stride=self.input_frequency_size,
        padding='VALID')
    # Rearrange so we can perform the batched matmul.
    # x: [num_filters, batch, input_time_size]
    x = tf.transpose(x, [2, 0, 1])
    # Apply the time filter on the outputs of the frequency filters.
    # weights_time: [num_filters, input_time_size, 1]
    # x: [num_filters, batch, 1]
    x = tf.matmul(x, self.weights_time)
    # Split num_units and rank into separate dimensions (the remaining
    # dimension is the batch size), squeezing the unused last dimension.
    x = tf.reshape(x, [self.num_units, self.rank, -1])
    # Sum the rank outputs per unit => [num_units, batch].
    x = tf.reduce_sum(x, axis=1)
    # Transpose to shape [batch, num_units].
    x = tf.transpose(x)
    return tf.nn.relu(tf.nn.bias_add(x, self.bias_weight))

  def get_config(self):
    config = super(SVDFLayer, self).get_config()
    config.update({
        'num_units': self.num_units,
        'rank': self.rank,
        'input_time_size': self.input_time_size,
        'input_frequency_size': self.input_frequency_size,
    })
    return config


def create_low_latency_svdf_model(model_settings):
  """Builds an SVDF model with low compute requirements.

  This model produces lower recognition accuracy than the 'conv' model above,
  but requires fewer weight parameters and, significantly fewer computations.

  See SVDFLayer's docstring for a note on the streaming/runtime-memory
  optimization from the original TF1 implementation that this training-time
  version doesn't replicate.

  Args:
    model_settings: Dictionary of information about the model.

  Returns:
    tf.keras.Model outputting logits.

  Raises:
    ValueError: If the fingerprint size isn't a multiple of the frame size.
  """
  fingerprint_size = model_settings['fingerprint_size']
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  label_count = model_settings['label_count']

  if fingerprint_size % input_frequency_size != 0:
    raise ValueError(
        'The fingerprint size must be a multiple of the frame size = {0}, '
        'but was {1}'.format(input_frequency_size, fingerprint_size))

  rank = 2
  num_units = 1280

  inputs = tf.keras.Input(shape=(fingerprint_size,))
  x = SVDFLayer(
      num_units=num_units,
      rank=rank,
      input_time_size=input_time_size,
      input_frequency_size=input_frequency_size)(
          inputs)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = tf.keras.layers.Dense(
      units=256,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(
          x)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = tf.keras.layers.Dense(
      units=256,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(
          x)
  x = tf.keras.layers.Dropout(0.5)(x)
  outputs = tf.keras.layers.Dense(
      units=label_count,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(
          x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_tiny_conv_model(model_settings):
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

  During training, dropout is applied after the relu.

  Args:
    model_settings: Dictionary of information about the model.

  Returns:
    tf.keras.Model outputting logits.
  """
  fingerprint_size = model_settings['fingerprint_size']
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  label_count = model_settings['label_count']

  return tf.keras.Sequential([
      tf.keras.layers.Input(shape=(fingerprint_size,)),
      tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)),
      tf.keras.layers.Conv2D(
          filters=8,
          kernel_size=(10, 8),
          strides=(2, 2),
          padding='same',
          activation='relu',
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          units=label_count,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
  ])


def create_tiny_embedding_conv_model(model_settings):
  """Builds a convolutional model aimed at microcontrollers, with embeddings.

  This is a two-layer convolutional model, halving the spatial size at each
  layer, aimed at being small and fast enough to run on microcontrollers.

  Args:
    model_settings: Dictionary of information about the model.

  Returns:
    tf.keras.Model outputting logits.
  """
  fingerprint_size = model_settings['fingerprint_size']
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  label_count = model_settings['label_count']

  return tf.keras.Sequential([
      tf.keras.layers.Input(shape=(fingerprint_size,)),
      tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)),
      tf.keras.layers.Conv2D(
          filters=8,
          kernel_size=(10, 8),
          strides=(2, 2),
          padding='same',
          activation='relu',
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Conv2D(
          filters=8,
          kernel_size=(10, 8),
          strides=(8, 8),
          padding='same',
          activation='relu',
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          units=label_count,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=0.01)),
  ])
