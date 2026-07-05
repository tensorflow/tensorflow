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
r"""Converts a trained checkpoint into a frozen model for mobile inference.

Once you've trained a model using the `train.py` script, you can use this tool
to convert it into a binary GraphDef file that can be loaded into the Android,
iOS, or Raspberry Pi example code. Here's an example of how to run it:

bazel run tensorflow/examples/speech_commands/freeze -- \
--sample_rate=16000 --dct_coefficient_count=40 --window_size_ms=20 \
--window_stride_ms=10 --clip_duration_ms=1000 \
--model_architecture=conv \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
--output_file=/tmp/my_frozen_graph.pb

One thing to watch out for is that you need to pass in the same arguments for
`sample_rate` and other command line variables here as you did for the training
script.

The resulting graph has an input for WAV-encoded data named 'wav_data', one for
raw PCM data (as floats in the range -1.0 to 1.0) called 'decoded_sample_data',
and the output is called 'labels_softmax'.

"""
import argparse
import os.path

import tensorflow as tf

import input_data
import models
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.ops import gen_audio_ops as audio_ops

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
# bazel run tensorflow/examples/speech_commands:freeze_graph
try:
  from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
  frontend_op = None

FLAGS = None


def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           feature_bin_count, model_architecture, preprocess):
  """Creates a model and a traced serving function wired up for inference.

  Builds the requested model, and a tf.function that takes WAV-encoded audio
  data and returns softmax class probabilities, suitable for tracing and
  freezing into a single self-contained inference graph.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    clip_stride_ms: How often to run recognition. Useful for models with cache.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    feature_bin_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
    preprocess: How the spectrogram is processed to produce features, for
      example 'mfcc', 'average', or 'micro'.

  Returns:
    A tuple of (model, serve). `model` is the Keras model that does the
    classification -- callers should restore trained weights onto it (e.g.
    with tf.train.Checkpoint) before tracing/freezing `serve`. `serve` is an
    (as yet untraced) tf.function that takes WAV-encoded bytes ('wav_data')
    and returns class probabilities ('labels_softmax'); the intermediate
    decoded-audio node is explicitly named 'decoded_sample_data' (outputs :0
    audio, :1 sample_rate) so tools that feed pre-decoded samples directly
    into the frozen graph (e.g. test_streaming_accuracy.py) keep working
    unchanged.

  Raises:
    Exception: If the preprocessing mode isn't recognized.
  """
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, feature_bin_count, preprocess)
  fingerprint_size = model_settings['fingerprint_size']

  model = models.create_model(model_settings, model_architecture)
  # Keras layers build their weight variables lazily on first call. Building
  # now (rather than waiting for `serve` to be traced) lets a caller restore a
  # trained checkpoint onto `model` before `serve` gets traced/frozen below.
  model.build(input_shape=(None, fingerprint_size))

  def _fingerprint_from_decoded(decoded_audio, decoded_sample_rate):
    """Runs the spectrogram/mfcc/average/micro feature pipeline."""
    spectrogram = audio_ops.audio_spectrogram(
        decoded_audio,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)

    if preprocess == 'average':
      fingerprint = tf.nn.pool(
          input=tf.expand_dims(spectrogram, -1),
          window_shape=[1, model_settings['average_window_width']],
          strides=[1, model_settings['average_window_width']],
          pooling_type='AVG',
          padding='SAME')
    elif preprocess == 'mfcc':
      fingerprint = audio_ops.mfcc(
          spectrogram,
          decoded_sample_rate,
          dct_coefficient_count=model_settings['fingerprint_width'])
    elif preprocess == 'micro':
      if not frontend_op:
        raise Exception(
            'Micro frontend op is currently not available when running'
            ' TensorFlow directly from Python, you need to build and run'
            ' through Bazel, for example'
            ' `bazel run tensorflow/examples/speech_commands:freeze_graph`')
      micro_sample_rate = model_settings['sample_rate']
      micro_window_size_ms = (model_settings['window_size_samples'] *
                              1000) / micro_sample_rate
      micro_window_step_ms = (model_settings['window_stride_samples'] *
                              1000) / micro_sample_rate
      int16_input = tf.cast(tf.multiply(decoded_audio, 32767), tf.int16)
      micro_frontend = frontend_op.audio_microfrontend(
          int16_input,
          sample_rate=micro_sample_rate,
          window_size=micro_window_size_ms,
          window_step=micro_window_step_ms,
          num_channels=model_settings['fingerprint_width'],
          out_scale=1,
          out_type=tf.float32)
      fingerprint = tf.multiply(micro_frontend, (10.0 / 256.0))
    else:
      raise Exception('Unknown preprocess mode "%s" (should be "mfcc",'
                      ' "average", or "micro")' % (preprocess))

    return tf.reshape(fingerprint, [-1, fingerprint_size])

  @tf.function(input_signature=[tf.TensorSpec([], tf.string, name='wav_data')])
  def serve(wav_data):
    decoded_sample_data = tf.audio.decode_wav(
        wav_data,
        desired_channels=1,
        desired_samples=model_settings['desired_samples'],
        name='decoded_sample_data')
    fingerprint_input = _fingerprint_from_decoded(
        decoded_sample_data.audio, decoded_sample_data.sample_rate)
    logits = model(fingerprint_input, training=False)
    return tf.nn.softmax(logits, name='labels_softmax')

  return model, serve


def save_graph_def(file_name, frozen_graph_def):
  """Writes a graph def file out to disk.

  Args:
    file_name: Where to save the file.
    frozen_graph_def: GraphDef proto object to save.
  """
  tf.io.write_graph(
      frozen_graph_def,
      os.path.dirname(file_name),
      os.path.basename(file_name),
      as_text=False)
  tf.get_logger().info('Saved frozen graph to %s', file_name)


def save_saved_model(file_name, model, serve_concrete_fn):
  """Writes a SavedModel out to disk.

  Args:
    file_name: Where to save the SavedModel.
    model: The Keras model whose trained weights should be saved alongside
      the serving signature (also gives tf.saved_model.save an object graph
      to walk for tracked variables).
    serve_concrete_fn: The traced serving tf.function (wav_data in,
      labels_softmax out) to expose as the default serving signature.
  """
  tf.saved_model.save(
      model,
      file_name,
      signatures={
          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: serve_concrete_fn
      })


def main(_):
  if FLAGS.quantize:
    # TODO: Quantization-aware export doesn't have a TF2 story yet in this
    # example. Since deployment targets TFLite, the intended follow-up is to
    # freeze the float model as usual (below) and then run it through
    # tf.lite.TFLiteConverter's post-training quantization (or, if that isn't
    # accurate enough, retrain with quantization-aware training via
    # tensorflow_model_optimization and freeze that instead) -- see the
    # matching TODO in train.py.
    raise Exception(
        '--quantize is not yet supported when exporting for TF2. See the '
        'TODO comment in main() in freeze.py for the intended follow-up '
        '(TFLite post-training quantization).')

  # Create the model and load its trained weights.
  model, serve_fn = create_inference_graph(
      FLAGS.wanted_words, FLAGS.sample_rate, FLAGS.clip_duration_ms,
      FLAGS.clip_stride_ms, FLAGS.window_size_ms, FLAGS.window_stride_ms,
      FLAGS.feature_bin_count, FLAGS.model_architecture, FLAGS.preprocess)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(FLAGS.start_checkpoint).expect_partial()

  # Trace the serving function now that the trained weights are in place, and
  # turn all the variables it references into inline constants.
  concrete_fn = serve_fn.get_concrete_function()
  frozen_func = convert_to_constants.convert_variables_to_constants_v2(
      concrete_fn)

  if FLAGS.save_format == 'graph_def':
    save_graph_def(FLAGS.output_file, frozen_func.graph.as_graph_def())
  elif FLAGS.save_format == 'saved_model':
    save_saved_model(FLAGS.output_file, model, concrete_fn)
  else:
    raise Exception('Unknown save format "%s" (should be "graph_def" or'
                    ' "saved_model")' % (FLAGS.save_format))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--clip_stride_ms',
      type=int,
      default=30,
      help='How often to run recognition. Useful for models with cache.',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long the stride is between spectrogram timeslices',)
  parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',
  )
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--output_file', type=str, help='Where to save the frozen graph.')
  parser.add_argument(
      '--quantize',
      type=bool,
      default=False,
      help='Whether to train the model for eight-bit deployment')
  parser.add_argument(
      '--preprocess',
      type=str,
      default='mfcc',
      help='Spectrogram processing mode. Can be "mfcc" or "average"')
  parser.add_argument(
      '--save_format',
      type=str,
      default='graph_def',
      help='How to save the result. Can be "graph_def" or "saved_model"')
  FLAGS, _ = parser.parse_known_args()
  main(None)
