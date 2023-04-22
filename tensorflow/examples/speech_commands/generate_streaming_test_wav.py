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
r"""Saves out a .wav file with synthesized conversational data and labels.

The best way to estimate the real-world performance of an audio recognition
model is by running it against a continuous stream of data, the way that it
would be used in an application. Training evaluations are only run against
discrete individual samples, so the results aren't as realistic.

To make it easy to run evaluations against audio streams, this script uses
samples from the testing partition of the data set, mixes them in at random
positions together with background noise, and saves out the result as one long
audio file.

Here's an example of generating a test file:

bazel run tensorflow/examples/speech_commands:generate_streaming_test_wav -- \
--data_dir=/tmp/my_wavs --background_dir=/tmp/my_backgrounds \
--background_volume=0.1 --test_duration_seconds=600 \
--output_audio_file=/tmp/streaming_test.wav \
--output_labels_file=/tmp/streaming_test_labels.txt

Once you've created a streaming audio file, you can then use the
test_streaming_accuracy tool to calculate accuracy metrics for a model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import sys

import numpy as np
import tensorflow as tf

import input_data
import models

FLAGS = None


def mix_in_audio_sample(track_data, track_offset, sample_data, sample_offset,
                        clip_duration, sample_volume, ramp_in, ramp_out):
  """Mixes the sample data into the main track at the specified offset.

  Args:
    track_data: Numpy array holding main audio data. Modified in-place.
    track_offset: Where to mix the sample into the main track.
    sample_data: Numpy array of audio data to mix into the main track.
    sample_offset: Where to start in the audio sample.
    clip_duration: How long the sample segment is.
    sample_volume: Loudness to mix the sample in at.
    ramp_in: Length in samples of volume increase stage.
    ramp_out: Length in samples of volume decrease stage.
  """
  ramp_out_index = clip_duration - ramp_out
  track_end = min(track_offset + clip_duration, track_data.shape[0])
  track_end = min(track_end,
                  track_offset + (sample_data.shape[0] - sample_offset))
  sample_range = track_end - track_offset
  for i in range(sample_range):
    if i < ramp_in:
      envelope_scale = i / ramp_in
    elif i > ramp_out_index:
      envelope_scale = (clip_duration - i) / ramp_out
    else:
      envelope_scale = 1
    sample_input = sample_data[sample_offset + i]
    track_data[track_offset
               + i] += sample_input * envelope_scale * sample_volume


def main(_):
  words_list = input_data.prepare_words_list(FLAGS.wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), FLAGS.sample_rate, FLAGS.clip_duration_ms,
      FLAGS.window_size_ms, FLAGS.window_stride_ms, FLAGS.feature_bin_count,
      'mfcc')
  audio_processor = input_data.AudioProcessor(
      '', FLAGS.data_dir, FLAGS.silence_percentage, 10,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings, FLAGS.data_dir)

  output_audio_sample_count = FLAGS.sample_rate * FLAGS.test_duration_seconds
  output_audio = np.zeros((output_audio_sample_count,), dtype=np.float32)

  # Set up background audio.
  background_crossover_ms = 500
  background_segment_duration_ms = (
      FLAGS.clip_duration_ms + background_crossover_ms)
  background_segment_duration_samples = int(
      (background_segment_duration_ms * FLAGS.sample_rate) / 1000)
  background_segment_stride_samples = int(
      (FLAGS.clip_duration_ms * FLAGS.sample_rate) / 1000)
  background_ramp_samples = int(
      ((background_crossover_ms / 2) * FLAGS.sample_rate) / 1000)

  # Mix the background audio into the main track.
  how_many_backgrounds = int(
      math.ceil(output_audio_sample_count / background_segment_stride_samples))
  for i in range(how_many_backgrounds):
    output_offset = int(i * background_segment_stride_samples)
    background_index = np.random.randint(len(audio_processor.background_data))
    background_samples = audio_processor.background_data[background_index]
    background_offset = np.random.randint(
        0, len(background_samples) - model_settings['desired_samples'])
    background_volume = np.random.uniform(0, FLAGS.background_volume)
    mix_in_audio_sample(output_audio, output_offset, background_samples,
                        background_offset, background_segment_duration_samples,
                        background_volume, background_ramp_samples,
                        background_ramp_samples)

  # Mix the words into the main track, noting their labels and positions.
  output_labels = []
  word_stride_ms = FLAGS.clip_duration_ms + FLAGS.word_gap_ms
  word_stride_samples = int((word_stride_ms * FLAGS.sample_rate) / 1000)
  clip_duration_samples = int(
      (FLAGS.clip_duration_ms * FLAGS.sample_rate) / 1000)
  word_gap_samples = int((FLAGS.word_gap_ms * FLAGS.sample_rate) / 1000)
  how_many_words = int(
      math.floor(output_audio_sample_count / word_stride_samples))
  all_test_data, all_test_labels = audio_processor.get_unprocessed_data(
      -1, model_settings, 'testing')
  for i in range(how_many_words):
    output_offset = (
        int(i * word_stride_samples) + np.random.randint(word_gap_samples))
    output_offset_ms = (output_offset * 1000) / FLAGS.sample_rate
    is_unknown = np.random.randint(100) < FLAGS.unknown_percentage
    if is_unknown:
      wanted_label = input_data.UNKNOWN_WORD_LABEL
    else:
      wanted_label = words_list[2 + np.random.randint(len(words_list) - 2)]
    test_data_start = np.random.randint(len(all_test_data))
    found_sample_data = None
    index_lookup = np.arange(len(all_test_data), dtype=np.int32)
    np.random.shuffle(index_lookup)
    for test_data_offset in range(len(all_test_data)):
      test_data_index = index_lookup[(
          test_data_start + test_data_offset) % len(all_test_data)]
      current_label = all_test_labels[test_data_index]
      if current_label == wanted_label:
        found_sample_data = all_test_data[test_data_index]
        break
    mix_in_audio_sample(output_audio, output_offset, found_sample_data, 0,
                        clip_duration_samples, 1.0, 500, 500)
    output_labels.append({'label': wanted_label, 'time': output_offset_ms})

  input_data.save_wav_file(FLAGS.output_audio_file, output_audio,
                           FLAGS.sample_rate)
  tf.compat.v1.logging.info('Saved streaming test wav to %s',
                            FLAGS.output_audio_file)

  with open(FLAGS.output_labels_file, 'w') as f:
    for output_label in output_labels:
      f.write('%s, %f\n' % (output_label['label'], output_label['time']))
  tf.compat.v1.logging.info('Saved streaming test labels to %s',
                            FLAGS.output_labels_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_dir',
      type=str,
      default='',
      help="""\
      Path to a directory of .wav files to mix in as background noise during training.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs.',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs.',)
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
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--output_audio_file',
      type=str,
      default='/tmp/speech_commands_train/streaming_test.wav',
      help='File to save the generated test audio to.')
  parser.add_argument(
      '--output_labels_file',
      type=str,
      default='/tmp/speech_commands_train/streaming_test_labels.txt',
      help='File to save the generated test labels to.')
  parser.add_argument(
      '--test_duration_seconds',
      type=int,
      default=600,
      help='How long the generated test audio file should be.',)
  parser.add_argument(
      '--word_gap_ms',
      type=int,
      default=2000,
      help='How long the average gap should be between words.',)
  parser.add_argument(
      '--unknown_percentage',
      type=int,
      default=30,
      help='What percentage of words should be unknown.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
