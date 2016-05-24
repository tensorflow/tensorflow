# Copyright 2016 Google Inc. All Rights Reserved.
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
# =============================================================================

"""Tests for third_party.tensorflow.contrib.ffmpeg.decode_audio_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

from tensorflow.contrib import ffmpeg
from tensorflow.python.platform import resource_loader


class DecodeAudioOpTest(tf.test.TestCase):

  def _loadFileAndTest(self, filename, file_format, duration_sec,
                       samples_per_second, channel_count):
    """Loads an audio file and validates the output tensor.

    Args:
      filename: The filename of the input file.
      file_format: The format of the input file.
      duration_sec: The duration of the audio contained in the file in seconds.
      samples_per_second: The desired sample rate in the output tensor.
      channel_count: The desired channel count in the output tensor.
    """
    with self.test_session():
      path = os.path.join(
          resource_loader.get_data_files_path(), 'testdata', filename)
      with open(path, 'rb') as f:
        contents = f.read()

      audio_op = ffmpeg.decode_audio(
          contents, file_format=file_format,
          samples_per_second=samples_per_second, channel_count=channel_count)
      audio = audio_op.eval()
      self.assertEqual(len(audio.shape), 2)
      self.assertNear(duration_sec * samples_per_second,
                      audio.shape[0],
                      # Duration should be specified within 10%:
                      0.1 * audio.shape[0])
      self.assertEqual(audio.shape[1], channel_count)

  def testMonoMp3(self):
    self._loadFileAndTest('mono_16khz.mp3', 'mp3', 0.57, 20000, 1)
    self._loadFileAndTest('mono_16khz.mp3', 'mp3', 0.57, 20000, 2)

  def testStereoMp3(self):
    self._loadFileAndTest('stereo_48khz.mp3', 'mp3', 0.79, 50000, 1)
    self._loadFileAndTest('stereo_48khz.mp3', 'mp3', 0.79, 20000, 2)

  def testMonoWav(self):
    self._loadFileAndTest('mono_10khz.wav', 'wav', 0.57, 5000, 1)
    self._loadFileAndTest('mono_10khz.wav', 'wav', 0.57, 10000, 4)

  def testOgg(self):
    self._loadFileAndTest('mono_10khz.ogg', 'ogg', 0.57, 10000, 1)


if __name__ == '__main__':
  tf.test.main()
