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
"""Tests for data input for speech commands."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.examples.speech_commands import wav_to_features
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class WavToFeaturesTest(test.TestCase):

  def _getWavData(self):
    with self.cached_session() as sess:
      sample_data = tf.zeros([32000, 2])
      wav_encoder = contrib_audio.encode_wav(sample_data, 16000)
      wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _saveTestWavFile(self, filename, wav_data):
    with open(filename, "wb") as f:
      f.write(wav_data)

  def _saveWavFolders(self, root_dir, labels, how_many):
    wav_data = self._getWavData()
    for label in labels:
      dir_name = os.path.join(root_dir, label)
      os.mkdir(dir_name)
      for i in range(how_many):
        file_path = os.path.join(dir_name, "some_audio_%d.wav" % i)
        self._saveTestWavFile(file_path, wav_data)

  @test_util.run_deprecated_v1
  def testWavToFeatures(self):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    input_file_path = os.path.join(tmp_dir, "input.wav")
    output_file_path = os.path.join(tmp_dir, "output.c")
    wav_data = self._getWavData()
    self._saveTestWavFile(input_file_path, wav_data)
    wav_to_features.wav_to_features(16000, 1000, 10, 10, 40, True, "average",
                                    input_file_path, output_file_path)
    with open(output_file_path, "rb") as f:
      content = f.read()
      self.assertTrue(b"const unsigned char g_input_data" in content)

  @test_util.run_deprecated_v1
  def testWavToFeaturesMicro(self):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    input_file_path = os.path.join(tmp_dir, "input.wav")
    output_file_path = os.path.join(tmp_dir, "output.c")
    wav_data = self._getWavData()
    self._saveTestWavFile(input_file_path, wav_data)
    wav_to_features.wav_to_features(16000, 1000, 10, 10, 40, True, "micro",
                                    input_file_path, output_file_path)
    with open(output_file_path, "rb") as f:
      content = f.read()
      self.assertIn(b"const unsigned char g_input_data", content)


if __name__ == "__main__":
  test.main()
