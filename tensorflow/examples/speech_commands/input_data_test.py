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
"""Tests for data input for speech commands."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.examples.speech_commands import input_data
from tensorflow.examples.speech_commands import models
from tensorflow.python.platform import test


class InputDataTest(test.TestCase):

  def _getWavData(self):
    with self.cached_session() as sess:
      sample_data = tf.zeros([32000, 2])
      wav_encoder = contrib_audio.encode_wav(sample_data, 16000)
      wav_data = sess.run(wav_encoder)
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

  def _model_settings(self):
    return {
        "desired_samples": 160,
        "fingerprint_size": 40,
        "label_count": 4,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "fingerprint_width": 40,
        "preprocess": "mfcc",
    }

  def _runGetDataTest(self, preprocess, window_length_ms):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    background_dir = os.path.join(wav_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._getWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._saveTestWavFile(file_path, wav_data)
    model_settings = models.prepare_model_settings(
        4, 16000, 1000, window_length_ms, 20, 40, preprocess)
    with self.cached_session() as sess:
      audio_processor = input_data.AudioProcessor(
          "", wav_dir, 10, 10, ["a", "b"], 10, 10, model_settings, tmp_dir)
      result_data, result_labels = audio_processor.get_data(
          10, 0, model_settings, 0.3, 0.1, 100, "training", sess)
      self.assertEqual(10, len(result_data))
      self.assertEqual(10, len(result_labels))

  def testPrepareWordsList(self):
    words_list = ["a", "b"]
    self.assertGreater(
        len(input_data.prepare_words_list(words_list)), len(words_list))

  def testWhichSet(self):
    self.assertEqual(
        input_data.which_set("foo.wav", 10, 10),
        input_data.which_set("foo.wav", 10, 10))
    self.assertEqual(
        input_data.which_set("foo_nohash_0.wav", 10, 10),
        input_data.which_set("foo_nohash_1.wav", 10, 10))

  def testPrepareDataIndex(self):
    tmp_dir = self.get_temp_dir()
    self._saveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    audio_processor = input_data.AudioProcessor("", tmp_dir, 10, 10,
                                                ["a", "b"], 10, 10,
                                                self._model_settings(), tmp_dir)
    self.assertLess(0, audio_processor.set_size("training"))
    self.assertTrue("training" in audio_processor.data_index)
    self.assertTrue("validation" in audio_processor.data_index)
    self.assertTrue("testing" in audio_processor.data_index)
    self.assertEquals(input_data.UNKNOWN_WORD_INDEX,
                      audio_processor.word_to_index["c"])

  def testPrepareDataIndexEmpty(self):
    tmp_dir = self.get_temp_dir()
    self._saveWavFolders(tmp_dir, ["a", "b", "c"], 0)
    with self.assertRaises(Exception) as e:
      _ = input_data.AudioProcessor("", tmp_dir, 10, 10, ["a", "b"], 10, 10,
                                    self._model_settings(), tmp_dir)
    self.assertTrue("No .wavs found" in str(e.exception))

  def testPrepareDataIndexMissing(self):
    tmp_dir = self.get_temp_dir()
    self._saveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    with self.assertRaises(Exception) as e:
      _ = input_data.AudioProcessor("", tmp_dir, 10, 10, ["a", "b", "d"], 10,
                                    10, self._model_settings(), tmp_dir)
    self.assertTrue("Expected to find" in str(e.exception))

  def testPrepareBackgroundData(self):
    tmp_dir = self.get_temp_dir()
    background_dir = os.path.join(tmp_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._getWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._saveTestWavFile(file_path, wav_data)
    self._saveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    audio_processor = input_data.AudioProcessor("", tmp_dir, 10, 10,
                                                ["a", "b"], 10, 10,
                                                self._model_settings(), tmp_dir)
    self.assertEqual(10, len(audio_processor.background_data))

  def testLoadWavFile(self):
    tmp_dir = self.get_temp_dir()
    file_path = os.path.join(tmp_dir, "load_test.wav")
    wav_data = self._getWavData()
    self._saveTestWavFile(file_path, wav_data)
    sample_data = input_data.load_wav_file(file_path)
    self.assertIsNotNone(sample_data)

  def testSaveWavFile(self):
    tmp_dir = self.get_temp_dir()
    file_path = os.path.join(tmp_dir, "load_test.wav")
    save_data = np.zeros([16000, 1])
    input_data.save_wav_file(file_path, save_data, 16000)
    loaded_data = input_data.load_wav_file(file_path)
    self.assertIsNotNone(loaded_data)
    self.assertEqual(16000, len(loaded_data))

  def testPrepareProcessingGraph(self):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    background_dir = os.path.join(wav_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._getWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._saveTestWavFile(file_path, wav_data)
    model_settings = {
        "desired_samples": 160,
        "fingerprint_size": 40,
        "label_count": 4,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "fingerprint_width": 40,
        "preprocess": "mfcc",
    }
    audio_processor = input_data.AudioProcessor("", wav_dir, 10, 10, ["a", "b"],
                                                10, 10, model_settings, tmp_dir)
    self.assertIsNotNone(audio_processor.wav_filename_placeholder_)
    self.assertIsNotNone(audio_processor.foreground_volume_placeholder_)
    self.assertIsNotNone(audio_processor.time_shift_padding_placeholder_)
    self.assertIsNotNone(audio_processor.time_shift_offset_placeholder_)
    self.assertIsNotNone(audio_processor.background_data_placeholder_)
    self.assertIsNotNone(audio_processor.background_volume_placeholder_)
    self.assertIsNotNone(audio_processor.output_)

  def testGetDataAverage(self):
    self._runGetDataTest("average", 10)

  def testGetDataAverageLongWindow(self):
    self._runGetDataTest("average", 30)

  def testGetDataMfcc(self):
    self._runGetDataTest("mfcc", 30)

  def testGetUnprocessedData(self):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    model_settings = {
        "desired_samples": 160,
        "fingerprint_size": 40,
        "label_count": 4,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "fingerprint_width": 40,
        "preprocess": "mfcc",
    }
    audio_processor = input_data.AudioProcessor("", wav_dir, 10, 10, ["a", "b"],
                                                10, 10, model_settings, tmp_dir)
    result_data, result_labels = audio_processor.get_unprocessed_data(
        10, model_settings, "training")
    self.assertEqual(10, len(result_data))
    self.assertEqual(10, len(result_labels))


if __name__ == "__main__":
  test.main()
