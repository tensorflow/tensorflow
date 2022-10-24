# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import os
import unittest

import tensorflow as tf

from tensorflow.examples.speech_commands import train
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


def requires_contrib(test_method):
  try:
    _ = tf.contrib
  except AttributeError:
    test_method = unittest.skip(
        'This test requires tf.contrib:\n    `pip install tensorflow<=1.15`')(
            test_method)

  return test_method


# Used to convert a dictionary into an object, for mocking parsed flags.
class DictStruct(object):

  def __init__(self, **entries):
    self.__dict__.update(entries)


class TrainTest(test.TestCase):

  def _getWavData(self):
    with self.cached_session():
      sample_data = tf.zeros([32000, 2])
      wav_encoder = tf.audio.encode_wav(sample_data, 16000)
      wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _saveTestWavFile(self, filename, wav_data):
    with open(filename, 'wb') as f:
      f.write(wav_data)

  def _saveWavFolders(self, root_dir, labels, how_many):
    wav_data = self._getWavData()
    for label in labels:
      dir_name = os.path.join(root_dir, label)
      os.mkdir(dir_name)
      for i in range(how_many):
        file_path = os.path.join(dir_name, 'some_audio_%d.wav' % i)
        self._saveTestWavFile(file_path, wav_data)

  def _prepareDummyTrainingData(self):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, 'wavs')
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ['a', 'b', 'c'], 100)
    background_dir = os.path.join(wav_dir, '_background_noise_')
    os.mkdir(background_dir)
    wav_data = self._getWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, 'background_audio_%d.wav' % i)
      self._saveTestWavFile(file_path, wav_data)
    return wav_dir

  def _getDefaultFlags(self):
    flags = {
        'data_url': '',
        'data_dir': self._prepareDummyTrainingData(),
        'wanted_words': 'a,b,c',
        'sample_rate': 16000,
        'clip_duration_ms': 1000,
        'window_size_ms': 30,
        'window_stride_ms': 20,
        'feature_bin_count': 40,
        'preprocess': 'mfcc',
        'silence_percentage': 25,
        'unknown_percentage': 25,
        'validation_percentage': 10,
        'testing_percentage': 10,
        'summaries_dir': os.path.join(self.get_temp_dir(), 'summaries'),
        'train_dir': os.path.join(self.get_temp_dir(), 'train'),
        'time_shift_ms': 100,
        'how_many_training_steps': '2',
        'learning_rate': '0.01',
        'quantize': False,
        'model_architecture': 'conv',
        'check_nans': False,
        'start_checkpoint': '',
        'batch_size': 1,
        'background_volume': 0.25,
        'background_frequency': 0.8,
        'eval_step_interval': 1,
        'save_step_interval': 1,
        'verbosity': tf.compat.v1.logging.INFO,
        'optimizer': 'gradient_descent'
    }
    return DictStruct(**flags)

  @test_util.run_deprecated_v1
  def testTrain(self):
    train.FLAGS = self._getDefaultFlags()
    train.main('')
    self.assertTrue(
        gfile.Exists(
            os.path.join(train.FLAGS.train_dir,
                         train.FLAGS.model_architecture + '.pbtxt')))
    self.assertTrue(
        gfile.Exists(
            os.path.join(train.FLAGS.train_dir,
                         train.FLAGS.model_architecture + '_labels.txt')))
    self.assertTrue(
        gfile.Exists(
            os.path.join(train.FLAGS.train_dir,
                         train.FLAGS.model_architecture + '.ckpt-1.meta')))


if __name__ == '__main__':
  test.main()
