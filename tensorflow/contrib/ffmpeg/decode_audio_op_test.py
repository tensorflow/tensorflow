# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import six

from tensorflow.contrib import ffmpeg
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class DecodeAudioOpTest(test.TestCase):

  def _loadFileAndTest(self, filename, file_format, duration_sec,
                       samples_per_second, channel_count,
                       samples_per_second_tensor=None, feed_dict=None,
                       stream=None):
    """Loads an audio file and validates the output tensor.

    Args:
      filename: The filename of the input file.
      file_format: The format of the input file.
      duration_sec: The duration of the audio contained in the file in seconds.
      samples_per_second: The desired sample rate in the output tensor.
      channel_count: The desired channel count in the output tensor.
      samples_per_second_tensor: The value to pass to the corresponding
        parameter in the instantiated `decode_audio` op. If not
        provided, will default to a constant value of
        `samples_per_second`. Useful for providing a placeholder.
      feed_dict: Used when evaluating the `decode_audio` op. If not
        provided, will be empty. Useful when providing a placeholder for
        `samples_per_second_tensor`.
      stream: A string specifying which stream from the content file
        should be decoded. The default value is '' which leaves the
        decision to ffmpeg.
    """
    if samples_per_second_tensor is None:
      samples_per_second_tensor = samples_per_second
    with self.test_session():
      path = os.path.join(resource_loader.get_data_files_path(), 'testdata',
                          filename)
      with open(path, 'rb') as f:
        contents = f.read()

      audio_op = ffmpeg.decode_audio(
          contents,
          file_format=file_format,
          samples_per_second=samples_per_second_tensor,
          channel_count=channel_count, stream=stream)
      audio = audio_op.eval(feed_dict=feed_dict or {})
      self.assertEqual(len(audio.shape), 2)
      self.assertNear(
          duration_sec * samples_per_second,
          audio.shape[0],
          # Duration should be specified within 10%:
          0.1 * audio.shape[0])
      self.assertEqual(audio.shape[1], channel_count)

  def testStreamIdentifier(self):
    # mono_16khz_mp3_32khz_aac.mp4 was generated from:
    # ffmpeg -i tensorflow/contrib/ffmpeg/testdata/mono_16khz_mp3.mp4 \
    #        -i tensorflow/contrib/ffmpeg/testdata/mono_32khz_aac.mp4 \
    #        -strict -2 -map 0:a -map 1:a \
    #        tensorflow/contrib/ffmpeg/testdata/mono_16khz_mp3_32khz_aac.mp4
    self._loadFileAndTest('mono_16khz_mp3_32khz_aac.mp4', 'mp4', 2.77, 20000,
                          1, stream='0')
    self._loadFileAndTest('mono_16khz_mp3_32khz_aac.mp4', 'mp4', 2.77, 20000,
                          1, stream='1')

  def testMonoMp3(self):
    self._loadFileAndTest('mono_16khz.mp3', 'mp3', 0.57, 20000, 1)
    self._loadFileAndTest('mono_16khz.mp3', 'mp3', 0.57, 20000, 2)

  def testMonoMp4Mp3Codec(self):
    # mp3 compressed audio streams in mp4 container.
    self._loadFileAndTest('mono_16khz_mp3.mp4', 'mp4', 2.77, 20000, 1)
    self._loadFileAndTest('mono_16khz_mp3.mp4', 'mp4', 2.77, 20000, 2)

  def testMonoMp4AacCodec(self):
    # aac compressed audio streams in mp4 container.
    self._loadFileAndTest('mono_32khz_aac.mp4', 'mp4', 2.77, 20000, 1)
    self._loadFileAndTest('mono_32khz_aac.mp4', 'mp4', 2.77, 20000, 2)

  def testStereoMp3(self):
    self._loadFileAndTest('stereo_48khz.mp3', 'mp3', 0.79, 50000, 1)
    self._loadFileAndTest('stereo_48khz.mp3', 'mp3', 0.79, 20000, 2)

  def testStereoMp4Mp3Codec(self):
    # mp3 compressed audio streams in mp4 container.
    self._loadFileAndTest('stereo_48khz_mp3.mp4', 'mp4', 0.79, 50000, 1)
    self._loadFileAndTest('stereo_48khz_mp3.mp4', 'mp4', 0.79, 20000, 2)

  def testStereoMp4AacCodec(self):
    # aac compressed audio streams in mp4 container.
    self._loadFileAndTest('stereo_48khz_aac.mp4', 'mp4', 0.79, 50000, 1)
    self._loadFileAndTest('stereo_48khz_aac.mp4', 'mp4', 0.79, 20000, 2)

  def testMonoWav(self):
    self._loadFileAndTest('mono_10khz.wav', 'wav', 0.57, 5000, 1)
    self._loadFileAndTest('mono_10khz.wav', 'wav', 0.57, 10000, 4)

  def testOgg(self):
    self._loadFileAndTest('mono_10khz.ogg', 'ogg', 0.57, 10000, 1)

  def testInvalidFile(self):
    with self.test_session():
      contents = 'invalid file'
      audio_op = ffmpeg.decode_audio(
          contents,
          file_format='wav',
          samples_per_second=10000,
          channel_count=2)
      audio = audio_op.eval()
      self.assertEqual(audio.shape, (0, 0))

  def testSampleRatePlaceholder(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    self._loadFileAndTest('mono_16khz.mp3', 'mp3', 0.57, 20000, 1,
                          samples_per_second_tensor=placeholder,
                          feed_dict={placeholder: 20000})

  def testSampleRateBadType(self):
    placeholder = array_ops.placeholder(dtypes.float32)
    with self.assertRaises(TypeError):
      self._loadFileAndTest('mono_16khz.mp3', 'mp3', 0.57, 20000.0, 1,
                            samples_per_second_tensor=placeholder,
                            feed_dict={placeholder: 20000.0})

  def testSampleRateBadValue_Zero(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    with six.assertRaisesRegex(self, Exception,
                               r'samples_per_second must be positive'):
      self._loadFileAndTest('mono_16khz.mp3', 'mp3', 0.57, 20000.0, 1,
                            samples_per_second_tensor=placeholder,
                            feed_dict={placeholder: 0})

  def testSampleRateBadValue_Negative(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    with six.assertRaisesRegex(self, Exception,
                               r'samples_per_second must be positive'):
      self._loadFileAndTest('mono_16khz.mp3', 'mp3', 0.57, 20000.0, 1,
                            samples_per_second_tensor=placeholder,
                            feed_dict={placeholder: -2})

  def testInvalidFileFormat(self):
    with six.assertRaisesRegex(self, Exception,
                               r'file_format must be one of'):
      self._loadFileAndTest('mono_16khz.mp3', 'docx', 0.57, 20000, 1)

  def testStaticShapeInference_ConstantChannelCount(self):
    with self.test_session():
      audio_op = ffmpeg.decode_audio(b'~~~ wave ~~~',
                                     file_format='wav',
                                     samples_per_second=44100,
                                     channel_count=2)
      self.assertEqual([None, 2], audio_op.shape.as_list())

  def testStaticShapeInference_NonConstantChannelCount(self):
    with self.test_session():
      channel_count = array_ops.placeholder(dtypes.int32)
      audio_op = ffmpeg.decode_audio(b'~~~ wave ~~~',
                                     file_format='wav',
                                     samples_per_second=44100,
                                     channel_count=channel_count)
      self.assertEqual([None, None], audio_op.shape.as_list())

  def testStaticShapeInference_ZeroChannelCountInvalid(self):
    with self.test_session():
      with six.assertRaisesRegex(self, Exception,
                                 r'channel_count must be positive'):
        ffmpeg.decode_audio(b'~~~ wave ~~~',
                            file_format='wav',
                            samples_per_second=44100,
                            channel_count=0)

  def testStaticShapeInference_NegativeChannelCountInvalid(self):
    with self.test_session():
      with six.assertRaisesRegex(self, Exception,
                                 r'channel_count must be positive'):
        ffmpeg.decode_audio(b'~~~ wave ~~~',
                            file_format='wav',
                            samples_per_second=44100,
                            channel_count=-2)


if __name__ == '__main__':
  test.main()
