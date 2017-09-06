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
"""Encoding and decoding audio using FFmpeg."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.ffmpeg.ops import gen_decode_audio_op_py
from tensorflow.contrib.ffmpeg.ops import gen_encode_audio_op_py
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

_ffmpeg_so = loader.load_op_library(
    resource_loader.get_path_to_datafile('ffmpeg.so'))


def decode_audio(contents, file_format=None, samples_per_second=None,
                 channel_count=None):
  """Create an op that decodes the contents of an audio file.

  Note that ffmpeg is free to select the "best" audio track from an mp4.
  https://trac.ffmpeg.org/wiki/Map

  Args:
    contents: The binary contents of the audio file to decode. This is a
        scalar.
    file_format: A string or scalar string tensor specifying which
        format the contents will conform to. This can be mp3, mp4, ogg,
        or wav.
    samples_per_second: The number of samples per second that is
        assumed, as an `int` or scalar `int32` tensor. In some cases,
        resampling will occur to generate the correct sample rate.
    channel_count: The number of channels that should be created from the
        audio contents, as an `int` or scalar `int32` tensor. If the
        `contents` have more than this number, then some channels will
        be merged or dropped. If `contents` has fewer than this, then
        additional channels will be created from the existing ones.

  Returns:
    A rank-2 tensor that has time along dimension 0 and channels along
    dimension 1. Dimension 0 will be `samples_per_second *
    length_in_seconds` wide, and dimension 1 will be `channel_count`
    wide. If ffmpeg fails to decode the audio then an empty tensor will
    be returned.
  """
  return gen_decode_audio_op_py.decode_audio_v2(
      contents, file_format=file_format, samples_per_second=samples_per_second,
      channel_count=channel_count)


ops.NotDifferentiable('DecodeAudio')


def encode_audio(audio, file_format=None, samples_per_second=None):
  """Creates an op that encodes an audio file using sampled audio from a tensor.

  Args:
    audio: A rank-2 `Tensor` that has time along dimension 0 and
        channels along dimension 1. Dimension 0 is `samples_per_second *
        length_in_seconds` long.
    file_format: The type of file to encode, as a string or rank-0
        string tensor. "wav" is the only supported format.
    samples_per_second: The number of samples in the audio tensor per
        second of audio, as an `int` or rank-0 `int32` tensor.

  Returns:
    A scalar tensor that contains the encoded audio in the specified file
    format.
  """
  return gen_encode_audio_op_py.encode_audio_v2(
      audio,
      file_format=file_format,
      samples_per_second=samples_per_second,
      bits_per_second=192000)  # not used by WAV


ops.NotDifferentiable('EncodeAudio')
