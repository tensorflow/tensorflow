# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Audio feature."""

import os
import struct
import wave

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core import lazy_imports_lib
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.features import feature
from tensorflow.data.experimental.core.utils import type_utils

Json = type_utils.Json


class Audio(feature.Tensor):
  """`FeatureConnector` for audio, encoded as raw integer wave form."""

  def __init__(
      self,
      *,
      file_format=None,
      shape=(None,),
      dtype=tf.int64,
      sample_rate=None,
  ):
    """Constructs the connector.

    Args:
      file_format: `str`, the audio file format. Can be any format ffmpeg
        understands. If `None`, will attempt to infer from the file extension.
      shape: `tuple`, shape of the data.
      dtype: The dtype of the data.
      sample_rate: `int`, additional metadata exposed to the user through
        `info.features['audio'].sample_rate`. This value isn't used neither in
        encoding nor decoding.
    """
    self._file_format = file_format
    if len(shape) != 1:
      raise TypeError(
          'Audio feature currently only supports 1-D values, got %s.' % shape)
    self._shape = shape
    self._sample_rate = sample_rate
    super(Audio, self).__init__(shape=shape, dtype=dtype)

  def _encode_file(self, fobj, file_format):
    audio_segment = lazy_imports_lib.lazy_imports.pydub.AudioSegment.from_file(
        fobj, format=file_format)
    np_dtype = np.dtype(self.dtype.as_numpy_dtype)
    return super(Audio, self).encode_example(
        np.array(audio_segment.get_array_of_samples()).astype(np_dtype))

  def encode_example(self, audio_or_path_or_fobj):
    if isinstance(audio_or_path_or_fobj, (np.ndarray, list)):
      return audio_or_path_or_fobj
    elif isinstance(audio_or_path_or_fobj, type_utils.PathLikeCls):
      filename = os.fspath(audio_or_path_or_fobj)
      file_format = self._file_format or filename.split('.')[-1]
      with tf.io.gfile.GFile(filename, 'rb') as audio_f:
        try:
          return self._encode_file(audio_f, file_format)
        except Exception as e:  # pylint: disable=broad-except
          utils.reraise(e, prefix=f'Error for {filename}: ')
    else:
      return self._encode_file(audio_or_path_or_fobj, self._file_format)

  @property
  def sample_rate(self):
    """Returns the `sample_rate` metadata associated with the dataset."""
    return self._sample_rate

  def repr_html(self, ex: np.ndarray) -> str:
    """Audio are displayed in the player."""
    if self.sample_rate:
      rate = self.sample_rate
    else:
      # We should display an error message once to warn the user the sample
      # rate was auto-infered. Requirements:
      # * Should appear only once (even though repr_html is called once per
      #   examples)
      # * Ideally should appear on Colab (while `logging.warning` is hidden
      #   by default)
      rate = 16000

    audio_str = utils.get_base64(
        lambda buff: _save_wav(buff, ex, rate)
    )
    return (
        f'<audio controls src="data:audio/ogg;base64,{audio_str}" '
        ' controlsList="nodownload" />'
    )

  @classmethod
  def from_json_content(cls, value: Json) -> 'Audio':
    return cls(
        file_format=value['file_format'],
        shape=tuple(value['shape']),
        dtype=tf.dtypes.as_dtype(value['dtype']),
        sample_rate=value['sample_rate'],
    )

  def to_json_content(self) -> Json:
    return {
        'file_format': self._file_format,
        'shape': list(self._shape),
        'dtype': self._dtype.name,
        'sample_rate': self._sample_rate,
    }


def _save_wav(buff, data, rate) -> None:
  """Transform a numpy array to a PCM bytestring."""
  # Code inspired from `IPython.display.Audio`
  data = np.array(data, dtype=float)
  if len(data.shape) > 1:
    raise ValueError('encoding of stereo PCM signals are unsupported')
  scaled = np.int16(data / np.max(np.abs(data)) * 32767).tolist()

  with wave.open(buff, mode='wb') as waveobj:
    waveobj.setnchannels(1)
    waveobj.setframerate(rate)
    waveobj.setsampwidth(2)
    waveobj.setcomptype('NONE', 'NONE')
    waveobj.writeframes(b''.join([struct.pack('<h', x) for x in scaled]))
