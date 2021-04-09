# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for preprocessing sequence data."""
# pylint: disable=invalid-name

from keras_preprocessing import sequence

from tensorflow.python.keras.utils import data_utils
from tensorflow.python.util.tf_export import keras_export

make_sampling_table = sequence.make_sampling_table
skipgrams = sequence.skipgrams
# TODO(fchollet): consider making `_remove_long_seq` public.
_remove_long_seq = sequence._remove_long_seq  # pylint: disable=protected-access


@keras_export('keras.preprocessing.sequence.TimeseriesGenerator')
class TimeseriesGenerator(sequence.TimeseriesGenerator, data_utils.Sequence):
  """Utility class for generating batches of temporal data.

  This class takes in a sequence of data-points gathered at
  equal intervals, along with time series parameters such as
  stride, length of history, etc., to produce batches for
  training/validation.
  # Arguments
      data: Indexable generator (such as list or Numpy array)
          containing consecutive data points (timesteps).
          The data should be at 2D, and axis 0 is expected
          to be the time dimension.
      targets: Targets corresponding to timesteps in `data`.
          It should have same length as `data`.
      length: Length of the output sequences (in number of timesteps).
      sampling_rate: Period between successive individual timesteps
          within sequences. For rate `r`, timesteps
          `data[i]`, `data[i-r]`, ... `data[i - length]`
          are used for create a sample sequence.
      stride: Period between successive output sequences.
          For stride `s`, consecutive output samples would
          be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
      start_index: Data points earlier than `start_index` will not be used
          in the output sequences. This is useful to reserve part of the
          data for test or validation.
      end_index: Data points later than `end_index` will not be used
          in the output sequences. This is useful to reserve part of the
          data for test or validation.
      shuffle: Whether to shuffle output samples,
          or instead draw them in chronological order.
      reverse: Boolean: if `true`, timesteps in each output sample will be
          in reverse chronological order.
      batch_size: Number of timeseries samples in each batch
          (except maybe the last one).
  # Returns
      A [Sequence](/utils/#sequence) instance.
  # Examples
  ```python
  from keras.preprocessing.sequence import TimeseriesGenerator
  import numpy as np
  data = np.array([[i] for i in range(50)])
  targets = np.array([[i] for i in range(50)])
  data_gen = TimeseriesGenerator(data, targets,
                                 length=10, sampling_rate=2,
                                 batch_size=2)
  assert len(data_gen) == 20
  batch_0 = data_gen[0]
  x, y = batch_0
  assert np.array_equal(x,
                        np.array([[[0], [2], [4], [6], [8]],
                                  [[1], [3], [5], [7], [9]]]))
  assert np.array_equal(y,
                        np.array([[10], [11]]))
  ```
  """
  pass


@keras_export('keras.preprocessing.sequence.pad_sequences')
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
  """Pads sequences to the same length.

  This function transforms a list (of length `num_samples`)
  of sequences (lists of integers)
  into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
  `num_timesteps` is either the `maxlen` argument if provided,
  or the length of the longest sequence in the list.

  Sequences that are shorter than `num_timesteps`
  are padded with `value` until they are `num_timesteps` long.

  Sequences longer than `num_timesteps` are truncated
  so that they fit the desired length.

  The position where padding or truncation happens is determined by
  the arguments `padding` and `truncating`, respectively.
  Pre-padding or removing values from the beginning of the sequence is the
  default.

  >>> sequence = [[1], [2, 3], [4, 5, 6]]
  >>> tf.keras.preprocessing.sequence.pad_sequences(sequence)
  array([[0, 0, 1],
         [0, 2, 3],
         [4, 5, 6]], dtype=int32)

  >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, value=-1)
  array([[-1, -1,  1],
         [-1,  2,  3],
         [ 4,  5,  6]], dtype=int32)

  >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')
  array([[1, 0, 0],
         [2, 3, 0],
         [4, 5, 6]], dtype=int32)

  >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=2)
  array([[0, 1],
         [2, 3],
         [5, 6]], dtype=int32)

  Args:
      sequences: List of sequences (each sequence is a list of integers).
      maxlen: Optional Int, maximum length of all sequences. If not provided,
          sequences will be padded to the length of the longest individual
          sequence.
      dtype: (Optional, defaults to int32). Type of the output sequences.
          To pad sequences with variable length strings, you can use `object`.
      padding: String, 'pre' or 'post' (optional, defaults to 'pre'):
          pad either before or after each sequence.
      truncating: String, 'pre' or 'post' (optional, defaults to 'pre'):
          remove values from sequences larger than
          `maxlen`, either at the beginning or at the end of the sequences.
      value: Float or String, padding value. (Optional, defaults to 0.)

  Returns:
      Numpy array with shape `(len(sequences), maxlen)`

  Raises:
      ValueError: In case of invalid values for `truncating` or `padding`,
          or in case of invalid shape for a `sequences` entry.
  """
  return sequence.pad_sequences(
      sequences, maxlen=maxlen, dtype=dtype,
      padding=padding, truncating=truncating, value=value)

keras_export(
    'keras.preprocessing.sequence.make_sampling_table')(make_sampling_table)
keras_export('keras.preprocessing.sequence.skipgrams')(skipgrams)
