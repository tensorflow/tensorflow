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
"""Preprocessing utilities for sequence data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
  """Pads each sequence to the same length (length of the longest sequence).

  If maxlen is provided, any sequence longer
  than maxlen is truncated to maxlen.
  Truncation happens off either the beginning (default) or
  the end of the sequence.

  Supports post-padding and pre-padding (default).

  Arguments:
      sequences: list of lists where each element is a sequence
      maxlen: int, maximum length
      dtype: type to cast the resulting sequence.
      padding: 'pre' or 'post', pad either before or after each sequence.
      truncating: 'pre' or 'post', remove values from sequences larger than
          maxlen either in the beginning or in the end of the sequence
      value: float, value to pad the sequences to the desired value.

  Returns:
      x: numpy array with dimensions (number_of_sequences, maxlen)

  Raises:
      ValueError: in case of invalid values for `truncating` or `padding`,
          or in case of invalid shape for a `sequences` entry.
  """
  if not hasattr(sequences, '__len__'):
    raise ValueError('`sequences` must be iterable.')
  lengths = []
  for x in sequences:
    if not hasattr(x, '__len__'):
      raise ValueError('`sequences` must be a list of iterables. '
                       'Found non-iterable: ' + str(x))
    lengths.append(len(x))

  num_samples = len(sequences)
  if maxlen is None:
    maxlen = np.max(lengths)

  # take the sample shape from the first non empty sequence
  # checking for consistency in the main loop below.
  sample_shape = tuple()
  for s in sequences:
    if len(s) > 0:  # pylint: disable=g-explicit-length-test
      sample_shape = np.asarray(s).shape[1:]
      break

  x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
  for idx, s in enumerate(sequences):
    if not len(s):  # pylint: disable=g-explicit-length-test
      continue  # empty list/array was found
    if truncating == 'pre':
      trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
    elif truncating == 'post':
      trunc = s[:maxlen]
    else:
      raise ValueError('Truncating type "%s" not understood' % truncating)

    # check `trunc` has expected shape
    trunc = np.asarray(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
      raise ValueError(
          'Shape of sample %s of sequence at position %s is different from '
          'expected shape %s'
          % (trunc.shape[1:], idx, sample_shape))

    if padding == 'post':
      x[idx, :len(trunc)] = trunc
    elif padding == 'pre':
      x[idx, -len(trunc):] = trunc
    else:
      raise ValueError('Padding type "%s" not understood' % padding)
  return x


def make_sampling_table(size, sampling_factor=1e-5):
  """Generates a word rank-based probabilistic sampling table.

  This generates an array where the ith element
  is the probability that a word of rank i would be sampled,
  according to the sampling distribution used in word2vec.

  The word2vec formula is:
      p(word) = min(1, sqrt(word.frequency/sampling_factor) /
      (word.frequency/sampling_factor))

  We assume that the word frequencies follow Zipf's law (s=1) to derive
  a numerical approximation of frequency(rank):
     frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))
      where gamma is the Euler-Mascheroni constant.

  Arguments:
      size: int, number of possible words to sample.
      sampling_factor: the sampling factor in the word2vec formula.

  Returns:
      A 1D Numpy array of length `size` where the ith entry
      is the probability that a word of rank i should be sampled.
  """
  gamma = 0.577
  rank = np.array(list(range(size)))
  rank[0] = 1
  inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1. / (12. * rank)
  f = sampling_factor * inv_fq

  return np.minimum(1., f / np.sqrt(f))


def skipgrams(sequence,
              vocabulary_size,
              window_size=4,
              negative_samples=1.,
              shuffle=True,
              categorical=False,
              sampling_table=None):
  """Generates skipgram word pairs.

  Takes a sequence (list of indexes of words),
  returns couples of [word_index, other_word index] and labels (1s or 0s),
  where label = 1 if 'other_word' belongs to the context of 'word',
  and label=0 if 'other_word' is randomly sampled

  Arguments:
      sequence: a word sequence (sentence), encoded as a list
          of word indices (integers). If using a `sampling_table`,
          word indices are expected to match the rank
          of the words in a reference dataset (e.g. 10 would encode
          the 10-th most frequently occuring token).
          Note that index 0 is expected to be a non-word and will be skipped.
      vocabulary_size: int. maximum possible word index + 1
      window_size: int. actually half-window.
          The window of a word wi will be [i-window_size, i+window_size+1]
      negative_samples: float >= 0. 0 for no negative (=random) samples.
          1 for same number as positive samples. etc.
      shuffle: whether to shuffle the word couples before returning them.
      categorical: bool. if False, labels will be
          integers (eg. [0, 1, 1 .. ]),
          if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]
      sampling_table: 1D array of size `vocabulary_size` where the entry i
          encodes the probabibily to sample a word of rank i.

  Returns:
      couples, labels: where `couples` are int pairs and
          `labels` are either 0 or 1.

  # Note
      By convention, index 0 in the vocabulary is
      a non-word and will be skipped.
  """
  couples = []
  labels = []
  for i, wi in enumerate(sequence):
    if not wi:
      continue
    if sampling_table is not None:
      if sampling_table[wi] < random.random():
        continue

    window_start = max(0, i - window_size)
    window_end = min(len(sequence), i + window_size + 1)
    for j in range(window_start, window_end):
      if j != i:
        wj = sequence[j]
        if not wj:
          continue
        couples.append([wi, wj])
        if categorical:
          labels.append([0, 1])
        else:
          labels.append(1)

  if negative_samples > 0:
    num_negative_samples = int(len(labels) * negative_samples)
    words = [c[0] for c in couples]
    random.shuffle(words)

    couples += [[words[i % len(words)], random.randint(1, vocabulary_size - 1)]
                for i in range(num_negative_samples)]
    if categorical:
      labels += [[1, 0]] * num_negative_samples
    else:
      labels += [0] * num_negative_samples

  if shuffle:
    seed = random.randint(0, 10e6)
    random.seed(seed)
    random.shuffle(couples)
    random.seed(seed)
    random.shuffle(labels)

  return couples, labels
