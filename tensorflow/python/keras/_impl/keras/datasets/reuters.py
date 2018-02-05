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
"""Reuters topic classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np

from tensorflow.python.keras._impl.keras.preprocessing.sequence import _remove_long_seq
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.platform import tf_logging as logging


def load_data(path='reuters.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              test_split=0.2,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3,
              **kwargs):
  """Loads the Reuters newswire classification dataset.

  Arguments:
      path: where to cache the data (relative to `~/.keras/dataset`).
      num_words: max number of words to include. Words are ranked
          by how often they occur (in the training set) and only
          the most frequent words are kept
      skip_top: skip the top N most frequently occurring words
          (which may not be informative).
      maxlen: truncate sequences after this length.
      test_split: Fraction of the dataset to be used as test data.
      seed: random seed for sample shuffling.
      start_char: The start of a sequence will be marked with this character.
          Set to 1 because 0 is usually the padding character.
      oov_char: words that were cut out because of the `num_words`
          or `skip_top` limit will be replaced with this character.
      index_from: index actual words with this index and higher.
      **kwargs: Used for backwards compatibility.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  Note that the 'out of vocabulary' character is only used for
  words that were present in the training set but are not included
  because they're not making the `num_words` cut here.
  Words that were not seen in the training set but are in the test set
  have simply been skipped.
  """
  # Legacy support
  if 'nb_words' in kwargs:
    logging.warning('The `nb_words` argument in `load_data` '
                    'has been renamed `num_words`.')
    num_words = kwargs.pop('nb_words')
  if kwargs:
    raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

  path = get_file(
      path,
      origin='https://s3.amazonaws.com/text-datasets/reuters.npz',
      file_hash='87aedbeb0cb229e378797a632c1997b6')
  with np.load(path) as f:
    xs, labels = f['x'], f['y']

  np.random.seed(seed)
  indices = np.arange(len(xs))
  np.random.shuffle(indices)
  xs = xs[indices]
  labels = labels[indices]

  if start_char is not None:
    xs = [[start_char] + [w + index_from for w in x] for x in xs]
  elif index_from:
    xs = [[w + index_from for w in x] for x in xs]

  if maxlen:
    xs, labels = _remove_long_seq(maxlen, xs, labels)

  if not num_words:
    num_words = max([max(x) for x in xs])

  # by convention, use 2 as OOV word
  # reserve 'index_from' (=3 by default) characters:
  # 0 (padding), 1 (start), 2 (OOV)
  if oov_char is not None:
    xs = [[w if skip_top <= w < num_words else oov_char for w in x] for x in xs]
  else:
    xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

  idx = int(len(xs) * (1 - test_split))
  x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
  x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

  return (x_train, y_train), (x_test, y_test)


def get_word_index(path='reuters_word_index.json'):
  """Retrieves the dictionary mapping word indices back to words.

  Arguments:
      path: where to cache the data (relative to `~/.keras/dataset`).

  Returns:
      The word index dictionary.
  """
  path = get_file(
      path,
      origin='https://s3.amazonaws.com/text-datasets/reuters_word_index.json',
      file_hash='4d44cc38712099c9e383dc6e5f11a921')
  f = open(path)
  data = json.load(f)
  f.close()
  return data
