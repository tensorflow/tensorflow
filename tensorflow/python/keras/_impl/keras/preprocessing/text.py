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
"""Utilities for text input preprocessing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from hashlib import md5
import string
import sys

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


if sys.version_info < (3,):
  maketrans = string.maketrans
else:
  maketrans = str.maketrans


@tf_export('keras.preprocessing.text.text_to_word_sequence')
def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=' '):
  """Converts a text to a sequence of words (or tokens).

  Arguments:
      text: Input text (string).
      filters: Sequence of characters to filter out.
      lower: Whether to convert the input to lowercase.
      split: Sentence split marker (string).

  Returns:
      A list of words (or tokens).
  """
  if lower:
    text = text.lower()

  if sys.version_info < (3,) and isinstance(text, unicode):
    translate_map = dict((ord(c), unicode(split)) for c in filters)
  else:
    translate_map = maketrans(filters, split * len(filters))

  text = text.translate(translate_map)
  seq = text.split(split)
  return [i for i in seq if i]


@tf_export('keras.preprocessing.text.one_hot')
def one_hot(text,
            n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
  """One-hot encodes a text into a list of word indexes of size n.

  This is a wrapper to the `hashing_trick` function using `hash` as the
  hashing function; unicity of word to index mapping non-guaranteed.

  Arguments:
      text: Input text (string).
      n: Dimension of the hashing space.
      filters: Sequence of characters to filter out.
      lower: Whether to convert the input to lowercase.
      split: Sentence split marker (string).

  Returns:
      A list of integer word indices (unicity non-guaranteed).
  """
  return hashing_trick(
      text, n, hash_function=hash, filters=filters, lower=lower, split=split)


def hashing_trick(text,
                  n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
  """Converts a text to a sequence of indexes in a fixed-size hashing space.

  Arguments:
      text: Input text (string).
      n: Dimension of the hashing space.
      hash_function: if `None` uses python `hash` function, can be 'md5' or
          any function that takes in input a string and returns a int.
          Note that `hash` is not a stable hashing function, so
          it is not consistent across different runs, while 'md5'
          is a stable hashing function.
      filters: Sequence of characters to filter out.
      lower: Whether to convert the input to lowercase.
      split: Sentence split marker (string).

  Returns:
      A list of integer word indices (unicity non-guaranteed).

  `0` is a reserved index that won't be assigned to any word.

  Two or more words may be assigned to the same index, due to possible
  collisions by the hashing function.
  The
  probability
  of a collision is in relation to the dimension of the hashing space and
  the number of distinct objects.
  """
  if hash_function is None:
    hash_function = hash
  elif hash_function == 'md5':
    hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

  seq = text_to_word_sequence(text, filters=filters, lower=lower, split=split)
  return [(hash_function(w) % (n - 1) + 1) for w in seq]


@tf_export('keras.preprocessing.text.Tokenizer')
class Tokenizer(object):
  """Text tokenization utility class.

  This class allows to vectorize a text corpus, by turning each
  text into either a sequence of integers (each integer being the index
  of a token in a dictionary) or into a vector where the coefficient
  for each token could be binary, based on word count, based on tf-idf...

  Arguments:
      num_words: the maximum number of words to keep, based
          on word frequency. Only the most common `num_words` words will
          be kept.
      filters: a string where each element is a character that will be
          filtered from the texts. The default is all punctuation, plus
          tabs and line breaks, minus the `'` character.
      lower: boolean. Whether to convert the texts to lowercase.
      split: character or string to use for token splitting.
      char_level: if True, every character will be treated as a token.
      oov_token: if given, it will be added to word_index and used to
          replace out-of-vocabulary words during text_to_sequence calls

  By default, all punctuation is removed, turning the texts into
  space-separated sequences of words
  (words maybe include the `'` character). These sequences are then
  split into lists of tokens. They will then be indexed or vectorized.

  `0` is a reserved index that won't be assigned to any word.
  """

  def __init__(self,
               num_words=None,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=' ',
               char_level=False,
               oov_token=None,
               **kwargs):
    # Legacy support
    if 'nb_words' in kwargs:
      logging.warning('The `nb_words` argument in `Tokenizer` '
                      'has been renamed `num_words`.')
      num_words = kwargs.pop('nb_words')
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    self.word_counts = OrderedDict()
    self.word_docs = {}
    self.filters = filters
    self.split = split
    self.lower = lower
    self.num_words = num_words
    self.document_count = 0
    self.char_level = char_level
    self.oov_token = oov_token

  def fit_on_texts(self, texts):
    """Updates internal vocabulary based on a list of texts.

    Required before using `texts_to_sequences` or `texts_to_matrix`.

    Arguments:
        texts: can be a list of strings,
            or a generator of strings (for memory-efficiency)
    """
    self.document_count = 0
    for text in texts:
      self.document_count += 1
      seq = text if self.char_level else text_to_word_sequence(
          text, self.filters, self.lower, self.split)
      for w in seq:
        if w in self.word_counts:
          self.word_counts[w] += 1
        else:
          self.word_counts[w] = 1
      for w in set(seq):
        if w in self.word_docs:
          self.word_docs[w] += 1
        else:
          self.word_docs[w] = 1

    wcounts = list(self.word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    # note that index 0 is reserved, never assigned to an existing word
    self.word_index = dict(
        list(zip(sorted_voc, list(range(1,
                                        len(sorted_voc) + 1)))))

    if self.oov_token is not None:
      i = self.word_index.get(self.oov_token)
      if i is None:
        self.word_index[self.oov_token] = len(self.word_index) + 1

    self.index_docs = {}
    for w, c in list(self.word_docs.items()):
      self.index_docs[self.word_index[w]] = c

  def fit_on_sequences(self, sequences):
    """Updates internal vocabulary based on a list of sequences.

    Required before using `sequences_to_matrix`
    (if `fit_on_texts` was never called).

    Arguments:
        sequences: A list of sequence.
            A "sequence" is a list of integer word indices.
    """
    self.document_count = len(sequences)
    self.index_docs = {}
    for seq in sequences:
      seq = set(seq)
      for i in seq:
        if i not in self.index_docs:
          self.index_docs[i] = 1
        else:
          self.index_docs[i] += 1

  def texts_to_sequences(self, texts):
    """Transforms each text in texts in a sequence of integers.

    Only top "num_words" most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.

    Arguments:
        texts: A list of texts (strings).

    Returns:
        A list of sequences.
    """
    res = []
    for vect in self.texts_to_sequences_generator(texts):
      res.append(vect)
    return res

  def texts_to_sequences_generator(self, texts):
    """Transforms each text in texts in a sequence of integers.

    Only top "num_words" most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.

    Arguments:
        texts: A list of texts (strings).

    Yields:
        Yields individual sequences.
    """
    num_words = self.num_words
    for text in texts:
      seq = text if self.char_level else text_to_word_sequence(
          text, self.filters, self.lower, self.split)
      vect = []
      for w in seq:
        i = self.word_index.get(w)
        if i is not None:
          if num_words and i >= num_words:
            continue
          else:
            vect.append(i)
        elif self.oov_token is not None:
          i = self.word_index.get(self.oov_token)
          if i is not None:
            vect.append(i)
      yield vect

  def texts_to_matrix(self, texts, mode='binary'):
    """Convert a list of texts to a Numpy matrix.

    Arguments:
        texts: list of strings.
        mode: one of "binary", "count", "tfidf", "freq".

    Returns:
        A Numpy matrix.
    """
    sequences = self.texts_to_sequences(texts)
    return self.sequences_to_matrix(sequences, mode=mode)

  def sequences_to_matrix(self, sequences, mode='binary'):
    """Converts a list of sequences into a Numpy matrix.

    Arguments:
        sequences: list of sequences
            (a sequence is a list of integer word indices).
        mode: one of "binary", "count", "tfidf", "freq"

    Returns:
        A Numpy matrix.

    Raises:
        ValueError: In case of invalid `mode` argument,
            or if the Tokenizer requires to be fit to sample data.
    """
    if not self.num_words:
      if self.word_index:
        num_words = len(self.word_index) + 1
      else:
        raise ValueError('Specify a dimension (num_words argument), '
                         'or fit on some text data first.')
    else:
      num_words = self.num_words

    if mode == 'tfidf' and not self.document_count:
      raise ValueError('Fit the Tokenizer on some data '
                       'before using tfidf mode.')

    x = np.zeros((len(sequences), num_words))
    for i, seq in enumerate(sequences):
      if not seq:
        continue
      counts = {}
      for j in seq:
        if j >= num_words:
          continue
        if j not in counts:
          counts[j] = 1.
        else:
          counts[j] += 1
      for j, c in list(counts.items()):
        if mode == 'count':
          x[i][j] = c
        elif mode == 'freq':
          x[i][j] = c / len(seq)
        elif mode == 'binary':
          x[i][j] = 1
        elif mode == 'tfidf':
          # Use weighting scheme 2 in
          # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
          tf = 1 + np.log(c)
          idf = np.log(1 + self.document_count /
                       (1 + self.index_docs.get(j, 0)))
          x[i][j] = tf * idf
        else:
          raise ValueError('Unknown vectorization mode:', mode)
    return x
