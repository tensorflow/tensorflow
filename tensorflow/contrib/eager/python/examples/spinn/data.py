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
"""Utilities of SNLI data and GloVe word vectors for SPINN model.

See more details about the SNLI data set at:
  https://nlp.stanford.edu/projects/snli/

See more details about the GloVe pretrained word embeddings at:
  https://nlp.stanford.edu/projects/glove/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os
import random

import numpy as np

POSSIBLE_LABELS = ("entailment", "contradiction", "neutral")

UNK_CODE = 0   # Code for unknown word tokens.
PAD_CODE = 1   # Code for padding tokens.

SHIFT_CODE = 3
REDUCE_CODE = 2

WORD_VECTOR_LEN = 300  # Embedding dimensions.

LEFT_PAREN = "("
RIGHT_PAREN = ")"
PARENTHESES = (LEFT_PAREN, RIGHT_PAREN)


def get_non_parenthesis_words(items):
  """Get the non-parenthesis items from a SNLI parsed sentence.

  Args:
    items: Data items from a parsed SNLI sentence, with parentheses. E.g.,
      ["(", "Man", "(", "(", "(", "(", "(", "wearing", "pass", ")", ...

  Returns:
    A list of non-parentheses word items, all converted to lower case. E.g.,
      ["man", "wearing", "pass", ...
  """
  return [x.lower() for x in items if x not in PARENTHESES and x]


def get_shift_reduce(items):
  """Obtain shift-reduce vector from a list of items from the SNLI data.

  Args:
    items: Data items as a list of str, e.g.,
       ["(", "Man", "(", "(", "(", "(", "(", "wearing", "pass", ")", ...

  Returns:
    A list of shift-reduce transitions, encoded as `SHIFT_CODE` for shift and
      `REDUCE_CODE` for reduce. See code above for the values of `SHIFT_CODE`
      and `REDUCE_CODE`.
  """
  trans = []
  for item in items:
    if item == LEFT_PAREN:
      continue
    elif item == RIGHT_PAREN:
      trans.append(REDUCE_CODE)
    else:
      trans.append(SHIFT_CODE)
  return trans


def pad_and_reverse_word_ids(sentences):
  """Pad a list of sentences to the common maximum length + 1.

  Args:
    sentences: A list of sentences as a list of list of integers. Each integer
      is a word ID. Each list of integer corresponds to one sentence.

  Returns:
    A numpy.ndarray of shape (num_sentences, max_length + 1), wherein max_length
      is the maximum sentence length (in # of words). Each sentence is reversed
      and then padded with an extra one at head, as required by the model.
  """
  max_len = max(len(sent) for sent in sentences)
  for sent in sentences:
    if len(sent) < max_len:
      sent.extend([PAD_CODE] * (max_len - len(sent)))
  # Reverse in time order and pad an extra one.
  sentences = np.fliplr(np.array(sentences, dtype=np.int64))
  sentences = np.concatenate(
      [np.ones([sentences.shape[0], 1], dtype=np.int64), sentences], axis=1)
  return sentences


def pad_transitions(sentences_transitions):
  """Pad a list of shift-reduce transitions to the maximum length."""
  max_len = max(len(transitions) for transitions in sentences_transitions)
  for transitions in sentences_transitions:
    if len(transitions) < max_len:
      transitions.extend([PAD_CODE] * (max_len - len(transitions)))
  return np.array(sentences_transitions, dtype=np.int64)


def load_vocabulary(data_root):
  """Load vocabulary from SNLI data files.

  Args:
    data_root: Root directory of the data. It is assumed that the SNLI data
      files have been downloaded and extracted to the "snli/snli_1.0"
      subdirectory of it.

  Returns:
    Vocabulary as a set of strings.

  Raises:
    ValueError: If SNLI data files cannot be found.
  """
  snli_path = os.path.join(data_root, "snli")
  snli_glob_pattern = os.path.join(snli_path, "snli_1.0/snli_1.0_*.txt")
  file_names = glob.glob(snli_glob_pattern)
  if not file_names:
    raise ValueError(
        "Cannot find SNLI data files at %s. "
        "Please download and extract SNLI data first." % snli_glob_pattern)

  print("Loading vocabulary...")
  vocab = set()
  for file_name in file_names:
    with open(os.path.join(snli_path, file_name), "rt") as f:
      for i, line in enumerate(f):
        if i == 0:
          continue
        items = line.split("\t")
        premise_words = get_non_parenthesis_words(items[1].split(" "))
        hypothesis_words = get_non_parenthesis_words(items[2].split(" "))
        vocab.update(premise_words)
        vocab.update(hypothesis_words)
  return vocab


def load_word_vectors(data_root, vocab):
  """Load GloVe word vectors for words present in the vocabulary.

  Args:
    data_root: Data root directory. It is assumed that the GloVe file
     has been downloaded and extracted at the "glove/" subdirectory of it.
    vocab: A `set` of words, representing the vocabulary.

  Returns:
    1. word2index: A dict from lower-case word to row index in the embedding
       matrix, i.e, `embed` below.
    2. embed: The embedding matrix as a float32 numpy array. Its shape is
       [vocabulary_size, WORD_VECTOR_LEN]. vocabulary_size is len(vocab).
       WORD_VECTOR_LEN is the embedding dimension (300).

  Raises:
    ValueError: If GloVe embedding file cannot be found.
  """
  glove_path = os.path.join(data_root, "glove/glove.42B.300d.txt")
  if not os.path.isfile(glove_path):
    raise ValueError(
        "Cannot find GloVe embedding file at %s. "
        "Please download and extract GloVe embeddings first." % glove_path)

  print("Loading word vectors...")

  word2index = dict()
  embed = []

  embed.append([0] * WORD_VECTOR_LEN)  # <unk>
  embed.append([0] * WORD_VECTOR_LEN)  # <pad>
  word2index["<unk>"] = UNK_CODE
  word2index["<pad>"] = PAD_CODE

  with open(glove_path, "rt") as f:
    for line in f:
      items = line.split(" ")
      word = items[0]
      if word in vocab and word not in word2index:
        word2index[word] = len(embed)
        vector = np.array([float(item) for item in items[1:]])
        assert (WORD_VECTOR_LEN,) == vector.shape
        embed.append(vector)
  embed = np.array(embed, dtype=np.float32)
  return word2index, embed


def calculate_bins(length2count, min_bin_size):
  """Calculate bin boundaries given a histogram of lengths and minimum bin size.

  Args:
    length2count: A `dict` mapping length to sentence count.
    min_bin_size: Minimum bin size in terms of total number of sentence pairs
      in the bin.

  Returns:
    A `list` representing the right bin boundaries, starting from the inclusive
    right boundary of the first bin. For example, if the output is
      [10, 20, 35],
    it means there are three bins: [1, 10], [11, 20] and [21, 35].
  """
  bounds = []
  lengths = sorted(length2count.keys())
  cum_count = 0
  for length in lengths:
    cum_count += length2count[length]
    if cum_count >= min_bin_size:
      bounds.append(length)
      cum_count = 0
  if bounds[-1] != lengths[-1]:
    bounds.append(lengths[-1])
  return bounds


class SnliData(object):
  """A split of SNLI data."""

  def __init__(self, data_file, word2index, sentence_len_limit=-1):
    """SnliData constructor.

    Args:
      data_file: Full path to the data file, e.g.,
        "/tmp/spinn-data/snli/snli_1.0/snli_1.0.train.txt"
      word2index: A dict from lower-case word to row index in the embedding
        matrix (see `load_word_vectors()` for details).
      sentence_len_limit: Maximum allowed sentence length (# of words).
        A value of <= 0 means unlimited. Sentences longer than this limit
        are currently discarded, not truncated.
    """

    self._labels = []
    self._premises = []
    self._premise_transitions = []
    self._hypotheses = []
    self._hypothesis_transitions = []

    with open(data_file, "rt") as f:
      for i, line in enumerate(f):
        if i == 0:
          # Skip header line.
          continue
        items = line.split("\t")
        if items[0] not in POSSIBLE_LABELS:
          continue

        premise_items = items[1].split(" ")
        hypothesis_items = items[2].split(" ")
        premise_words = get_non_parenthesis_words(premise_items)
        hypothesis_words = get_non_parenthesis_words(hypothesis_items)

        if (sentence_len_limit > 0 and
            (len(premise_words) > sentence_len_limit or
             len(hypothesis_words) > sentence_len_limit)):
          # TODO(cais): Maybe truncate; do not discard.
          continue

        premise_ids = [
            word2index.get(word, UNK_CODE) for word in premise_words]
        hypothesis_ids = [
            word2index.get(word, UNK_CODE) for word in hypothesis_words]

        self._premises.append(premise_ids)
        self._hypotheses.append(hypothesis_ids)
        self._premise_transitions.append(get_shift_reduce(premise_items))
        self._hypothesis_transitions.append(get_shift_reduce(hypothesis_items))
        assert (len(self._premise_transitions[-1]) ==
                2 * len(premise_words) - 1)
        assert (len(self._hypothesis_transitions[-1]) ==
                2 * len(hypothesis_words) - 1)

        self._labels.append(POSSIBLE_LABELS.index(items[0]) + 1)

    assert len(self._labels) == len(self._premises)
    assert len(self._labels) == len(self._hypotheses)
    assert len(self._labels) == len(self._premise_transitions)
    assert len(self._labels) == len(self._hypothesis_transitions)

  def num_batches(self, batch_size):
    """Calculate number of batches given batch size."""
    return int(math.ceil(len(self._labels) / batch_size))

  def get_generator(self, batch_size):
    """Obtain a generator for batched data.

    All examples of this SnliData object are randomly shuffled, sorted
    according to the maximum sentence length of the premise and hypothesis
    sentences in the pair, and batched.

    Args:
      batch_size: Desired batch size.

    Returns:
      A generator for data batches. The generator yields a 5-tuple:
        label: An array of the shape (batch_size,).
        premise: An array of the shape (max_premise_len, batch_size), wherein
          max_premise_len is the maximum length of the (padded) premise
          sentence in the batch.
        premise_transitions: An array of the shape (2 * max_premise_len -3,
          batch_size).
        hypothesis: Same as `premise`, but for hypothesis sentences.
        hypothesis_transitions: Same as `premise_transitions`, but for
          hypothesis sentences.
      All the elements of the 5-tuple have dtype `int64`.
    """
    # Randomly shuffle examples.
    zipped = list(zip(
        self._labels, self._premises, self._premise_transitions,
        self._hypotheses, self._hypothesis_transitions))
    random.shuffle(zipped)
    # Then sort the examples by maximum of the premise and hypothesis sentence
    # lengths in the pair. During training, the batches are expected to be
    # shuffled. So it is okay to leave them sorted by max length here.
    (labels, premises, premise_transitions, hypotheses,
     hypothesis_transitions) = zip(
         *sorted(zipped, key=lambda x: max(len(x[1]), len(x[3]))))

    def _generator():
      begin = 0
      while begin < len(labels):
        # The sorting above and the batching here makes sure that sentences of
        # similar max lengths are batched together, minimizing the inefficiency
        # due to uneven max lengths. The sentences are batched differently in
        # each call to get_generator() due to the shuffling before sorting
        # above. The pad_and_reverse_word_ids() and pad_transitions() functions
        # take care of any remaining unevenness of the max sentence lengths.
        end = min(begin + batch_size, len(labels))
        # Transpose, because the SPINN model requires time-major, instead of
        # batch-major.
        yield (labels[begin:end],
               pad_and_reverse_word_ids(premises[begin:end]).T,
               pad_transitions(premise_transitions[begin:end]).T,
               pad_and_reverse_word_ids(hypotheses[begin:end]).T,
               pad_transitions(hypothesis_transitions[begin:end]).T)
        begin = end
    return _generator
