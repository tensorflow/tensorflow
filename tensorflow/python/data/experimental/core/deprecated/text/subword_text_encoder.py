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

# coding=utf-8
"""SubwordTextEncoder."""
# This implementation is based on SubwordTextEncoder in Tensor2Tensor,
# originally written by Noam Shazeer (GitHub: nshazeer).
from __future__ import unicode_literals

import collections

from absl import logging
import six
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core.deprecated.text import text_encoder

# Internally, an underscore indicates a single space, so, to ensure
# user-supplied underscores are encoded properly, they are replaced with this
# string during encoding.
_UNDERSCORE_REPLACEMENT = "\\&undsc"


class SubwordTextEncoder(text_encoder.TextEncoder):
  """Invertible `TextEncoder` using word pieces with a byte-level fallback.

  Encoding is fully invertible because all out-of-vocab wordpieces are
  byte-encoded.

  The vocabulary is "trained" on a corpus and all wordpieces are stored in a
  vocabulary file. To generate a vocabulary from a corpus, use
  `tfds.deprecated.text.SubwordTextEncoder.build_from_corpus`.

  Typical usage:

  ```
  # Build
  encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      corpus_generator, target_vocab_size=2**15)
  encoder.save_to_file(vocab_fname)

  # Load
  encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(vocab_fname)
  ids = encoder.encode("hello world")
  text = encoder.decode([1, 2, 3, 4])
  ```
  """

  def __init__(self, vocab_list=None):
    r"""Constructs a SubwordTextEncoder from a vocabulary list.

    Note: To generate a vocabulary from a corpus, use
    `tfds.deprecated.text.SubwordTextEncoder.build_from_corpus`.

    Args:
      vocab_list: `list<str>`, list of subwords for the vocabulary. Note that an
        underscore at the end of a subword indicates the end of the word (i.e. a
        space will be inserted afterwards when decoding). Underscores in the
        interior of subwords are disallowed and should use the underscore
        escape sequence.
    """
    self._init_from_list(vocab_list)

  def encode(self, s):
    """Encodes text into a list of integers."""
    s = tf.compat.as_text(s)
    tokens = self._tokenizer.tokenize(s)
    tokens = _prepare_tokens_for_encode(tokens)
    ids = []
    for token in tokens:
      ids.extend(self._token_to_ids(token))
    return text_encoder.pad_incr(ids)

  def decode(self, ids):
    """Decodes a list of integers into text."""
    ids = text_encoder.pad_decr(ids)
    subword_ids = ids
    del ids

    subwords = []

    # Some ids correspond to bytes. Because unicode characters are composed of
    # possibly multiple bytes, we attempt to decode contiguous lists of bytes
    # all together. Invalid byte sequences are replaced with the unicode
    # replacement (i.e. unknown) character U+FFFD.
    prev_bytes = []

    def consume_prev_bytes():
      if prev_bytes:
        bytestr = b"".join(prev_bytes)
        bytes_text = bytestr.decode("utf-8", "replace")
        subwords.append(bytes_text)
      return []

    for subword_id in subword_ids:
      subword = self._id_to_subword(subword_id)
      if isinstance(subword, six.binary_type):
        # Byte-encoded
        prev_bytes.append(subword)
      else:
        # If there were bytes previously, convert to unicode.
        prev_bytes = consume_prev_bytes()
        trimmed, add_space = _trim_underscore_and_tell(subword)
        subwords.append(trimmed)
        if add_space:
          subwords.append(" ")
    # If there were trailing bytes, convert to unicode.
    prev_bytes = consume_prev_bytes()

    return tf.compat.as_text("".join(subwords))

  @property
  def vocab_size(self):
    # Vocab is:
    # * pad=0
    # * subwords
    # * bytes
    return 1 + len(self._subwords) + text_encoder.NUM_BYTES

  @property
  def subwords(self):
    return list(self._subwords)

  def _token_to_ids(self, token):
    """Convert a single token to a list of integer ids."""
    # Check cache
    cache_location = hash(token) % self._cache_size
    cache_key, cache_value = self._token_to_ids_cache[cache_location]
    if cache_key == token:
      return cache_value

    subwords = self._token_to_subwords(token)
    ids = []
    for subword in subwords:
      if subword == _UNDERSCORE_REPLACEMENT:
        ids.append(len(self._subwords) + ord("_"))
        continue
      subword_id = self._subword_to_id.get(subword)
      if subword_id is None:
        # Byte-encode
        ids.extend(self._byte_encode(subword))
      else:
        ids.append(subword_id)

    # Update cache
    self._token_to_ids_cache[cache_location] = (token, ids)

    return ids

  def _byte_encode(self, token):
    """Encode a single token byte-wise into integer ids."""
    # Vocab ids for all bytes follow ids for the subwords
    offset = len(self._subwords)
    if token == "_":
      return [len(self._subwords) + ord(" ")]
    return [i + offset for i in list(bytearray(tf.compat.as_bytes(token)))]

  def _id_to_subword(self, subword_id):
    """Converts a subword integer ID to a subword string."""
    if subword_id < 0 or subword_id >= (self.vocab_size - 1):
      raise ValueError("Received id %d which is invalid. Ids must be within "
                       "[0, %d)." % (subword_id + 1, self.vocab_size))

    if 0 <= subword_id < len(self._subwords):
      # Subword
      return self._subwords[subword_id]
    else:
      # Byte
      offset = len(self._subwords)
      subword_id -= offset
      bytestr = bytes(bytearray([subword_id]))
      return bytestr

  def _token_to_subwords(self, token):
    """Greedily split token into subwords."""
    subwords = []

    start = 0
    while start < len(token):
      subword = None
      for end in range(
          min(len(token), start + self._max_subword_len), start, -1):
        candidate = token[start:end]
        if (candidate in self._subword_to_id or
            candidate == _UNDERSCORE_REPLACEMENT):
          subword = candidate
          subwords.append(subword)
          start = end
          break
      # No subword match found. Consume a single (unicode) character.
      if subword is None:
        subwords.append(token[start])
        start += 1

    return subwords

  def _init_from_list(self, subwords):
    """Initializes the encoder from a list of subwords."""
    subwords = [tf.compat.as_text(s) for s in subwords if s]
    self._subwords = subwords
    # Note that internally everything is 0-indexed. Padding is dealt with at the
    # end of encode and the beginning of decode.
    self._subword_to_id = {s: i for i, s in enumerate(subwords)}

    # We remember the maximum length of any subword to avoid having to
    # check arbitrarily long strings.
    self._max_subword_len = max(
        len(_UNDERSCORE_REPLACEMENT), max([len(s) for s in subwords] or [1]))

    # Initialize the cache
    self._cache_size = 2**20
    self._token_to_ids_cache = [(None, None)] * self._cache_size

    # Setup tokenizer
    # Reserved tokens are all tokens that are mixed alphanum and non-alphanum.
    reserved_tokens = set([_UNDERSCORE_REPLACEMENT])
    for t in self._subwords:
      if text_encoder.is_mixed_alphanum(t):
        reserved_tokens.add(t)
    self._tokenizer = text_encoder.Tokenizer(
        alphanum_only=False, reserved_tokens=reserved_tokens)

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + ".subwords"

  def save_to_file(self, filename_prefix):
    """Save the vocabulary to a file."""
    # Wrap in single quotes to make it easier to see the full subword when
    # it has spaces and make it easier to search with ctrl+f.
    filename = self._filename(filename_prefix)
    lines = ["'%s'" % s for s in self._subwords]
    self._write_lines_to_file(filename, lines)

  @classmethod
  def load_from_file(cls, filename_prefix):
    """Extracts list of subwords from file."""
    filename = cls._filename(filename_prefix)
    lines, _ = cls._read_lines_from_file(filename)
    # Strip wrapping single quotes
    vocab_list = [line[1:-1] for line in lines]
    return cls(vocab_list=vocab_list)

  @classmethod
  def build_from_corpus(cls,
                        corpus_generator,
                        target_vocab_size,
                        max_subword_length=20,
                        max_corpus_chars=None,
                        reserved_tokens=None):
    """Builds a `SubwordTextEncoder` based on the `corpus_generator`.

    Args:
      corpus_generator: generator yielding `str`, from which subwords will be
        constructed.
      target_vocab_size: `int`, approximate size of the vocabulary to create.
      max_subword_length: `int`, maximum length of a subword. Note that memory
        and compute scale quadratically in the length of the longest token.
      max_corpus_chars: `int`, the maximum number of characters to consume from
        `corpus_generator` for the purposes of building the subword vocabulary.
      reserved_tokens: `list<str>`, list of tokens that will always be treated
        as whole tokens and not split up. Note that these must contain a mix of
        alphanumeric and non-alphanumeric characters (e.g. "<EOS>") and not end
        in an underscore.

    Returns:
      `SubwordTextEncoder`.
    """
    reserved_tokens = reserved_tokens or []
    _validate_build_arguments(
        max_subword_length=max_subword_length,
        reserved_tokens=reserved_tokens,
        target_vocab_size=target_vocab_size)
    token_counts = _token_counts_from_generator(
        generator=corpus_generator,
        max_chars=max_corpus_chars,
        reserved_tokens=reserved_tokens)

    # Binary search on the minimum token count to build a vocabulary with
    # approximately the right size
    def _binary_search(min_token_count, max_token_count):
      """Binary search min_token_count to build SubwordTextEncoder vocab."""
      candidate_min = (min_token_count + max_token_count) // 2
      logging.info("SubwordTextEncoder build: trying min_token_count %d",
                   candidate_min)
      encoder = cls._build_from_token_counts(
          token_counts=token_counts,
          min_token_count=candidate_min,
          reserved_tokens=reserved_tokens,
          num_iterations=4,
          max_subword_length=max_subword_length)
      vocab_size = encoder.vocab_size

      # Being within 1% of the target vocab size is ok
      target_achieved = (
          abs(vocab_size - target_vocab_size) * 100 < target_vocab_size)
      if (target_achieved or min_token_count >= max_token_count or
          candidate_min <= 1):
        # Search complete
        return encoder

      # Recurse
      if vocab_size > target_vocab_size:
        next_encoder = _binary_search(candidate_min + 1, max_token_count)
      else:
        next_encoder = _binary_search(min_token_count, candidate_min - 1)

      # Return the one that's closest to the target_vocab_size
      if (abs(vocab_size - target_vocab_size) <
          abs(next_encoder.vocab_size - target_vocab_size)):
        return encoder
      else:
        return next_encoder

    # Get min and max token counts.
    min_token_count = max(min(token_counts.values()), 1)
    max_token_count = max(token_counts.values())

    # Another option could be to do a binary search over *ranks* of the tokens.
    return _binary_search(min_token_count, max_token_count)

  @classmethod
  def _build_from_token_counts(cls, token_counts, min_token_count,
                               reserved_tokens, num_iterations,
                               max_subword_length):
    # Start with subwords initialized to only reserved_tokens
    subwords = list(reserved_tokens)

    for _ in range(num_iterations):
      encoder = cls(vocab_list=subwords)
      subword_counts = collections.defaultdict(int)
      for token, count in six.iteritems(token_counts):
        start_idx = 0
        for subword in encoder._token_to_subwords(token):  # pylint: disable=protected-access
          last_idx = min(len(token), start_idx + max_subword_length)
          for end_idx in range(start_idx + 1, last_idx + 1):
            candidate_subword = token[start_idx:end_idx]
            subword_counts[candidate_subword] += count
          start_idx += len(subword)

      # Group subword candidates by length and filter bad candidates
      len_to_subwords = [set() for _ in range(max_subword_length + 1)]
      for subword, count in six.iteritems(subword_counts):
        if count < min_token_count:
          continue
        # Skip single bytes because they're always in the vocab
        if len(tf.compat.as_bytes(subword)) <= 1:
          continue
        len_to_subwords[len(subword)].add(subword)

      # Consider subword candidates by descending length so that if a longer
      # subword is accepted, its prefixes can have their counts decremented.
      candidate_subwords = []
      for subword_len in reversed(range(max_subword_length + 1)):
        for subword in len_to_subwords[subword_len]:
          count = subword_counts[subword]
          if count < min_token_count:
            continue
          candidate_subwords.append((count, subword))
          # Decrement prefix counts
          for end_idx in range(1, subword_len):
            subword_counts[subword[:end_idx]] -= count

      # Sort subwords by count in descending order, keeping reserved_tokens as
      # the beginning.
      candidate_subwords.sort(reverse=True)
      subwords = reserved_tokens + [s for _, s in candidate_subwords]

    return cls(vocab_list=subwords)


def _token_counts_from_generator(generator, max_chars, reserved_tokens):
  """Builds token counts from generator."""
  reserved_tokens = list(reserved_tokens) + [_UNDERSCORE_REPLACEMENT]
  tokenizer = text_encoder.Tokenizer(
      alphanum_only=False, reserved_tokens=reserved_tokens)
  num_chars = 0
  token_counts = collections.defaultdict(int)
  for s in generator:
    s = tf.compat.as_text(s)
    if max_chars and (num_chars + len(s)) >= max_chars:
      s = s[:(max_chars - num_chars)]
    tokens = tokenizer.tokenize(s)
    tokens = _prepare_tokens_for_encode(tokens)
    for t in tokens:
      token_counts[t] += 1
    if max_chars:
      num_chars += len(s)
      if num_chars > max_chars:
        break
  return token_counts


def _validate_build_arguments(max_subword_length, reserved_tokens,
                              target_vocab_size):
  """Validate arguments for SubwordTextEncoder.build_from_corpus."""
  if max_subword_length <= 0:
    raise ValueError(
        "max_subword_length must be > 0. Note that memory and compute for "
        "building the vocabulary scale quadratically in the length of the "
        "longest token.")
  for t in reserved_tokens:
    if t.endswith("_") or not text_encoder.is_mixed_alphanum(t):
      raise ValueError(
          "Reserved tokens must not end with _ and they must contain a mix "
          "of alphanumeric and non-alphanumeric characters. For example, "
          "'<EOS>'.")
  # Minimum vocab size = bytes + pad + 1
  minimum_vocab_size = text_encoder.NUM_BYTES + 1 + 1
  if target_vocab_size < minimum_vocab_size:
    raise ValueError("target_vocab_size must be >= %d. Got %d" %
                     (minimum_vocab_size, target_vocab_size))


def _trim_underscore(token):
  if token.endswith("_"):
    return token[:-1]
  return token


def _trim_underscore_and_tell(token):
  if token.endswith("_"):
    return token[:-1], True
  return token, False


def _escape(s):
  return s.replace("_", _UNDERSCORE_REPLACEMENT)


def _unescape(s):
  return s.replace(_UNDERSCORE_REPLACEMENT, "_")


def _prepare_tokens_for_encode(tokens):
  """Prepare tokens for encoding.

  Tokens followed by a single space have "_" appended and the single space token
  is dropped.

  If a token is _UNDERSCORE_REPLACEMENT, it is broken up into 2 tokens.

  Args:
    tokens: `list<str>`, tokens to prepare.

  Returns:
    `list<str>` prepared tokens.
  """
  prepared_tokens = []

  def _prepare_token(t, next_t):
    skip_next = False
    t = _escape(t)
    # If next token is a single space, add _ suffix to token and skip the
    # empty space.
    if next_t == " ":
      t += "_"
      skip_next = True
    return t, skip_next

  next_tokens = tokens[1:] + [None]
  skip_single_token = False
  for token, next_token in zip(tokens, next_tokens):
    if skip_single_token:
      skip_single_token = False
      continue

    # If the user-supplied string contains the underscore replacement string,
    # break it into 2 tokens and encode those separately.
    if token == _UNDERSCORE_REPLACEMENT:
      t1, t2 = _UNDERSCORE_REPLACEMENT[:2], _UNDERSCORE_REPLACEMENT[2:]
      t1, _ = _prepare_token(t1, None)
      t2, _ = _prepare_token(t2, next_token)
      prepared_tokens.append(t1)
      prepared_tokens.append(t2)
      continue

    token, skip_single_token = _prepare_token(token, next_token)
    prepared_tokens.append(token)
  return prepared_tokens
