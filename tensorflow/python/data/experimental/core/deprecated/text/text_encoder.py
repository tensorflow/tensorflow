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
"""TextEncoders convert between text and integers."""
from __future__ import unicode_literals

import abc
import hashlib
import json
import re

import six
import tensorflow.compat.v2 as tf


def _re_compile(pattern):
  return re.compile(pattern, flags=re.UNICODE)  # pytype: disable=wrong-keyword-args


NUM_BYTES = 2**8
ALPHANUM_REGEX = _re_compile(r"\W+")
ALL_REGEX = _re_compile(r"(\W+)")




class TextEncoderConfig(object):
  """Configuration for `tfds.features.Text`."""

  def __init__(self, encoder=None, encoder_cls=None, vocab_size=None,
               name=None):
    if encoder:
      if (encoder_cls or vocab_size):
        raise ValueError("If encoder is provided, encoder_cls and "
                         "vocab_size must be None")
      encoder_cls = type(encoder)
      vocab_size = encoder.vocab_size
    else:
      if encoder_cls is ByteTextEncoder:
        encoder = encoder_cls()

    self.encoder = encoder
    self.encoder_cls = encoder_cls
    self.vocab_size = vocab_size
    self.name = name


@six.add_metaclass(abc.ABCMeta)
class TextEncoder(object):
  """Abstract base class for converting between text and integers.

  **A note on padding**:

    Because text data is typically variable length and nearly always requires
    padding during training, ID 0 is always reserved for padding. To accommodate
    this, all `TextEncoder`s behave in certain ways:

    * `encode`: never returns id 0 (all ids are 1+)
    * `decode`: drops 0 in the input ids
    * `vocab_size`: includes ID 0

    New subclasses should be careful to match this behavior.
  """

  @abc.abstractmethod
  def encode(self, s):
    """Encodes text into a list of integers."""
    raise NotImplementedError

  @abc.abstractmethod
  def decode(self, ids):
    """Decodes a list of integers into text."""
    raise NotImplementedError

  @abc.abstractproperty
  def vocab_size(self):
    """Size of the vocabulary. Decode produces ints [1, vocab_size)."""
    raise NotImplementedError

  @abc.abstractmethod
  def save_to_file(self, filename_prefix):
    """Store to file. Inverse of load_from_file."""
    raise NotImplementedError

  @classmethod
  @abc.abstractmethod
  def load_from_file(cls, filename_prefix):  # pylint: disable=no-self-argument
    """Load from file. Inverse of save_to_file."""
    raise NotImplementedError

  @classmethod
  def _write_lines_to_file(cls, filename, lines, metadata_dict=None):
    """Writes lines to file prepended by header and metadata."""
    write_lines_to_file(cls.__name__, filename, lines, metadata_dict)

  @classmethod
  def _read_lines_from_file(cls, filename):
    return read_lines_from_file(cls.__name__, filename)

  def __repr__(self):
    return "<%s vocab_size=%d>" % (type(self).__name__, self.vocab_size)


class ByteTextEncoder(TextEncoder):
  """Byte-encodes text."""

  def __init__(self, additional_tokens=None):
    """Constructs ByteTextEncoder.

    Args:
      additional_tokens: `list<str>`, list of additional tokens. These will be
        assigned vocab ids `[1, 1+len(additional_tokens)]`. Useful for things
        like "end-of-string" tokens (e.g. "<EOS>").
    """
    self._additional_tokens, self._additional_tokens_re = (
        _prepare_reserved_tokens(additional_tokens))
    # Note that internally everything is 0-indexed. Padding is dealt with at the
    # end of encode and the beginning of decode.
    self._additional_token_to_id = dict(
        zip(self._additional_tokens, range(len(self._additional_tokens))))

  def encode(self, s):
    if not self.additional_tokens:
      return pad_incr(list(bytearray(tf.compat.as_bytes(s))))

    # Handle additional tokens
    s = tf.compat.as_text(s)
    ids = []
    for substr in self._additional_tokens_re.split(s):
      if not substr:
        continue
      tok_id = self._additional_token_to_id.get(substr)
      if tok_id is None:
        offset = len(self.additional_tokens)
        tok_ids = [i + offset for i in
                   list(bytearray(tf.compat.as_bytes(substr)))]
      else:
        tok_ids = [tok_id]
      ids.extend(tok_ids)

    return pad_incr(ids)

  def decode(self, ids):
    ids = pad_decr(ids)
    if not self.additional_tokens:
      return tf.compat.as_text(bytes(bytearray(ids)))

    # Handle additional tokens
    # First pass picks out the additional tokens
    tmp_decoded = []
    for byte_id in ids:
      is_additional_token = byte_id < len(self.additional_tokens)
      if is_additional_token:
        tmp_decoded.append(self.additional_tokens[byte_id])
      else:
        # Leave these as ints so that we can contiguously decode bytes
        # afterwards
        tmp_decoded.append(byte_id - len(self.additional_tokens))

    # Second pass to decode contiguous bytes
    strs = []
    i = 0
    while i < len(tmp_decoded):
      el = tmp_decoded[i]
      if isinstance(el, six.string_types):
        strs.append(el)
        i += 1
      else:
        # Decode contiguous bytes
        byte_ids = []
        while i < len(tmp_decoded):
          b = tmp_decoded[i]
          if isinstance(b, int):
            byte_ids.append(b)
            i += 1
          else:
            break
        strs.append(bytes(bytearray(byte_ids)).decode("utf-8", "replace"))
    return "".join(strs)

  @property
  def vocab_size(self):
    # Plus 1 for pad
    return len(self.additional_tokens) + NUM_BYTES + 1

  @property
  def additional_tokens(self):
    return self._additional_tokens

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + ".bytes"

  def save_to_file(self, filename_prefix):
    self._write_lines_to_file(
        self._filename(filename_prefix), self.additional_tokens)

  @classmethod
  def load_from_file(cls, filename_prefix):
    lines, _ = cls._read_lines_from_file(cls._filename(filename_prefix))
    return cls(additional_tokens=lines)


class TokenTextEncoder(TextEncoder):
  r"""TextEncoder backed by a list of tokens.

  Tokenization splits on (and drops) non-alphanumeric characters with
  regex "\W+".
  """

  def __init__(self,
               vocab_list,
               oov_buckets=1,
               oov_token="UNK",
               lowercase=False,
               tokenizer=None,
               strip_vocab=True,
               decode_token_separator=" "):
    """Constructs a TokenTextEncoder.

    To load from a file saved with `TokenTextEncoder.save_to_file`, use
    `TokenTextEncoder.load_from_file`.

    Args:
      vocab_list: `list<str>`, list of tokens.
      oov_buckets: `int`, the number of `int`s to reserve for OOV hash buckets.
        Tokens that are OOV will be hash-modded into a OOV bucket in `encode`.
      oov_token: `str`, the string to use for OOV ids in `decode`.
      lowercase: `bool`, whether to make all text and tokens lowercase.
      tokenizer: `Tokenizer`, responsible for converting incoming text into a
        list of tokens.
      strip_vocab: `bool`, whether to strip whitespace from the beginning and
        end of elements of `vocab_list`.
      decode_token_separator: `str`, the string used to separate tokens when
        decoding.
    """
    self._vocab_list = [tf.compat.as_text(el) for el in vocab_list]
    if strip_vocab:
      self._vocab_list = [el.strip() for el in self._vocab_list]
    self._lowercase = lowercase
    if self._lowercase:
      self._vocab_list = [t.lower() for t in self._vocab_list]
    # Note that internally everything is 0-indexed. Padding is dealt with at the
    # end of encode and the beginning of decode.
    self._token_to_id = dict(
        zip(self._vocab_list, range(len(self._vocab_list))))
    self._oov_buckets = oov_buckets
    self._oov_token = tf.compat.as_text(oov_token)

    # Reserved tokens are all tokens that are mixed alphanum and non-alphanum.
    reserved_tokens = [t for t in self._vocab_list if is_mixed_alphanum(t)]
    self._tokenizer = (tokenizer or Tokenizer(reserved_tokens=reserved_tokens))
    self._user_defined_tokenizer = tokenizer

    self._decode_token_separator = decode_token_separator

  def encode(self, s):
    s = tf.compat.as_text(s)
    if self.lowercase:
      s = s.lower()
    ids = []
    for token in self._tokenizer.tokenize(s):
      int_id = self._token_to_id.get(token, -1)
      if int_id < 0:
        int_id = self._oov_bucket(token)
        if int_id is None:
          raise ValueError("Out of vocabulary token %s" % token)
      ids.append(int_id)

    # Increment for pad id 0
    return pad_incr(ids)

  def decode(self, ids):
    ids = pad_decr(ids)

    tokens = []
    for int_id in ids:
      if int_id < len(self._vocab_list):
        tokens.append(self._vocab_list[int_id])
      else:
        tokens.append(self._oov_token)
    return self._decode_token_separator.join(tokens)

  @property
  def vocab_size(self):
    # Plus 1 for pad
    return len(self._vocab_list) + self._oov_buckets + 1

  @property
  def tokens(self):
    return list(self._vocab_list)

  @property
  def oov_token(self):
    return self._oov_token

  @property
  def lowercase(self):
    return self._lowercase

  @property
  def tokenizer(self):
    return self._tokenizer

  def _oov_bucket(self, token):
    if self._oov_buckets <= 0:
      return None
    if self._oov_buckets == 1:
      return len(self._vocab_list)
    hash_val = int(hashlib.md5(tf.compat.as_bytes(token)).hexdigest(), 16)
    return len(self._vocab_list) + hash_val % self._oov_buckets

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + ".tokens"

  def save_to_file(self, filename_prefix):
    filename = self._filename(filename_prefix)
    kwargs = {
        "oov_buckets": self._oov_buckets,
        "lowercase": self._lowercase,
        "oov_token": self._oov_token,
    }
    if self._user_defined_tokenizer is not None:
      self._tokenizer.save_to_file(filename)
      kwargs["has_tokenizer"] = True
    self._write_lines_to_file(filename, self._vocab_list, kwargs)

  @classmethod
  def load_from_file(cls, filename_prefix):
    filename = cls._filename(filename_prefix)
    vocab_lines, kwargs = cls._read_lines_from_file(filename)
    has_tokenizer = kwargs.pop("has_tokenizer", False)
    if has_tokenizer:
      kwargs["tokenizer"] = Tokenizer.load_from_file(filename)
    return cls(vocab_list=vocab_lines, **kwargs)


class Tokenizer(object):
  """Splits a string into tokens, and joins them back."""

  def __init__(self, alphanum_only=True, reserved_tokens=None):
    """Constructs a Tokenizer.

    Note that the Tokenizer is invertible if `alphanum_only=False`.
    i.e. `s == t.join(t.tokenize(s))`.

    Args:
      alphanum_only: `bool`, if `True`, only parse out alphanumeric tokens
        (non-alphanumeric characters are dropped);
        otherwise, keep all characters (individual tokens will still be either
        all alphanumeric or all non-alphanumeric).
      reserved_tokens: `list<str>`, a list of strings that, if any are in `s`,
        will be preserved as whole tokens, even if they contain mixed
        alphanumeric/non-alphanumeric characters.
    """
    self._alphanum_only = alphanum_only
    reserved_tokens, self._reserved_tokens_re = _prepare_reserved_tokens(
        reserved_tokens)
    self._reserved_tokens = set(reserved_tokens)

  @property
  def alphanum_only(self):
    return self._alphanum_only

  @property
  def reserved_tokens(self):
    return self._reserved_tokens

  def tokenize(self, s):
    """Splits a string into tokens."""
    s = tf.compat.as_text(s)

    if self.reserved_tokens:
      # First split out the reserved tokens
      substrs = self._reserved_tokens_re.split(s)
    else:
      substrs = [s]

    toks = []
    for substr in substrs:
      if substr in self.reserved_tokens:
        toks.append(substr)
      elif self._alphanum_only:
        toks.extend(ALPHANUM_REGEX.split(substr))
      else:
        toks.extend(ALL_REGEX.split(substr))

    # Filter out empty strings
    toks = [t for t in toks if t]
    return toks

  def join(self, tokens):
    """Joins tokens into a string."""
    if self._alphanum_only:
      return " ".join(tokens)
    else:
      # Fully invertible
      return "".join(tokens)

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + ".tokenizer"

  def save_to_file(self, filename_prefix):
    filename = self._filename(filename_prefix)
    kwargs = {
        "reserved_tokens": list(self._reserved_tokens),
        "alphanum_only": self._alphanum_only
    }
    write_lines_to_file(type(self).__name__, filename, [], kwargs)

  @classmethod
  def load_from_file(cls, filename_prefix):
    filename = cls._filename(filename_prefix)
    _, kwargs = read_lines_from_file(cls.__name__, filename)
    return cls(**kwargs)


def pad_decr(ids):
  """Strip ID 0 and decrement ids by 1."""
  if len(ids) < 1:
    return list(ids)
  if not any(ids):
    return []  # all padding.
  idx = -1
  while not ids[idx]:
    idx -= 1
  if idx == -1:
    ids = ids  # pylint: disable=self-assigning-variable
  else:
    ids = ids[:idx + 1]
  return [i - 1 for i in ids]


def pad_incr(ids):
  """Add 1 to ids to account for pad."""
  return [i + 1 for i in ids]


def _prepare_reserved_tokens(reserved_tokens):
  """Prepare reserved tokens and a regex for splitting them out of strings."""
  reserved_tokens = [tf.compat.as_text(tok) for tok in reserved_tokens or []]
  dups = _find_duplicates(reserved_tokens)
  if dups:
    raise ValueError("Duplicates found in tokens: %s" % dups)
  reserved_tokens_re = _make_reserved_tokens_re(reserved_tokens)
  return reserved_tokens, reserved_tokens_re


def _re_escape(s):
  """Escape regex control characters."""
  escaped = re.sub(r"[(){}\[\].*?|^$\\+-]", r"\\\g<0>", s)
  return escaped


def _make_reserved_tokens_re(reserved_tokens):
  """Constructs compiled regex to parse out reserved tokens."""
  if not reserved_tokens:
    return None
  escaped_tokens = [_re_escape(rt) for rt in reserved_tokens]
  pattern = "(%s)" % "|".join(escaped_tokens)
  reserved_tokens_re = _re_compile(pattern)
  return reserved_tokens_re


def _find_duplicates(els):
  seen = set()
  dups = []
  for x in els:
    if x in seen:
      dups.append(x)
    else:
      seen.add(x)
  return dups


def is_mixed_alphanum(token):
  return len([s for s in ALL_REGEX.split(token) if s]) > 1


_HEADER_PREFIX = "### "
_METADATA_PREFIX = "### Metadata: "


def write_lines_to_file(cls_name, filename, lines, metadata_dict):
  """Writes lines to file prepended by header and metadata."""
  metadata_dict = metadata_dict or {}
  header_line = "%s%s" % (_HEADER_PREFIX, cls_name)
  metadata_line = "%s%s" % (_METADATA_PREFIX,
                            json.dumps(metadata_dict, sort_keys=True))
  with tf.io.gfile.GFile(filename, "wb") as f:
    for line in [header_line, metadata_line]:
      f.write(tf.compat.as_bytes(line))
      f.write(tf.compat.as_bytes("\n"))
    if lines:
      f.write(tf.compat.as_bytes("\n".join(lines)))
      f.write(tf.compat.as_bytes("\n"))


def read_lines_from_file(cls_name, filename):
  """Read lines from file, parsing out header and metadata."""
  with tf.io.gfile.GFile(filename, "rb") as f:
    lines = [tf.compat.as_text(line)[:-1] for line in f]
  header_line = "%s%s" % (_HEADER_PREFIX, cls_name)
  if lines[0] != header_line:
    raise ValueError("File {fname} does not seem to have been created from "
                     "{name}.save_to_file.".format(
                         fname=filename, name=cls_name))
  metadata_dict = json.loads(lines[1][len(_METADATA_PREFIX):])
  return lines[2:], metadata_dict
