# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Abstract base classes for all tokenizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.module import module
from tensorflow_text.python.ops.splitter import Splitter
from tensorflow_text.python.ops.splitter import SplitterWithOffsets


class Tokenizer(Splitter):
  """Base class for tokenizer implementations.

  A Tokenizer is a `text.Splitter` that splits strings into *tokens*.  Tokens
  generally correspond to short substrings of the source string.  Tokens can be
  encoded using either strings or integer ids (where integer ids could be
  created by hashing strings or by looking them up in a fixed vocabulary table
  that maps strings to ids).

  Each Tokenizer subclass must implement a `tokenize` method, which splits each
  string in a Tensor into tokens. E.g.:

  >>> class SimpleTokenizer(tf_text.Tokenizer):
  ...   def tokenize(self, input):
  ...     return tf.strings.split(input)
  >>> print(SimpleTokenizer().tokenize(["hello world", "this is a test"]))
  <tf.RaggedTensor [[b'hello', b'world'], [b'this', b'is', b'a', b'test']]>

  By default, the `split` method simply delegates to `tokenize`.
  """

  @abc.abstractmethod
  def tokenize(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes the input tensor.

    Splits each string in the input tensor into a sequence of tokens.  Tokens
    generally correspond to short substrings of the source string.  Tokens can
    be encoded using either strings or integer ids.

    Example:

    >>> print(tf_text.WhitespaceTokenizer().tokenize("small medium large"))
    tf.Tensor([b'small' b'medium' b'large'], shape=(3,), dtype=string)

    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
        `RaggedTensor`.

    Returns:
      An N+1-dimensional UTF-8 string or integer `Tensor` or `RaggedTensor`.
      For each string from the input tensor, the final, extra dimension contains
      the tokens that string was split into.
    """
    raise NotImplementedError("Abstract method")

  def split(self, input):  # pylint: disable=redefined-builtin
    """Alias for `Tokenizer.tokenize`."""
    return self.tokenize(input)


class TokenizerWithOffsets(Tokenizer, SplitterWithOffsets):
  r"""Base class for tokenizer implementations that return offsets.

  The offsets indicate which substring from the input string was used to
  generate each token.  E.g., if `input` is a single string, then each token
  `token[i]` was generated from the substring `input[starts[i]:ends[i]]`.

  Each TokenizerWithOffsets subclass must implement the `tokenize_with_offsets`
  method, which returns a tuple containing both the pieces and the start and
  end offsets where those pieces occurred in the input string.  I.e., if
  `tokens, starts, ends = tokenize_with_offsets(s)`, then each token `token[i]`
  corresponds with `tf.strings.substr(s, starts[i], ends[i] - starts[i])`.

  If the tokenizer encodes tokens as strings (and not token ids), then it will
  usually be the case that these corresponding strings are equal; but that is
  not technically required.  For example, a tokenizer might choose to downcase
  strings

  Example:

  >>> class CharTokenizer(TokenizerWithOffsets):
  ...   def tokenize_with_offsets(self, input):
  ...     chars, starts = tf.strings.unicode_split_with_offsets(input, 'UTF-8')
  ...     lengths = tf.expand_dims(tf.strings.length(input), -1)
  ...     ends = tf.concat([starts[..., 1:], tf.cast(lengths, tf.int64)], -1)
  ...     return chars, starts, ends
  ...   def tokenize(self, input):
  ...     return self.tokenize_with_offsets(input)[0]
  >>> pieces, starts, ends = CharTokenizer().split_with_offsets("a😊c")
  >>> print(pieces.numpy(), starts.numpy(), ends.numpy())
  [b'a' b'\xf0\x9f\x98\x8a' b'c'] [0 1 5] [1 5 6]

  """

  @abc.abstractmethod
  def tokenize_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes the input tensor and returns the result with byte-offsets.

    The offsets indicate which substring from the input string was used to
    generate each token.  E.g., if `input` is a `tf.string` tensor, then each
    token `token[i]` was generated from the substring
    `tf.substr(input, starts[i], len=ends[i]-starts[i])`.

    Note: Remember that the `tf.string` type is a byte-string. The returned
    indices are in units of bytes, not characters like a Python `str`.

    Example:

    >>> splitter = tf_text.WhitespaceTokenizer()
    >>> pieces, starts, ends = splitter.tokenize_with_offsets("a bb ccc")
    >>> print(pieces.numpy(), starts.numpy(), ends.numpy())
    [b'a' b'bb' b'ccc'] [0 2 5] [1 4 8]
    >>> print(tf.strings.substr("a bb ccc", starts, ends-starts))
    tf.Tensor([b'a' b'bb' b'ccc'], shape=(3,), dtype=string)

    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
        `RaggedTensor`.

    Returns:
      A tuple `(tokens, start_offsets, end_offsets)` where:

        * `tokens` is an N+1-dimensional UTF-8 string or integer `Tensor` or
            `RaggedTensor`.
        * `start_offsets` is an N+1-dimensional integer `Tensor` or
            `RaggedTensor` containing the starting indices of each token (byte
            indices for input strings).
        * `end_offsets` is an N+1-dimensional integer `Tensor` or
            `RaggedTensor` containing the exclusive ending indices of each token
            (byte indices for input strings).
    """
    raise NotImplementedError("Abstract method")

  def split_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Alias for `TokenizerWithOffsets.tokenize_with_offsets`."""
    return self.tokenize_with_offsets(input)


class Detokenizer(module.Module):
  """Base class for detokenizer implementations.

  A Detokenizer is a module that combines tokens to form strings.  Generally,
  subclasses of `Detokenizer` will also be subclasses of `Tokenizer`; and the
  `detokenize` method will be the inverse of the `tokenize` method.  I.e.,
  `tokenizer.detokenize(tokenizer.tokenize(s)) == s`.

  Each Detokenizer subclass must implement a `detokenize` method, which combines
  tokens together to form strings.  E.g.:

  >>> class SimpleDetokenizer(tf_text.Detokenizer):
  ...   def detokenize(self, input):
  ...     return tf.strings.reduce_join(input, axis=-1, separator=" ")
  >>> text = tf.ragged.constant([["hello", "world"], ["a", "b", "c"]])
  >>> print(SimpleDetokenizer().detokenize(text))
  tf.Tensor([b'hello world' b'a b c'], shape=(2,), dtype=string)
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def detokenize(self, input):  # pylint: disable=redefined-builtin
    """Assembles the tokens in the input tensor into a string.

    Generally, `detokenize` is the inverse of the `tokenize` method, and can
    be used to reconstrct a string from a set of tokens.  This is especially
    helpful in cases where the tokens are integer ids, such as indexes into a
    vocabulary table -- in that case, the tokenized encoding is not very
    human-readable (since it's just a list of integers), so the `detokenize`
    method can be used to turn it back into something that's more readable.

    Args:
      input: An N-dimensional UTF-8 string or integer `Tensor` or
        `RaggedTensor`.

    Returns:
      An (N-1)-dimensional UTF-8 string `Tensor` or `RaggedTensor`.
    """
    raise NotImplementedError("Abstract method")
