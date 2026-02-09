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

"""Abstract base classes for all splitters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.module import module


class Splitter(module.Module):
  """An abstract base class for splitting text.

  A Splitter is a module that splits strings into pieces.  Generally, the pieces
  returned by a splitter correspond to substrings of the original string, and
  can be encoded using either strings or integer ids (where integer ids could be
  created by hashing strings or by looking them up in a fixed vocabulary table
  that maps strings to ids).

  Each Splitter subclass must implement a `split` method, which subdivides
  each string in an input Tensor into pieces.  E.g.:

  >>> class SimpleSplitter(tf_text.Splitter):
  ...   def split(self, input):
  ...     return tf.strings.split(input)
  >>> print(SimpleSplitter().split(["hello world", "this is a test"]))
  <tf.RaggedTensor [[b'hello', b'world'], [b'this', b'is', b'a', b'test']]>
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def split(self, input):  # pylint: disable=redefined-builtin
    """Splits the input tensor into pieces.

    Generally, the pieces returned by a splitter correspond to substrings of the
    original string, and can be encoded using either strings or integer ids.

    Example:

    >>> print(tf_text.WhitespaceTokenizer().split("small medium large"))
    tf.Tensor([b'small' b'medium' b'large'], shape=(3,), dtype=string)

    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
        `RaggedTensor`.

    Returns:
      An N+1-dimensional UTF-8 string or integer `Tensor` or `RaggedTensor`.
      For each string from the input tensor, the final, extra dimension contains
      the pieces that string was split into.
    """
    raise NotImplementedError("Abstract method")


class SplitterWithOffsets(Splitter):
  r"""An abstract base class for splitters that return offsets.

  Each SplitterWithOffsets subclass must implement the `split_with_offsets`
  method, which returns a tuple containing both the pieces and the offsets where
  those pieces occurred in the input string.  E.g.:

  >>> class CharSplitter(SplitterWithOffsets):
  ...   def split_with_offsets(self, input):
  ...     chars, starts = tf.strings.unicode_split_with_offsets(input, 'UTF-8')
  ...     lengths = tf.expand_dims(tf.strings.length(input), -1)
  ...     ends = tf.concat([starts[..., 1:], tf.cast(lengths, tf.int64)], -1)
  ...     return chars, starts, ends
  ...   def split(self, input):
  ...     return self.split_with_offsets(input)[0]
  >>> pieces, starts, ends = CharSplitter().split_with_offsets("a😊c")
  >>> print(pieces.numpy(), starts.numpy(), ends.numpy())
  [b'a' b'\xf0\x9f\x98\x8a' b'c'] [0 1 5] [1 5 6]
  """

  @abc.abstractmethod
  def split_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Splits the input tensor, and returns the resulting pieces with offsets.

    Example:

    >>> splitter = tf_text.WhitespaceTokenizer()
    >>> pieces, starts, ends = splitter.split_with_offsets("a bb ccc")
    >>> print(pieces.numpy(), starts.numpy(), ends.numpy())
    [b'a' b'bb' b'ccc'] [0 2 5] [1 4 8]

    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
        `RaggedTensor`.

    Returns:
      A tuple `(pieces, start_offsets, end_offsets)` where:

        * `pieces` is an N+1-dimensional UTF-8 string or integer `Tensor` or
            `RaggedTensor`.
        * `start_offsets` is an N+1-dimensional integer `Tensor` or
            `RaggedTensor` containing the starting indices of each piece (byte
            indices for input strings).
        * `end_offsets` is an N+1-dimensional integer `Tensor` or
            `RaggedTensor` containing the exclusive ending indices of each piece
            (byte indices for input strings).
    """
    raise NotImplementedError("Abstract method")
