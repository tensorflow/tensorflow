# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for string_split_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringSplitOpTest(test.TestCase):

  def testStringSplit(self):
    strings = ["pigs on the wing", "animals"]

    with self.cached_session() as sess:
      tokens = string_ops.string_split(strings)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]])
      self.assertAllEqual(values, [b"pigs", b"on", b"the", b"wing", b"animals"])
      self.assertAllEqual(shape, [2, 4])

  def testStringSplitEmptyDelimiter(self):
    strings = ["hello", "hola", b"\xF0\x9F\x98\x8E"]  # Last string is U+1F60E

    with self.cached_session() as sess:
      tokens = string_ops.string_split(strings, delimiter="")
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                                    [1, 0], [1, 1], [1, 2], [1, 3], [2, 0],
                                    [2, 1], [2, 2], [2, 3]])
      expected = np.array(
          [
              "h", "e", "l", "l", "o", "h", "o", "l", "a", b"\xf0", b"\x9f",
              b"\x98", b"\x8e"
          ],
          dtype="|S1")
      self.assertAllEqual(values.tolist(), expected)
      self.assertAllEqual(shape, [3, 5])

  def testStringSplitEmptyToken(self):
    strings = ["", " a", "b ", " c", " ", " d ", "  e", "f  ", "  g  ", "  "]

    with self.cached_session() as sess:
      tokens = string_ops.string_split(strings)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(
          indices,
          [[1, 0], [2, 0], [3, 0], [5, 0], [6, 0], [7, 0], [8, 0]])
      self.assertAllEqual(values, [b"a", b"b", b"c", b"d", b"e", b"f", b"g"])
      self.assertAllEqual(shape, [10, 1])

  def testStringSplitOnSetEmptyToken(self):
    strings = ["", " a", "b ", " c", " ", " d ", ". e", "f .", " .g. ", " ."]

    with self.cached_session() as sess:
      tokens = string_ops.string_split(strings, delimiter=" .")
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(
          indices,
          [[1, 0], [2, 0], [3, 0], [5, 0], [6, 0], [7, 0], [8, 0]])
      self.assertAllEqual(values, [b"a", b"b", b"c", b"d", b"e", b"f", b"g"])
      self.assertAllEqual(shape, [10, 1])

  def testStringSplitWithDelimiter(self):
    strings = ["hello|world", "hello world"]

    with self.cached_session() as sess:
      self.assertRaises(
          ValueError, string_ops.string_split, strings, delimiter=["|", ""])

      self.assertRaises(
          ValueError, string_ops.string_split, strings, delimiter=["a"])

      tokens = string_ops.string_split(strings, delimiter="|")
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [1, 0]])
      self.assertAllEqual(values, [b"hello", b"world", b"hello world"])
      self.assertAllEqual(shape, [2, 2])

      tokens = string_ops.string_split(strings, delimiter="| ")
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(values, [b"hello", b"world", b"hello", b"world"])
      self.assertAllEqual(shape, [2, 2])

  def testStringSplitWithDelimiterTensor(self):
    strings = ["hello|world", "hello world"]

    with self.cached_session() as sess:
      delimiter = array_ops.placeholder(dtypes.string)

      tokens = string_ops.string_split(strings, delimiter=delimiter)

      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a", "b"]})
      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a"]})
      indices, values, shape = sess.run(tokens, feed_dict={delimiter: "|"})

      self.assertAllEqual(indices, [[0, 0], [0, 1], [1, 0]])
      self.assertAllEqual(values, [b"hello", b"world", b"hello world"])
      self.assertAllEqual(shape, [2, 2])

  def testStringSplitWithDelimitersTensor(self):
    strings = ["hello.cruel,world", "hello cruel world"]

    with self.cached_session() as sess:
      delimiter = array_ops.placeholder(dtypes.string)

      tokens = string_ops.string_split(strings, delimiter=delimiter)

      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a", "b"]})
      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a"]})
      indices, values, shape = sess.run(tokens, feed_dict={delimiter: ".,"})

      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [1, 0]])
      self.assertAllEqual(values,
                          [b"hello", b"cruel", b"world", b"hello cruel world"])
      self.assertAllEqual(shape, [2, 3])

  def testStringSplitWithNoSkipEmpty(self):
    strings = ["#a", "b#", "#c#"]

    with self.cached_session() as sess:
      tokens = string_ops.string_split(strings, "#", skip_empty=False)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1],
                                    [1, 0], [1, 1],
                                    [2, 0], [2, 1], [2, 2]])
      self.assertAllEqual(values, [b"", b"a", b"b", b"", b"", b"c", b""])
      self.assertAllEqual(shape, [3, 3])

    with self.cached_session() as sess:
      tokens = string_ops.string_split(strings, "#")
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(values, [b"a", b"b", b"c"])
      self.assertAllEqual(indices, [[0, 0], [1, 0], [2, 0]])
      self.assertAllEqual(shape, [3, 1])


class StringSplitV2OpTest(test.TestCase):

  def testSplitV2(self):
    strings = ["pigs on the wing", "animals"]

    with self.cached_session() as sess:
      tokens = string_ops.string_split_v2(strings)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]])
      self.assertAllEqual(values, [b"pigs", b"on", b"the", b"wing", b"animals"])
      self.assertAllEqual(shape, [2, 4])

  def testSplitV2MultiCharSeparator(self):
    # Match Python behavior:
    # >>> '1<>2<>3'.split('<>')
    # ['1', '2', '3']
    # >>> "<><>4<>5<><>6<>".split("<>")
    # ['', '', '4', '5', '', '6', '']
    strings = ["1<>2<>3", "<><>4<>5<><>6<>"]

    with self.cached_session() as sess:
      tokens = string_ops.string_split_v2(strings, sep="<>")
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(
          indices, [[0, 0], [0, 1], [0, 2],
                    [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
      self.assertAllEqual(values, [b"1", b"2", b"3",
                                   b"", b"", b"4", b"5", b"", b"6", b""])
      self.assertAllEqual(shape, [2, 7])

  def testSplitV2SimpleSeparator(self):
    # Match Python behavior:
    # >>> '1,2,3'.split(',')
    # ['1', '2', '3']
    # >>> '1,2,,3,'.split(',')
    # ['1', '2', '', '3', '']
    strings = ["1,2,3", "4,5,,6,"]

    with self.cached_session() as sess:
      tokens = string_ops.string_split_v2(strings, sep=',')
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2],
                                    [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]])
      self.assertAllEqual(values, [b"1", b"2", b"3",
                                   b"4", b"5", b"", b"6", b""])
      self.assertAllEqual(shape, [2, 5])

  def testSplitV2EmptySeparator(self):
    # Match Python behavior:
    # >>> '1 2 3'.split()
    # ['1', '2', '3']
    #>>> '   1   2   3   '.split()
    #['1', '2', '3']
    strings = ["1 2 3", "  4  5    6  "]

    with self.cached_session() as sess:
      tokens = string_ops.string_split_v2(strings)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2],
                                    [1, 0], [1, 1], [1, 2]])
      self.assertAllEqual(values, [b"1", b"2", b"3", b"4", b"5", b"6"])
      self.assertAllEqual(shape, [2, 3])

  def testSplitV2SimpleSeparatorMaxSplit(self):
    # Match Python behavior:
    # >>> '1,2,3'.split(',', maxsplit=1)
    # ['1', '2,3']
    # >>> '4,5,,6,'.split(',', maxsplit=1)
    # ['4', '5,,6,']
    strings = ["1,2,3", "4,5,,6,"]

    with self.cached_session() as sess:
      tokens = string_ops.string_split_v2(strings, sep=',', maxsplit=1)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1],
                                    [1, 0], [1, 1]])
      self.assertAllEqual(values, [b"1", b"2,3", b"4", b"5,,6,"])
      self.assertAllEqual(shape, [2, 2])

  def testSplitV2EmptySeparatorMaxSplit(self):
    # Match Python behavior:
    # '1 2 3'.split(maxsplit=1)
    # ['1', '2 3']
    # >>> "  4  5    6  ".split(maxsplit=1)
    # ['4', '5    6  ']
    strings = ["1 2 3", "  4  5    6  "]

    with self.cached_session() as sess:
      tokens = string_ops.string_split_v2(strings, maxsplit=1)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1],
                                    [1, 0], [1, 1]])
      self.assertAllEqual(values, [b"1", b"2 3", b"4", b"5    6  "])
      self.assertAllEqual(shape, [2, 2])


if __name__ == "__main__":
  test.main()
