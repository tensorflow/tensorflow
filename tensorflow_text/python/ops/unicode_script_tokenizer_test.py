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

# -*- coding: utf-8 -*-
"""Tests for unicode_script_tokenizer_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops.unicode_script_tokenizer import UnicodeScriptTokenizer


@test_util.run_all_in_graph_and_eager_modes
class UnicodeScriptTokenizerOpTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(UnicodeScriptTokenizerOpTest, self).setUp()
    self.tokenizer = UnicodeScriptTokenizer()

  def testRequireParams(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        self.tokenizer.tokenize()

  def testScalar(self):
    test_value = constant_op.constant(b'I love Flume!')
    expected_tokens = [b'I', b'love', b'Flume', b'!']
    expected_offset_starts = [0, 2, 7, 12]
    expected_offset_ends = [1, 6, 12, 13]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testVectorSingleValue(self):
    test_value = constant_op.constant([b'I love Flume!'])
    expected_tokens = [[b'I', b'love', b'Flume', b'!']]
    expected_offset_starts = [[0, 2, 7, 12]]
    expected_offset_ends = [[1, 6, 12, 13]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testVector(self):
    test_value = constant_op.constant([b'I love Flume!', b'Good day'])
    expected_tokens = [[b'I', b'love', b'Flume', b'!'], [b'Good', b'day']]
    expected_offset_starts = [[0, 2, 7, 12], [0, 5]]
    expected_offset_ends = [[1, 6, 12, 13], [4, 8]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testMatrix(self):
    test_value = constant_op.constant([[b'I love Flume!', b'Good day'],
                                       [b'I don\'t want', b'no scrubs']])
    expected_tokens = [[[b'I', b'love', b'Flume', b'!'], [b'Good', b'day']],
                       [[b'I', b'don', b'\'', b't', b'want'],
                        [b'no', b'scrubs']]]
    expected_offset_starts = [[[0, 2, 7, 12], [0, 5]],
                              [[0, 2, 5, 6, 8], [0, 3]]]
    expected_offset_ends = [[[1, 6, 12, 13], [4, 8]],
                            [[1, 5, 6, 7, 12], [2, 9]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testMatrixRagged(self):
    test_value = ragged_factory_ops.constant([[b'I love Flume!'],
                                              [b'I don\'t want', b'no scrubs']])
    expected_tokens = [[[b'I', b'love', b'Flume', b'!']],
                       [[b'I', b'don', b'\'', b't', b'want'],
                        [b'no', b'scrubs']]]
    expected_offset_starts = [[[0, 2, 7, 12]],
                              [[0, 2, 5, 6, 8], [0, 3]]]
    expected_offset_ends = [[[1, 6, 12, 13]],
                            [[1, 5, 6, 7, 12], [2, 9]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def test3DimMatrix(self):
    test_value = constant_op.constant([[[b'I love Flume!', b'Good day'],
                                        [b'I don\'t want', b'no scrubs']],
                                       [[b'I love Zhu!', b'Good night'],
                                        [b'A scrub is', b'a guy']]])
    expected_tokens = [[[[b'I', b'love', b'Flume', b'!'], [b'Good', b'day']],
                        [[b'I', b'don', b'\'', b't', b'want'],
                         [b'no', b'scrubs']]],
                       [[[b'I', b'love', b'Zhu', b'!'], [b'Good', b'night']],
                        [[b'A', b'scrub', b'is'], [b'a', b'guy']]]]
    expected_offset_starts = [[[[0, 2, 7, 12], [0, 5]],
                               [[0, 2, 5, 6, 8], [0, 3]]],
                              [[[0, 2, 7, 10], [0, 5]],
                               [[0, 2, 8], [0, 2]]]]
    expected_offset_ends = [[[[1, 6, 12, 13], [4, 8]],
                             [[1, 5, 6, 7, 12], [2, 9]]],
                            [[[1, 6, 10, 11], [4, 10]],
                             [[1, 7, 10], [1, 5]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def test3DimMatrixRagged(self):
    test_value = ragged_factory_ops.constant([[[b'I love Flume!'],
                                               [b'I don\'t want',
                                                b'no scrubs']],
                                              [[b'I love Zhu!',
                                                b'Good night']]])
    expected_tokens = [[[[b'I', b'love', b'Flume', b'!']],
                        [[b'I', b'don', b'\'', b't', b'want'],
                         [b'no', b'scrubs']]],
                       [[[b'I', b'love', b'Zhu', b'!'], [b'Good', b'night']]]]
    expected_offset_starts = [[[[0, 2, 7, 12]],
                               [[0, 2, 5, 6, 8], [0, 3]]],
                              [[[0, 2, 7, 10], [0, 5]]]]
    expected_offset_ends = [[[[1, 6, 12, 13]],
                             [[1, 5, 6, 7, 12], [2, 9]]],
                            [[[1, 6, 10, 11], [4, 10]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testInternationalization(self):
    test_value = constant_op.constant([u"J'adore la灯".encode('utf8'),
                                       u'¡Escríbeme!'.encode('utf8')])
    expected_tokens = [[b'J', b"'", b'adore', b'la', u'灯'.encode('utf8')],
                       [u'¡'.encode('utf8'), u'Escríbeme'.encode('utf8'), b'!']]
    expected_offset_starts = [[0, 1, 2, 8, 10], [0, 2, 12]]
    expected_offset_ends = [[1, 2, 7, 10, 13], [2, 12, 13]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testSpaceBoundaries(self):
    test_value = constant_op.constant([b' Hook em! ', b' .Ok.   Go  '])
    expected_tokens = [[b'Hook', b'em', b'!'], [b'.', b'Ok', b'.', b'Go']]
    expected_offset_starts = [[1, 6, 8], [1, 2, 4, 8]]
    expected_offset_ends = [[5, 8, 9], [2, 4, 5, 10]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testKeepWhitespace(self):
    test_value = constant_op.constant([
        b'\'Black Panther,\' \t \xe2\x80\x98A Star Is Born\xe2\x80\x98 among AFI Awards honorees',
        b' .Ok.   Go  '
    ])
    expected_tokens = [[
        b'\'', b'Black', b' ', b'Panther', b',\'', b' \t ', b'\xe2\x80\x98',
        b'A', b' ', b'Star', b' ', b'Is', b' ', b'Born', b'\xe2\x80\x98', b' ',
        b'among', b' ', b'AFI', b' ', b'Awards', b' ', b'honorees'
    ], [b' ', b'.', b'Ok', b'.', b'   ', b'Go', b'  ']]
    expected_offset_starts = [
        [0, 1, 6, 7, 14, 16, 19, 22, 23, 24, 28, 29, 31, 32, 36, 39, 40,
         45, 46, 49, 50, 56, 57],
        [0, 1, 2, 4, 5, 8, 10]]
    expected_offset_ends = [
        [1, 6, 7, 14, 16, 19, 22, 23, 24, 28, 29, 31, 32, 36, 39, 40,
         45, 46, 49, 50, 56, 57, 65],
        [1, 2, 4, 5, 8, 10, 12]]
    self.tokenizer = UnicodeScriptTokenizer(keep_whitespace=True)
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testOnlySpaces(self):
    test_value = constant_op.constant([b' ', b'     '])
    expected_tokens = [[], []]
    expected_offset_starts = [[], []]
    expected_offset_ends = [[], []]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testWhitespaceCharacters(self):
    test_value = constant_op.constant([b'things:\tcarpet\rdesk\nlamp'])
    expected_tokens = [[b'things', b':', b'carpet', b'desk', b'lamp']]
    expected_offset_starts = [[0, 6, 8, 15, 20]]
    expected_offset_ends = [[6, 7, 14, 19, 24]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyStringSingle(self):
    test_value = constant_op.constant([b''])
    expected_tokens = [[]]
    expected_offset_starts = [[]]
    expected_offset_ends = [[]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyString(self):
    test_value = constant_op.constant(
        [b'', b'I love Flume!', b'', b'O hai', b''])
    expected_tokens = [[], [b'I', b'love', b'Flume', b'!'], [], [b'O', b'hai'],
                       []]
    expected_offset_starts = [[], [0, 2, 7, 12], [], [0, 2], []]
    expected_offset_ends = [[], [1, 6, 12, 13], [], [1, 5], []]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyDimensions(self):
    test_value = ragged_factory_ops.constant(
        [[[b'I love Flume!', b'Good day. . .'], []], [],
         [[b'I love Zhu!', b'Good night'], [b'A scrub is', b'a guy']]])
    expected_tokens = [[[[b'I', b'love', b'Flume', b'!'],
                         [b'Good', b'day', b'...']], []], [],
                       [[[b'I', b'love', b'Zhu', b'!'], [b'Good', b'night']],
                        [[b'A', b'scrub', b'is'], [b'a', b'guy']]]]
    expected_offset_starts = [[[[0, 2, 7, 12], [0, 5, 8]],
                               []],
                              [],
                              [[[0, 2, 7, 10], [0, 5]],
                               [[0, 2, 8], [0, 2]]]]
    expected_offset_ends = [[[[1, 6, 12, 13], [4, 8, 13]],
                             []],
                            [],
                            [[[1, 6, 10, 11], [4, 10]],
                             [[1, 7, 10], [1, 5]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)


if __name__ == '__main__':
  test.main()
