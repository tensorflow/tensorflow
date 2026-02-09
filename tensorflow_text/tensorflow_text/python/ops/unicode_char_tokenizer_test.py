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
"""Tests for unicode_char_tokenizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops.unicode_char_tokenizer import UnicodeCharTokenizer  # pylint: disable=line-too-long


@test_util.run_all_in_graph_and_eager_modes
class UnicodeCharTokenizerOpTest(test.TestCase):

  def setUp(self):
    super(UnicodeCharTokenizerOpTest, self).setUp()
    self.tokenizer = UnicodeCharTokenizer()

  def testRequireParams(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        self.tokenizer.tokenize()

  def testScalar(self):
    test_value = constant_op.constant(b'I love Flume!')
    expected_tokens = [
        ord('I'),
        ord(' '),
        ord('l'),
        ord('o'),
        ord('v'),
        ord('e'),
        ord(' '),
        ord('F'),
        ord('l'),
        ord('u'),
        ord('m'),
        ord('e'),
        ord('!')
    ]
    expected_offset_starts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    expected_offset_ends = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def testVectorSingleValue(self):
    test_value = constant_op.constant([b'I lov'])
    expected_tokens = [[ord('I'), ord(' '), ord('l'), ord('o'), ord('v')]]
    expected_offset_starts = [[0, 1, 2, 3, 4]]
    expected_offset_ends = [[1, 2, 3, 4, 5]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def testVector(self):
    test_value = constant_op.constant([b'123', b'456'])
    expected_tokens = [[ord('1'), ord('2'), ord('3')],
                       [ord('4'), ord('5'), ord('6')]]
    expected_offset_starts = [[0, 1, 2], [0, 1, 2]]
    expected_offset_ends = [[1, 2, 3], [1, 2, 3]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def testMatrix(self):
    test_value = constant_op.constant([[b'ab', b'cde'], [b'12', b'34']])
    expected_tokens = [[[ord('a'), ord('b')], [ord('c'),
                                               ord('d'),
                                               ord('e')]],
                       [[ord('1'), ord('2')], [ord('3'), ord('4')]]]
    expected_offset_starts = [[[0, 1], [0, 1, 2]], [[0, 1], [0, 1]]]
    expected_offset_ends = [[[1, 2], [1, 2, 3]], [[1, 2], [1, 2]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(
        self.evaluate(detokenized).to_list(),
        self.evaluate(test_value).tolist())

  def testMatrixRagged(self):
    test_value = ragged_factory_ops.constant([[u'I love ∰'], [b'a', b'bc']])
    expected_tokens = [[[
        ord('I'),
        ord(' '),
        ord('l'),
        ord('o'),
        ord('v'),
        ord('e'),
        ord(' '),
        ord(u'∰')
    ]], [[ord('a')], [ord('b'), ord('c')]]]
    expected_offset_starts = [[[0, 1, 2, 3, 4, 5, 6, 7]], [[0], [0, 1]]]
    expected_offset_ends = [[[1, 2, 3, 4, 5, 6, 7, 10]], [[1], [1, 2]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def test3DimMatrix(self):
    test_value = constant_op.constant([[[b'!!', b'Good'], [b'ab', b'cd']],
                                       [[b'12', b'3'], [b'scrub', b'a guy']]])
    expected_tokens = [[[[ord('!'), ord('!')],
                         [ord('G'), ord('o'),
                          ord('o'), ord('d')]],
                        [[ord('a'), ord('b')], [ord('c'), ord('d')]]],
                       [[[ord('1'), ord('2')], [ord('3')]],
                        [[ord('s'),
                          ord('c'),
                          ord('r'),
                          ord('u'),
                          ord('b')],
                         [ord('a'),
                          ord(' '),
                          ord('g'),
                          ord('u'),
                          ord('y')]]]]
    expected_offset_starts = [[[[0, 1], [0, 1, 2, 3]], [[0, 1], [0, 1]]],
                              [[[0, 1], [0]], [[0, 1, 2, 3, 4], [0, 1, 2, 3,
                                                                 4]]]]
    expected_offset_ends = [[[[1, 2], [1, 2, 3, 4]], [[1, 2], [1, 2]]],
                            [[[1, 2], [1]], [[1, 2, 3, 4, 5], [1, 2, 3, 4,
                                                               5]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    print(detokenized)
    self.assertAllEqual(
        self.evaluate(detokenized).to_list(),
        self.evaluate(test_value).tolist())

  def test3DimMatrixRagged(self):
    test_value = ragged_factory_ops.constant([[[b'11'], [b'12t', b'13']],
                                              [[b'21', b'22!']]])
    expected_tokens = [[[[ord('1'), ord('1')]],
                        [[ord('1'), ord('2'), ord('t')], [ord('1'),
                                                          ord('3')]]],
                       [[[ord('2'), ord('1')], [ord('2'),
                                                ord('2'),
                                                ord('!')]]]]
    expected_offset_starts = [[[[0, 1]], [[0, 1, 2], [0, 1]]],
                              [[[0, 1], [0, 1, 2]]]]
    expected_offset_ends = [[[[1, 2]], [[1, 2, 3], [1, 2]]],
                            [[[1, 2], [1, 2, 3]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def testInternationalization(self):
    test_value = constant_op.constant(
        [u"J'adore la灯".encode('utf8'), u'¡Escríbeme!'.encode('utf8')])
    expected_tokens = [[
        ord('J'),
        ord("'"),
        ord('a'),
        ord('d'),
        ord('o'),
        ord('r'),
        ord('e'),
        ord(' '),
        ord('l'),
        ord('a'),
        ord(u'灯')
    ],
                       [
                           ord(u'¡'),
                           ord('E'),
                           ord('s'),
                           ord('c'),
                           ord('r'),
                           ord(u'í'),
                           ord('b'),
                           ord('e'),
                           ord('m'),
                           ord('e'),
                           ord('!')
                       ]]
    expected_offset_starts = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]]
    expected_offset_ends = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
                            [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def testOnlySpaces(self):
    test_value = constant_op.constant([b' ', b'   '])
    expected_tokens = [[ord(' ')], [ord(' '), ord(' '), ord(' ')]]
    expected_offset_starts = [[0], [0, 1, 2]]
    expected_offset_ends = [[1], [1, 2, 3]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def testWhitespaceCharacters(self):
    test_value = constant_op.constant([b't\tc\rd\nl'])
    expected_tokens = [[
        ord('t'),
        ord('\t'),
        ord('c'),
        ord('\r'),
        ord('d'),
        ord('\n'),
        ord('l')
    ]]
    expected_offset_starts = [[0, 1, 2, 3, 4, 5, 6]]
    expected_offset_ends = [[1, 2, 3, 4, 5, 6, 7]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

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

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def testEmptyString(self):
    test_value = constant_op.constant([b'', b'I', b'', b'Oh', b''])
    expected_tokens = [[], [ord('I')], [], [ord('O'), ord('h')], []]
    expected_offset_starts = [[], [0], [], [0, 1], []]
    expected_offset_ends = [[], [1], [], [1, 2], []]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)

  def testEmptyDimensions(self):
    test_value = ragged_factory_ops.constant([[[b'F.', b'.'], []], [],
                                              [[b'Zk', b'k'], [b'A', b'a']]])
    expected_tokens = [[[[ord('F'), ord('.')], [ord('.')]], []], [],
                       [[[ord('Z'), ord('k')], [ord('k')]],
                        [[ord('A')], [ord('a')]]]]
    expected_offset_starts = [[[[0, 1], [0]], []], [],
                              [[[0, 1], [0]], [[0], [0]]]]
    expected_offset_ends = [[[[1, 2], [1]], []], [],
                            [[[1, 2], [1]], [[1], [1]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    detokenized = self.tokenizer.detokenize(tokens)
    self.assertAllEqual(detokenized, test_value)


if __name__ == '__main__':
  test.main()
