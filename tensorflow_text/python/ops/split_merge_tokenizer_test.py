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
"""Tests for split_merge_tokenizer op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow_text.python.ops.split_merge_tokenizer import SplitMergeTokenizer


def _Utf8(char):
  return char.encode('utf-8')


def _RaggedSubstr(text_input, begin, end):
  text_input_flat = None
  if ragged_tensor.is_ragged(text_input):
    text_input_flat = text_input.flat_values
  else:
    text_input_flat = ops.convert_to_tensor(text_input)

  if ragged_tensor.is_ragged(begin):
    broadcasted_text = array_ops.gather_v2(text_input_flat,
                                           begin.nested_value_rowids()[-1])

    # convert boardcasted_text into a 1D tensor.
    broadcasted_text = array_ops.reshape(broadcasted_text, [-1])
    size = math_ops.sub(end.flat_values, begin.flat_values)
    new_tokens = string_ops.substr_v2(broadcasted_text, begin.flat_values, size)
    return begin.with_flat_values(new_tokens)
  else:
    assert begin.shape.ndims == 1
    assert text_input_flat.shape.ndims == 0
    size = math_ops.sub(end, begin)
    new_tokens = string_ops.substr_v2(text_input_flat, begin, size)
    return new_tokens


@test_util.run_all_in_graph_and_eager_modes
class SplitMergeTokenizerTest(test.TestCase):

  def setUp(self):
    super(SplitMergeTokenizerTest, self).setUp()
    self.tokenizer = SplitMergeTokenizer()

  def testScalarValueSplitMerge(self):
    test_value = b'IloveFlume!'
    test_label = constant_op.constant(
        [
            # I
            0,
            # love
            0, 1, 1, 1,
            # Flume
            0, 1, 1, 1, 1,
            # !
            0
        ])
    expected_tokens = [b'I', b'love', b'Flume', b'!']
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value, test_label))
    self.assertAllEqual(tokens, expected_tokens)
    extracted_tokens = _RaggedSubstr(test_value, starts, ends)
    self.assertAllEqual(extracted_tokens, expected_tokens)

  def testVectorSingleValueSplitMerge(self):
    test_value = constant_op.constant([b'IloveFlume!'])
    test_label = constant_op.constant([
        [
            # I
            0,
            # love
            0, 1, 1, 1,
            # Flume
            0, 1, 1, 1, 1,
            # !
            0
        ]])
    expected_tokens = [[b'I', b'love', b'Flume', b'!']]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value, test_label))
    self.assertAllEqual(tokens, expected_tokens)
    extracted_tokens = _RaggedSubstr(test_value, starts, ends)
    self.assertAllEqual(extracted_tokens, expected_tokens)

  def testVectorSingleValueTokenCrossSpace(self):
    test_string = b'I love Flume!'
    test_value = constant_op.constant([test_string])
    test_label = constant_op.constant([
        [
            # I
            0,
            # ' '
            1,
            # love
            0, 1, 1, 1,
            # ' '
            0,
            # Flume
            1, 1, 1, 1, 1,
            # !
            0
        ]])

    # By default force_split_at_break_character is set True, so we start new
    # tokens after break characters regardless of the SPLIT/MERGE label of the
    # break character.
    expected_tokens = [[b'I', b'love', b'Flume', b'!']]
    expected_offset_starts = [[0, 2, 7, 12]]
    expected_offset_ends = [[1, 6, 12, 13]]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value, test_label))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_value, test_label)
    self.assertAllEqual(tokens, expected_tokens)

    # When force_split_at_break_character set false, we may combine two tokens
    # together to form a word according to the label of the first non-space
    # character.
    expected_tokens = [[b'I', b'loveFlume', b'!']]
    expected_offset_starts = [[0, 2, 12]]
    expected_offset_ends = [[1, 12, 13]]
    # Assertions below clarify what the expected offsets mean:
    self.assertEqual(test_string[0:1], b'I')

    # Notice that the original text between the [start, end) offsets for the
    # second token differs from the token text by an extra space: this is
    # by design, that space is not copied in the token.
    self.assertEqual(test_string[2:12], b'love Flume')
    self.assertEqual(test_string[12:13], b'!')

    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(
            test_value, test_label, force_split_at_break_character=False))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(
        test_value, test_label, force_split_at_break_character=False)
    self.assertAllEqual(tokens, expected_tokens)

  def testVectorSingleValueTokenChinese(self):
    # TODO(salcianu): clean-up.  We used the Unicode string, but Windows may
    # have problems with it, so we use the utf-8 bytes instead.
    #
    # test_value = constant_op.constant([_Utf8(u'我在谷歌　写代码')])
    test_value = constant_op.constant([
        b'\xe6\x88\x91\xe5\x9c\xa8\xe8\xb0\xb7\xe6\xad\x8c'
        + b'\xe3\x80\x80\xe5\x86\x99\xe4\xbb\xa3\xe7\xa0\x81'
    ])
    test_label = constant_op.constant([
        [
            # 我
            0,
            # 在
            0,
            # 谷歌
            0, 1,
            # '　', note this is a full-width space that contains 3 bytes.
            0,
            # 写代码
            0, 1, 1
        ]])

    # By default force_split_at_break_character is set True, so we start new
    # tokens after break characters regardless of the SPLIT/MERGE label of the
    # break character.
    expected_tokens = [[
        _Utf8(u'我'), _Utf8(u'在'), _Utf8(u'谷歌'), _Utf8(u'写代码')]]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value, test_label))
    self.assertAllEqual(tokens, expected_tokens)

    # Extract tokens according to the returned starts, ends.
    tokens_by_offsets = _RaggedSubstr(test_value, starts, ends)
    self.assertAllEqual(expected_tokens, tokens_by_offsets)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_value, test_label)
    self.assertAllEqual(tokens, expected_tokens)

    # Although force_split_at_break_character is set false we actually predict a
    # SPLIT at '写', so we still start a new token: '写代码'.
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(
            test_value, test_label, force_split_at_break_character=False))
    self.assertAllEqual(tokens, expected_tokens)

    # Extract tokens according to the returned starts, ends.
    tokens_by_offsets = _RaggedSubstr(test_value, starts, ends)
    self.assertAllEqual(expected_tokens, tokens_by_offsets)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(
        test_value, test_label, force_split_at_break_character=False)
    self.assertAllEqual(tokens, expected_tokens)

  def testHigherRank(self):
    # [2, 1]
    test_value = constant_op.constant([[b'IloveFlume!'],
                                       [b'and tensorflow']])
    test_label = constant_op.constant([
        [[
            # I
            0,
            # love
            0, 1, 1, 1,
            # Flume
            0, 1, 1, 1, 1,
            # !
            0,
            # paddings
            0, 0, 0
        ]], [[
            # and
            0, 1, 1,
            # ' '
            1,
            # tensorflow
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]]])
    expected_tokens = [[[b'I', b'love', b'Flume', b'!']],
                       [[b'and', b'tensorflow']]]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value, test_label))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual([[[0, 1, 5, 10]], [[0, 4]]], starts)
    self.assertAllEqual([[[1, 5, 10, 11]], [[3, 14]]], ends)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_value, test_label)
    self.assertAllEqual(tokens, expected_tokens)

  def testVectorMultipleValue(self):
    test_value = constant_op.constant([b'IloveFlume!',
                                       b'and tensorflow'])
    test_label = constant_op.constant([
        [
            # I
            0,
            # love
            0, 1, 1, 1,
            # Flume
            0, 1, 1, 1, 1,
            # !
            0,
            # paddings
            0, 0, 0
        ], [
            # and
            0, 1, 1,
            # ' '
            1,
            # tensorflow
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]])
    expected_tokens = [[b'I', b'love', b'Flume', b'!'],
                       [b'and', b'tensorflow']]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value, test_label))
    self.assertAllEqual(tokens, expected_tokens)
    tokens_by_offsets = _RaggedSubstr(test_value, starts, ends)
    self.assertAllEqual(tokens_by_offsets, expected_tokens)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_value, test_label)
    self.assertAllEqual(tokens, expected_tokens)

  def testRaggedInput(self):
    test_value = ragged_factory_ops.constant([
        [b'IloveFlume!', b'and tensorflow'],
        [b'go raggedtensor']
    ])
    test_label = ragged_factory_ops.constant([
        [
            [
                # I
                0,
                # love
                0, 1, 1, 1,
                # Flume
                0, 1, 1, 1, 1,
                # !
                0,
                # paddings
                0, 0, 0
            ], [
                # and
                0, 1, 1,
                # ' '
                1,
                # tensorflow
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]
        ],
        [
            [
                # go
                0, 1,
                # ' '
                0,
                # ragged
                0, 1, 1, 1, 1, 1,
                # tensor
                0, 1, 1, 1, 1, 1,
            ]
        ]])
    expected_tokens = [
        [[b'I', b'love', b'Flume', b'!'], [b'and', b'tensorflow']],
        [[b'go', b'ragged', b'tensor']]
    ]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value, test_label))
    self.assertAllEqual(tokens, expected_tokens)
    tokens_by_offsets = _RaggedSubstr(test_value, starts, ends)
    self.assertAllEqual(tokens_by_offsets, expected_tokens)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_value, test_label)
    self.assertAllEqual(tokens, expected_tokens)

  def testRaggedInputHigherRank(self):
    test_value = ragged_factory_ops.constant([
        [[b'IloveFlume!', b'and tensorflow']],
        [[b'go raggedtensor']]
    ])
    test_label = ragged_factory_ops.constant([
        [
            [[
                # I
                0,
                # love
                0, 1, 1, 1,
                # Flume
                0, 1, 1, 1, 1,
                # !
                0,
                # paddings
                0, 0, 0
            ], [
                # and
                0, 1, 1,
                # ' '
                1,
                # tensorflow
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]]
        ],
        [
            [[
                # go
                0, 1,
                # ' '
                0,
                # ragged
                0, 1, 1, 1, 1, 1,
                # tensor
                0, 1, 1, 1, 1, 1,
            ]]
        ]])
    expected_tokens = [
        [[[b'I', b'love', b'Flume', b'!'], [b'and', b'tensorflow']]],
        [[[b'go', b'ragged', b'tensor']]]
    ]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_value, test_label))
    self.assertAllEqual(tokens, expected_tokens)
    tokens_by_offsets = _RaggedSubstr(test_value, starts, ends)
    self.assertAllEqual(tokens_by_offsets, expected_tokens)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_value, test_label)
    self.assertAllEqual(tokens, expected_tokens)


if __name__ == '__main__':
  test.main()
