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
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test

from tensorflow_text.python.ops.split_merge_from_logits_tokenizer import SplitMergeFromLogitsTokenizer  # pylint: disable=line-too-long


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
class SplitMergeFromLogitsTokenizerTest(test.TestCase):

  def setUp(self):
    super(SplitMergeFromLogitsTokenizerTest, self).setUp()
    self.tokenizer = SplitMergeFromLogitsTokenizer()
    self.no_force_split_tokenizer = SplitMergeFromLogitsTokenizer(
        force_split_at_break_character=False)

  def testVectorSingleValue(self):
    test_strings = constant_op.constant([b'IloveFlume!'])

    # Below, each pair of logits [l1, l2] indicates a "split" action
    # if l1 > l2 and a "merge" otherwise.
    test_logits = constant_op.constant([
        [
            # I
            [2.7, -0.3],  # I: split
            # love
            [4.1, 0.82],  # l: split
            [-2.3, 4.3],  # o: merge
            [3.1, 12.2],  # v: merge
            [-3.0, 4.7],  # e: merge
            # Flume
            [2.7, -0.7],  # F: split
            [0.7, 15.0],  # l: merge
            [1.6, 23.0],  # u: merge
            [2.1, 11.0],  # m: merge
            [0.0, 20.0],  # e: merge
            # !
            [18.0, 0.7],  # !: split
        ]])
    expected_tokens = [[b'I', b'love', b'Flume', b'!']]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_strings, test_logits))
    self.assertAllEqual(expected_tokens, tokens)
    extracted_tokens = _RaggedSubstr(test_strings, starts, ends)
    self.assertAllEqual(expected_tokens, extracted_tokens)

  def testVectorSingleValueTokenAcrossSpace(self):
    test_string = b'I love Flume!'
    test_strings = constant_op.constant([test_string])

    # Below, each pair of logits [l1, l2] indicates a "split" action
    # if l1 < l2 and a "merge" otherwise.
    test_logits = constant_op.constant([
        [
            # I
            [2.7, -0.3],  # I: split
            # ' '
            [-1.5, 2.3],  # <space>: merge
            # love
            [4.1, 0.82],  # l: split
            [-2.3, 4.3],  # o: merge
            [3.1, 12.2],  # v: merge
            [-3.0, 4.7],  # e: merge
            # ' '
            [2.5, 32.0],  # <space>: merge
            # Flume
            [-2.7, 5.3],  # F: merge
            [0.7, 15.0],  # l: merge
            [1.6, 23.0],  # u: merge
            [2.1, 11.0],  # m: merge
            [0.0, 20.0],  # e: merge
            # !
            [18.0, 0.7],  # !: split
        ]])

    # By default force_split_at_break_character is set True, so we start new
    # tokens after break characters regardless of the SPLIT/MERGE label of the
    # break character.
    expected_tokens = [[b'I', b'love', b'Flume', b'!']]
    expected_start_offsets = [[0, 2, 7, 12]]
    expected_end_offsets = [[1, 6, 12, 13]]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_strings, test_logits))
    self.assertAllEqual(expected_tokens, tokens)
    self.assertAllEqual(expected_start_offsets, starts)
    self.assertAllEqual(expected_end_offsets, ends)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_strings, test_logits)
    self.assertAllEqual(expected_tokens, tokens)

    # When force_split_at_break_character set false, we may combine two tokens
    # together to form a word according to the label of the first non-space
    # character.
    expected_tokens = [[b'I', b'loveFlume', b'!']]
    expected_start_offsets = [[0, 2, 12]]
    expected_end_offsets = [[1, 12, 13]]
    # Assertions below clarify what the expected offsets mean:
    self.assertEqual(b'I', test_string[0:1])

    # Notice that the original text between the [start, end) offsets for the
    # second token differs from the token text by an extra space: this is
    # by design, that space is not copied in the token.
    self.assertEqual(b'love Flume', test_string[2:12])
    self.assertEqual(b'!', test_string[12:13])

    (tokens, starts, ends) = (
        self.no_force_split_tokenizer.tokenize_with_offsets(
            test_strings, test_logits))
    self.assertAllEqual(expected_tokens, tokens)
    self.assertAllEqual(expected_start_offsets, starts)
    self.assertAllEqual(expected_end_offsets, ends)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.no_force_split_tokenizer.tokenize(test_strings, test_logits)
    self.assertAllEqual(expected_tokens, tokens)

  def testVectorSingleValueTokenChinese(self):
    # TODO(salcianu): clean-up.  We used the Unicode string, but Windows may
    # have problems with it, so we use the utf-8 bytes instead.
    #
    # test_strings = constant_op.constant([_Utf8(u'我在谷歌　写代码')])
    test_strings = constant_op.constant([
        b'\xe6\x88\x91\xe5\x9c\xa8\xe8\xb0\xb7\xe6\xad\x8c'
        + b'\xe3\x80\x80\xe5\x86\x99\xe4\xbb\xa3\xe7\xa0\x81'
    ])

    # Below, each pair of logits [l1, l2] indicates a "split" action
    # if l1 < l2 and a "merge" otherwise.
    test_logits = constant_op.constant([
        [
            # 我
            [2.0, 0.3],  # split
            # 在
            [3.5, 2.1],  # split
            # 谷歌
            [5.0, 1.2],  # split
            [0.4, 3.0],  # merge
            # '　', note this is a full-width space that contains 3 bytes.
            [2.8, 0.0],  # split
            # 写代码
            [6.0, 2.1],  # split
            [2.6, 5.1],  # merge
            [1.0, 7.1],  # merge
        ]])

    # By default force_split_at_break_character is set True, so we start new
    # tokens after break characters regardless of the SPLIT/MERGE label of the
    # break character.
    expected_tokens = [[
        _Utf8(u'我'), _Utf8(u'在'), _Utf8(u'谷歌'), _Utf8(u'写代码')]]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_strings, test_logits))
    self.assertAllEqual(expected_tokens, tokens)

    # Extract tokens according to the returned starts, ends.
    tokens_by_offsets = _RaggedSubstr(test_strings, starts, ends)
    self.assertAllEqual(expected_tokens, tokens_by_offsets)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_strings, test_logits)
    self.assertAllEqual(expected_tokens, tokens)

    # Although force_split_at_break_character is set false we actually predict a
    # SPLIT at '写', so we still start a new token: '写代码'.
    (tokens, starts, ends) = (
        self.no_force_split_tokenizer.tokenize_with_offsets(
            test_strings, test_logits))
    self.assertAllEqual(expected_tokens, tokens)

    # Extract tokens according to the returned starts, ends.
    tokens_by_offsets = _RaggedSubstr(test_strings, starts, ends)
    self.assertAllEqual(expected_tokens, tokens_by_offsets)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.no_force_split_tokenizer.tokenize(test_strings, test_logits)
    self.assertAllEqual(expected_tokens, tokens)

  def testVectorMultipleValues(self):
    test_strings = constant_op.constant([b'IloveFlume!',
                                         b'and tensorflow'])

    # Below, each pair of logits [l1, l2] indicates a "split" action
    # if l1 < l2 and a "merge" otherwise.
    test_logits = constant_op.constant([
        [
            # "I"
            [5.0, -3.2],  # I: split
            # "love"
            [2.2, -1.0],  # l: split
            [0.2, 12.0],  # o: merge
            [0.0, 11.0],  # v: merge
            [-3.0, 3.0],  # e: merge
            # "Flume"
            [10.0, 0.0],  # F: split
            [0.0, 11.0],  # l: merge
            [0.0, 11.0],  # u: merge
            [0.0, 12.0],  # m: merge
            [0.0, 12.0],  # e: merge
            # "!"
            [5.2, -7.0],  # !: split
            # padding:
            [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
        ], [
            # "and"
            [2.0, 0.7],  # a: split
            [0.2, 1.5],  # n: merge
            [0.5, 2.3],  # d: merge
            # " "
            [1.7, 7.0],  # <space>: merge
            # "tensorflow"
            [2.2, 0.1],  # t: split
            [0.2, 3.1],  # e: merge
            [1.1, 2.5],  # n: merge
            [0.7, 0.9],  # s: merge
            [0.6, 1.0],  # o: merge
            [0.3, 1.0],  # r: merge
            [0.2, 2.2],  # f: merge
            [0.7, 3.1],  # l: merge
            [0.4, 5.0],  # o: merge
            [0.8, 6.0],  # w: merge
        ]])
    expected_tokens = [[b'I', b'love', b'Flume', b'!'],
                       [b'and', b'tensorflow']]
    expected_starts = [[0, 1, 5, 10], [0, 4]]
    expected_ends = [[1, 5, 10, 11], [3, 14]]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_strings, test_logits))
    self.assertAllEqual(expected_tokens, tokens)
    tokens_by_offsets = _RaggedSubstr(test_strings, starts, ends)
    self.assertAllEqual(expected_tokens, tokens_by_offsets)
    self.assertAllEqual(expected_starts, starts)
    self.assertAllEqual(expected_ends, ends)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_strings, test_logits)
    self.assertAllEqual(expected_tokens, tokens)

  def testVectorTooFewActions(self):
    test_strings = constant_op.constant([b'IloveFlume!',
                                         b'and tensorflow'])

    # Below, each pair of logits [l1, l2] indicates a "split" action
    # if l1 < l2 and a "merge" otherwise.
    test_logits = constant_op.constant([
        [
            # "I"
            [5.0, -3.2],  # I: split
            # "love"
            [2.2, -1.0],  # l: split
            [0.2, 12.0],  # o: merge
            [0.0, 11.0],  # v: merge
            [-3.0, 3.0],  # e: merge
            # "Flume"
            [10.0, 0.0],  # F: split
            [0.0, 11.0],  # l: merge
            [0.0, 11.0],  # u: merge
            [0.0, 12.0],  # m: merge
            [0.0, 12.0],  # e: merge
            # "!"
            [5.2, -7.0],  # !: split
            # no padding, instead, we truncated the logits for 2nd string.
        ], [
            # "and"
            [2.0, 0.7],  # a: split
            [0.2, 1.5],  # n: merge
            [0.5, 2.3],  # d: merge
            # " "
            [1.7, 7.0],  # <space>: merge
            # "tensorf"; no logits for final three chars, "low".
            [2.2, 0.1],  # t: split
            [0.2, 3.1],  # e: merge
            [1.1, 2.5],  # n: merge
            [0.7, 0.9],  # s: merge
            [0.6, 1.0],  # o: merge
            [0.3, 1.0],  # r: merge
            [0.2, 2.2],  # f: merge
        ]])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r'Number of logits, 11, is insufficient for text "and tensorflow"',
    ):
      self.evaluate(
          self.tokenizer.tokenize_with_offsets(test_strings, test_logits))

  def testTextWithWhitespaces(self):
    # The text from this example contains some whitespaces: we test that we
    # don't generate empty tokens, nor tokens that contain whitespaces.
    test_strings = constant_op.constant([b'\n Ilove Flume! ',
                                         b'and \t\ntensorflow'])

    # Below, each pair of logits [l1, l2] indicates a "split" action
    # if l1 < l2 and a "merge" otherwise.
    test_logits = constant_op.constant([
        [
            # "\n" and " "
            [12.0, 2.1],  # \n: split
            [0.3, 17.3],  # <space>: merge
            # "I"
            [5.0, -3.2],  # I: split
            # "love"
            [2.2, -1.0],  # l: split
            [0.2, 12.0],  # o: merge
            [0.0, 11.0],  # v: merge
            [-3.0, 3.0],  # e: merge
            # " "
            [15.4, 0.3],  # <space>: split
            # "Flume"
            [10.0, 0.0],  # F: split
            [0.0, 11.0],  # l: merge
            [0.0, 11.0],  # u: merge
            [0.0, 12.0],  # m: merge
            [0.0, 12.0],  # e: merge
            # "!"
            [5.2, -7.0],  # !: split
            # " "
            [15.4, 0.3],  # <space>: split
            # padding
            [2.0, 3.0]
        ], [
            # "and"
            [2.0, 0.7],  # a: split
            [0.2, 1.5],  # n: merge
            [0.5, 2.3],  # d: merge
            # " ", "\t", and "\n"
            [1.7, 7.0],  # <space>: merge
            [8.0, 2.1],  # \t: split
            [0.3, 7.3],  # \n: merge
            # "tensorflow"
            [2.2, 0.1],  # t: split
            [0.2, 3.1],  # e: merge
            [1.1, 2.5],  # n: merge
            [0.7, 0.9],  # s: merge
            [0.6, 1.0],  # o: merge
            [0.3, 1.0],  # r: merge
            [0.2, 2.2],  # f: merge
            [0.7, 3.1],  # l: merge
            [0.4, 5.0],  # o: merge
            [0.8, 6.0],  # w: merge
        ]])
    expected_tokens = [[b'I', b'love', b'Flume', b'!'],
                       [b'and', b'tensorflow']]
    (tokens, starts, ends) = (
        self.tokenizer.tokenize_with_offsets(test_strings, test_logits))
    self.assertAllEqual(expected_tokens, tokens)
    tokens_by_offsets = _RaggedSubstr(test_strings, starts, ends)
    self.assertAllEqual(expected_tokens, tokens_by_offsets)

    # Use the same arguments to test the tokenize() version, without offsets.
    tokens = self.tokenizer.tokenize(test_strings, test_logits)
    self.assertAllEqual(expected_tokens, tokens)


if __name__ == '__main__':
  test.main()
