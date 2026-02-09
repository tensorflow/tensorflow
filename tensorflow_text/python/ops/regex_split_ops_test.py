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

# encoding=utf-8
"""Tests for regex_split and regex_split_with_offsets ops."""
from absl.testing import parameterized

import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow_text.python.ops import regex_split_ops


def _utf8(char):
  return char.encode("utf-8")


# TODO(thuang513): It appears there isn't a Ragged version of substr; consider
#               checking this into core TF.
def _ragged_substr(text_input, begin, size):
  if not (isinstance(text_input, ragged_tensor.RaggedTensor) or
          isinstance(begin, ragged_tensor.RaggedTensor) or
          isinstance(size, ragged_tensor.RaggedTensor)):
    return string_ops.substr_v2(text_input, begin, size)

  # TODO(edloper) Update this to use ragged_tensor_shape.broadcast_dynamic_shape
  # once it's been updated to handle uniform_row_lengths correctly.
  if ragged_tensor.is_ragged(text_input):
    if text_input.ragged_rank != 1 or text_input.shape.rank != 2:
      return None  # Test only works for `shape=[N, None]`
    text_input_flat = text_input.flat_values
  else:
    text_input_flat = array_ops.reshape(text_input, [-1])
  broadcasted_text = array_ops.gather_v2(text_input_flat,
                                         begin.nested_value_rowids()[-1])
  new_tokens = string_ops.substr_v2(broadcasted_text, begin.flat_values,
                                    size.flat_values)
  return begin.with_flat_values(new_tokens)


@test_util.run_all_in_graph_and_eager_modes
class RegexSplitOpsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters([
      dict(
          text_input=[r"hello there"],
          delim_regex_pattern=r"\s",
          keep_delim_regex_pattern=r"\s",
          expected=[[b"hello", b" ", b"there"]],
      ),
      dict(
          text_input=[r"hello there"],
          delim_regex_pattern=r"\s",
          expected=[[b"hello", b"there"]],
      ),
      dict(
          text_input=[r"hello  there"],
          delim_regex_pattern=r"\s",
          expected=[[b"hello", b"there"]],
      ),
      dict(
          text_input=[_utf8(u"では４日")],
          delim_regex_pattern=r"\p{Hiragana}",
          keep_delim_regex_pattern=r"\p{Hiragana}",
          expected=[[_utf8(u"で"), _utf8(u"は"),
                     _utf8(u"４日")]],
      ),
      dict(
          text_input=[r"hello! (:$) there"],
          delim_regex_pattern=r"[\p{S}|\p{P}]+|\s",
          keep_delim_regex_pattern=r"[\p{S}|\p{P}]+",
          expected=[[b"hello", b"!", b"(:$)", b"there"]],
      ),
      dict(
          text_input=[r"hello12345there"],
          delim_regex_pattern=r"\p{N}+",
          keep_delim_regex_pattern=r"\p{N}+",
          expected=[[b"hello", b"12345", b"there"]],
      ),
      dict(
          text_input=[r"show me some $100 bills yo!"],
          delim_regex_pattern=r"\s|\p{S}",
          keep_delim_regex_pattern=r"\p{S}",
          expected=[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"]],
      ),
      dict(
          text_input=[
              [b"show me some $100 bills yo!",
               _utf8(u"では４日")],
              [b"hello there"],
          ],
          delim_regex_pattern=r"\s|\p{S}|\p{Hiragana}",
          keep_delim_regex_pattern=r"\p{S}|\p{Hiragana}",
          expected=[[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"],
                     [_utf8(u"で"), _utf8(u"は"),
                      _utf8(u"４日")]], [[b"hello", b"there"]]],
      ),
      dict(
          text_input=[[
              [b"show me some $100 bills yo!",
               _utf8(u"では４日")],
              [b"hello there"],
          ]],
          delim_regex_pattern=r"\s|\p{S}|\p{Hiragana}",
          keep_delim_regex_pattern=r"\p{S}|\p{Hiragana}",
          expected=[[[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"],
                      [_utf8(u"で"), _utf8(u"は"), _utf8(u"４日")]],
                     [[b"hello", b"there"]]]],
      ),
      dict(
          text_input=[
              [[b"a b", b"c"], [b"d", b"e f g"]],
              [[b"cat horse cow", b""]]],
          ragged_rank=1,
          delim_regex_pattern=r"\s",
          expected=[
              [[[b"a", b"b"], [b"c"]], [[b"d"], [b"e", b"f", b"g"]]],
              [[[b"cat", b"horse", b"cow"], []]]],
      ),
      # Test inputs that are Tensors.
      dict(
          text_input=[
              r"show me some $100 bills yo!",
              r"hello there",
          ],
          delim_regex_pattern=r"\s|\p{S}",
          keep_delim_regex_pattern=r"\p{S}",
          expected=[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"],
                    [b"hello", b"there"]],
          input_is_dense=True,
      ),
      dict(
          text_input=[
              [r"show me some $100 bills yo!"],
              [r"hello there"],
          ],
          delim_regex_pattern=r"\s|\p{S}",
          keep_delim_regex_pattern=r"\p{S}",
          expected=[[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"]],
                    [[b"hello", b"there"]]],
          input_is_dense=True,
      ),
      dict(
          input_is_dense=True,
          text_input=[
              [b"show me some $100 bills yo!",
               _utf8(u"では４日")],
              [b"hello there", b"woot woot"],
          ],
          delim_regex_pattern=r"\s|\p{S}|\p{Hiragana}",
          keep_delim_regex_pattern=r"\p{S}|\p{Hiragana}",
          expected=[[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"],
                     [_utf8(u"で"), _utf8(u"は"),
                      _utf8(u"４日")]], [[b"hello", b"there"], [b"woot",
                                                              b"woot"]]],
      ),
      dict(
          input_is_dense=True,
          text_input=[
              [[b"show me some $100 bills yo!"], [_utf8(u"では４日")]],
              [[b"hello there"], [b"woot woot"]],
          ],
          delim_regex_pattern=r"\s|\p{S}|\p{Hiragana}",
          keep_delim_regex_pattern=r"\p{S}|\p{Hiragana}",
          # expected shape = [2, 2, 1, ]
          expected=[[[[b"show", b"me", b"some", b"$", b"100", b"bills",
                       b"yo!"]], [[_utf8(u"で"),
                                   _utf8(u"は"),
                                   _utf8(u"４日")]]],
                    [[[b"hello", b"there"]], [[b"woot", b"woot"]]]],
      ),
  ])
  def testRegexSplitOp(self,
                       text_input,
                       delim_regex_pattern,
                       expected,
                       keep_delim_regex_pattern=r"",
                       input_is_dense=False,
                       ragged_rank=None):
    if input_is_dense:
      text_input = constant_op.constant(text_input)
    else:
      text_input = ragged_factory_ops.constant(text_input,
                                               ragged_rank=ragged_rank)

    actual_tokens, start, end = regex_split_ops.regex_split_with_offsets(
        input=text_input,
        delim_regex_pattern=delim_regex_pattern,
        keep_delim_regex_pattern=keep_delim_regex_pattern,
    )
    self.assertAllEqual(actual_tokens, expected)

    # Use the offsets to extract substrings and verify that the substrings match
    # up with the expected tokens
    extracted_tokens = _ragged_substr(array_ops.expand_dims(text_input, -1),
                                      start, end - start)
    if extracted_tokens is not None:
      self.assertAllEqual(extracted_tokens, expected)

  def testRegexSplitWithInvalidRegex(self):
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Invalid pattern.*"):
      result = regex_split_ops.regex_split("<img_1><img_2><img_3>", ">(?=<)")
      self.evaluate(result)


@test_util.run_all_in_graph_and_eager_modes
class RegexSplitterTestCases(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          text_input=[
              b"Hi there.\nWhat time is it?\nIt is gametime.",
              b"Who let the dogs out?\nWho?\nWho?\nWho?",
          ],
          expected=[[b"Hi there.", b"What time is it?", b"It is gametime."],
                    [b"Who let the dogs out?", b"Who?", b"Who?", b"Who?"]],
      ),
      dict(
          text_input=[
              b"Hi there.\nWhat time is it?\nIt is gametime.",
              b"Who let the dogs out?\nWho?\nWho?\nWho?\n",
          ],
          expected=[[b"Hi there.", b"What time is it?", b"It is gametime."],
                    [b"Who let the dogs out?", b"Who?", b"Who?", b"Who?"]],
      ),
      dict(
          text_input=[
              b"Hi there.\r\nWhat time is it?\r\nIt is gametime.",
              b"Who let the dogs out?\r\nWho?\r\nWho?\r\nWho?",
          ],
          expected=[[b"Hi there.", b"What time is it?", b"It is gametime."],
                    [b"Who let the dogs out?", b"Who?", b"Who?", b"Who?"]],
          new_sentence_regex="\r\n",
      ),
  ])
  def testRegexSplitter(self,
                        text_input,
                        expected,
                        new_sentence_regex=None):
    text_input = constant_op.constant(text_input)
    sentence_breaker = regex_split_ops.RegexSplitter(new_sentence_regex)
    actual = sentence_breaker.split(text_input)
    self.assertAllEqual(actual, expected)


if __name__ == "__main__":
  tf.test.main()
