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

"""Tests for sentence_breaking_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow_text.python.ops import sentence_breaking_ops


@test_util.run_all_in_graph_and_eager_modes
class SentenceFragmenterTestCasesV1(test.TestCase, parameterized.TestCase):

  def getTokenWord(self, text, token_starts, token_ends):
    def _FindSubstr(input_tensor):
      text, token_start, token_length = input_tensor
      return string_ops.substr(text, token_start, token_length)

    token_lengths = token_ends - token_starts
    token_word = ragged_map_ops.map_fn(
        _FindSubstr, (text, token_starts, token_lengths),
        dtype=ragged_tensor.RaggedTensorType(
            dtype=dtypes.string, ragged_rank=1),
        infer_shape=False)
    return token_word

  def getTokenOffsets(self, token_words):
    result_start = []
    result_end = []
    for sentence in token_words:
      sentence_string = ""
      sentence_start = []
      sentence_end = []
      for word in sentence:
        sentence_start.append(len(sentence_string))
        sentence_string = sentence_string.join([word, " "])
        sentence_end.append(len(sentence_string))
      result_start.append(sentence_start)
      result_end.append(sentence_end)
    return (constant_op.constant(result_start, dtype=dtypes.int64),
            constant_op.constant(result_end, dtype=dtypes.int64))

  @parameterized.parameters([
      dict(
          test_description="Test acronyms.",
          text=[["Welcome to the U.S. don't be surprised."]],
          token_starts=[[0, 8, 11, 15, 20, 26, 29, 38]],
          token_ends=[[7, 10, 14, 19, 25, 28, 38, 39]],
          token_properties=[[0, 0, 0, 256, 0, 0, 0, 0]],
          expected_fragment_start=[[0, 4]],
          expected_fragment_end=[[4, 8]],
          expected_fragment_properties=[[1, 1]],
          expected_terminal_punc=[[3, 7]],
      ),
      dict(
          test_description="Test batch containing acronyms.",
          text=[["Welcome to the U.S. don't be surprised."], ["I.B.M. yo"]],
          token_starts=[[0, 8, 11, 15, 20, 26, 29, 38], [0, 7]],
          token_ends=[[7, 10, 14, 19, 25, 28, 38, 39], [6, 9]],
          token_properties=[[0, 0, 0, 256, 0, 0, 0, 0], [0, 0]],
          expected_fragment_start=[[0, 4], [0]],
          expected_fragment_end=[[4, 8], [2]],
          expected_fragment_properties=[[1, 1], [0]],
          expected_terminal_punc=[[3, 7], [-1]],
      ),
      dict(
          test_description="Test for semicolons.",
          text=[["Welcome to the US; don't be surprised."]],
          token_starts=[[0, 8, 11, 15, 17, 19, 25, 28, 37]],
          token_ends=[[8, 10, 14, 19, 18, 24, 27, 37, 38]],
          token_properties=[[0, 0, 0, 0, 0, 0, 0, 0, 0]],
          expected_fragment_start=[[0]],
          expected_fragment_end=[[9]],
          expected_fragment_properties=[[1]],
          expected_terminal_punc=[[8]],
      ),
  ])
  def testSentenceFragmentOp(self, test_description, text, token_starts,
                             token_ends, token_properties,
                             expected_fragment_start, expected_fragment_end,
                             expected_fragment_properties,
                             expected_terminal_punc):
    text = constant_op.constant(text)
    token_starts = ragged_factory_ops.constant(token_starts, dtype=dtypes.int64)
    token_ends = ragged_factory_ops.constant(token_ends, dtype=dtypes.int64)
    token_properties = ragged_factory_ops.constant(
        token_properties, dtype=dtypes.int64)
    token_word = self.getTokenWord(text, token_starts, token_ends)

    fragments = sentence_breaking_ops.sentence_fragments(
        token_word, token_starts, token_ends, token_properties)

    fragment_starts, fragment_ends, fragment_properties, terminal_punc = (
        fragments)
    self.assertAllEqual(expected_fragment_start, fragment_starts)
    self.assertAllEqual(expected_fragment_end, fragment_ends)
    self.assertAllEqual(expected_fragment_properties, fragment_properties)
    self.assertAllEqual(expected_terminal_punc, terminal_punc)

  @parameterized.parameters([
      dict(
          test_description="Test acronyms.",
          token_word=[
              ["Welcome", "to", "the", "U.S.", "!", "Harry"],
              ["Wb", "Tang", "Clan", ";", "ain't", "nothing"],
          ],
          token_properties=[[0, 0, 0, 256, 0, 0], [0, 0, 0, 0, 0, 0]],
          expected_fragment_start=[[0, 5], [0]],
          expected_fragment_end=[[5, 6], [6]],
          expected_fragment_properties=[[3, 0], [0]],
          expected_terminal_punc=[[3, -1], [-1]],
      ),
  ])
  def testDenseInputs(self, test_description, token_word, token_properties,
                      expected_fragment_start, expected_fragment_end,
                      expected_fragment_properties, expected_terminal_punc):
    token_starts, token_ends = self.getTokenOffsets(token_word)
    token_properties = constant_op.constant(
        token_properties, dtype=dtypes.int64)
    token_word = constant_op.constant(token_word, dtype=dtypes.string)

    fragments = sentence_breaking_ops.sentence_fragments(
        token_word, token_starts, token_ends, token_properties)

    fragment_starts, fragment_ends, fragment_properties, terminal_punc = (
        fragments)
    self.assertAllEqual(expected_fragment_start, fragment_starts)
    self.assertAllEqual(expected_fragment_end, fragment_ends)
    self.assertAllEqual(expected_fragment_properties, fragment_properties)
    self.assertAllEqual(expected_terminal_punc, terminal_punc)

  @parameterized.parameters([
      dict(
          test_description="Too many ragged ranks.",
          token_word=[
              ["Welcome", "to", "the", "U.S.", "don't", "be", "surprised"],
          ],
          token_starts=[[1, 2, 3]],
          token_ends=[[[7, 10, 14, 19, 25, 28, 38, 39]]],
          token_properties=[[0, 0, 0, 256, 0, 0, 0, 0]],
      ),
      dict(
          test_description="Too many ranks in a dense Tensor.",
          token_word=[
              [[["Welcome", "to", "the", "U.S.", "don't", "be", "surprised"]]],
          ],
          token_starts=[[1, 2, 3]],
          token_ends=[[7, 10, 14, 19, 25, 28, 38, 39]],
          token_properties=[[0, 0, 0, 256, 0, 0, 0, 0]],
          is_ragged=False,
      ),
  ])
  def testBadInputShapes(self,
                         test_description,
                         token_word,
                         token_starts,
                         token_ends,
                         token_properties,
                         is_ragged=True):
    constant = ragged_factory_ops.constant if is_ragged else constant_op.constant
    token_starts = constant(token_starts, dtype=dtypes.int64)
    token_ends = constant(token_ends, dtype=dtypes.int64)
    token_properties = ragged_factory_ops.constant(
        token_properties, dtype=dtypes.int64)

    with self.assertRaises(errors.InvalidArgumentError):
      result = sentence_breaking_ops.sentence_fragments(
          token_word, token_starts, token_ends, token_properties)
      _ = self.evaluate(result)

if __name__ == "__main__":
  test.main()
