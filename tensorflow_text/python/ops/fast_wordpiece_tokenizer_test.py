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
"""Tests for fast_wordpiece_tokenizer op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow_text.python.ops.fast_wordpiece_tokenizer import FastWordpieceTokenizer

FLAGS = flags.FLAGS


def _Utf8(char):
  return char.encode("utf-8")


_ENGLISH_VOCAB = [
    b"don",
    b"##'",
    b"##t",
    b"tread",
    b"##ness",
    b"hel",
    b"##lo",
    b"there",
    b"my",
    b"na",
    b"##me",
    b"is",
    b"ter",
    b"##ry",
    b"what",
    b"##cha",
    b"##ma",
    b"##call",
    b"##it?",
    b"you",
    b"said",
    b"[UNK]",
]

_CHINESE_VOCAB = [
    _Utf8(u"貿"),
    _Utf8(u"易"),
    _Utf8(u"戰"),
    _Utf8(u"最"),
    _Utf8(u"大"),
    _Utf8(u"受"),
    _Utf8(u"益"),
    _Utf8(u"者"),
    _Utf8(u"越"),
    _Utf8(u"南"),
    _Utf8(u"總"),
    _Utf8(u"理"),
    _Utf8(u"阮"),
    _Utf8(u"春"),
    _Utf8(u"福"),
    "[UNK]",
]

_MIXED_LANG_VOCAB = [
    b"don",
    b"##'",
    b"##t",
    b"tread",
    b"##ness",
    b"hel",
    b"##lo",
    b"there",
    b"my",
    b"na",
    b"##me",
    b"is",
    b"ter",
    b"##ry",
    b"what",
    b"##cha",
    b"##ma",
    b"##call",
    b"##it?",
    b"you",
    b"said",
    _Utf8(u"貿"),
    _Utf8(u"易"),
    _Utf8(u"戰"),
    _Utf8(u"最"),
    _Utf8(u"大"),
    _Utf8(u"受"),
    _Utf8(u"益"),
    _Utf8(u"者"),
    _Utf8(u"越"),
    _Utf8(u"南"),
    _Utf8(u"總"),
    _Utf8(u"理"),
    _Utf8(u"阮"),
    _Utf8(u"春"),
    _Utf8(u"福"),
    "[UNK]",
]

_RUSSIAN_VOCAB = [
    _Utf8(u"к"),
    _Utf8(u"##уп"),
    _Utf8(u"##иха"),
    "[UNK]",
]

# Vocab with Unicode chars that crashed ICU in the past.
_DEATH_VOCAB = [
    _Utf8(u"क"),
    _Utf8(u"##र"),
    _Utf8(u"##े"),
    _Utf8(u"##ं"),
    b"##*",
    _Utf8(u"##👇"),
    "[UNK]",
]


def _GetTokensFromWordpieceOffsets(tokens, begin_indices, end_indices):
  begin_indices = begin_indices.to_list()
  end_indices = end_indices.to_list()
  result = []
  for docs_idx in range(0, len(tokens)):
    tokens_in_doc = []
    for tokens_idx in range(0, len(tokens[docs_idx])):
      token = bytes(tokens[docs_idx][tokens_idx])
      begin_offsets = begin_indices[docs_idx][tokens_idx]
      end_offsets = end_indices[docs_idx][tokens_idx]
      tokens_in_doc.append(b"".join(
          [token[begin:end] for begin, end in zip(begin_offsets, end_offsets)]))
    result.append(tokens_in_doc)
  return result


class FastWordpieceOpOriginalTest(test_util.TensorFlowTestCase,
                                  parameterized.TestCase):
  """Adapted from the original WordpieceTokenizer tests."""

  @parameterized.parameters([
      # Basic case
      dict(
          tokens=[[_Utf8(u"купиха")]],
          expected_subwords=[[[
              _Utf8(u"к"),
              _Utf8(u"##уп"),
              _Utf8(u"##иха"),
          ]]],
          vocab=_RUSSIAN_VOCAB,
      ),
      dict(
          tokens=[[b"don't", b"treadness"]],
          expected_subwords=[[[b"don", b"##'", b"##t"], [b"tread", b"##ness"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=[[b"hello", b"there", b"my", b"name", b"is", b"terry"],
                  [b"whatchamacallit?", b"you", b"said"]],
          expected_subwords=[[[b"hel", b"##lo"], [b"there"], [b"my"],
                              [b"na", b"##me"], [b"is"], [b"ter", b"##ry"]],
                             [[b"what", b"##cha", b"##ma", b"##call", b"##it?"],
                              [b"you"], [b"said"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      # Basic case w/ unknown token
      dict(
          tokens=[[b"don't", b"tread", b"cantfindme", b"treadcantfindme"]],
          expected_subwords=[[[b"don", b"##'", b"##t"], [b"tread"], [b"[UNK]"],
                              [b"[UNK]"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      # Basic case w/ int id lookup
      dict(
          tokens=[[b"don't", b"tread", b"cantfindme", b"treadcantfindme"]],
          token_out_type=dtypes.int64,
          expected_subwords=[[[0, 1, 2], [3], [21], [21]]],
          vocab=_ENGLISH_VOCAB,
      ),
      # Chinese test case
      dict(
          tokens=[[
              _Utf8(u"貿"),
              _Utf8(u"易"),
              _Utf8(u"戰"),
              _Utf8(u"最"),
              _Utf8(u"大"),
              _Utf8(u"受"),
              _Utf8(u"益"),
              _Utf8(u"者")
          ],
                  [
                      _Utf8(u"越"),
                      _Utf8(u"南"),
                      _Utf8(u"總"),
                      _Utf8(u"理"),
                      _Utf8(u"阮"),
                      _Utf8(u"春"),
                      _Utf8(u"福")
                  ]],
          expected_subwords=[[[_Utf8(u"貿")], [_Utf8(u"易")], [_Utf8(u"戰")],
                              [_Utf8(u"最")], [_Utf8(u"大")], [_Utf8(u"受")],
                              [_Utf8(u"益")], [_Utf8(u"者")]],
                             [[_Utf8(u"越")], [_Utf8(u"南")], [_Utf8(u"總")],
                              [_Utf8(u"理")], [_Utf8(u"阮")], [_Utf8(u"春")],
                              [_Utf8(u"福")]]],
          vocab=_CHINESE_VOCAB,
      ),
      # Mixed lang test cases
      dict(
          tokens=[
              [
                  _Utf8(u"貿"),
                  _Utf8(u"易"),
                  _Utf8(u"戰"),
                  _Utf8(u"最"),
                  _Utf8(u"大"),
                  _Utf8(u"受"),
                  _Utf8(u"益"),
                  _Utf8(u"者")
              ],
              [
                  _Utf8(u"越"),
                  _Utf8(u"南"),
                  _Utf8(u"總"),
                  _Utf8(u"理"),
                  _Utf8(u"阮"),
                  _Utf8(u"春"),
                  _Utf8(u"福")
              ],
              [b"don't", b"treadness"],
          ],
          expected_subwords=[
              [[_Utf8(u"貿")], [_Utf8(u"易")], [_Utf8(u"戰")],
               [_Utf8(u"最")], [_Utf8(u"大")], [_Utf8(u"受")],
               [_Utf8(u"益")], [_Utf8(u"者")]],
              [[_Utf8(u"越")], [_Utf8(u"南")], [_Utf8(u"總")],
               [_Utf8(u"理")], [_Utf8(u"阮")], [_Utf8(u"春")],
               [_Utf8(u"福")]],
              [[b"don", b"##'", b"##t"], [b"tread", b"##ness"]],
          ],
          vocab=_MIXED_LANG_VOCAB,
      ),
      # Test token whose size is > max_bytes_per_word. When "[UNK]" is returned,
      # FastWordpieceTokenizer sets the end_offset as the length of the input
      # word. This is different from the original WordpieceTokenizer. See the
      # comments of the FastWordpieceTokenizer class.
      dict(
          tokens=[[b"don't", b"treadness"]],
          expected_subwords=[[[b"don", b"##'", b"##t"], [b"[UNK]"]]],
          vocab=_ENGLISH_VOCAB,
          max_bytes_per_word=5,
          # Explicitly specify the offsets here because the current way of
          # testing offsets would require '[UNK]' to be part of tokens.
          expected_start=[[[0, 3, 4], [0]]],
          expected_end=[[[3, 4, 5], [9]]],
      ),
      # Test the token of death usecase.
      dict(
          tokens=[[_Utf8(u"करें*👇👇")]],
          token_out_type=dtypes.string,
          expected_subwords=[[[
              _Utf8(u"क"),
              _Utf8(u"##र"),
              _Utf8(u"##े"),
              _Utf8(u"##ं"), b"##*",
              _Utf8(u"##👇"),
              _Utf8(u"##👇")
          ]]],
          vocab=_DEATH_VOCAB,
          max_bytes_per_word=40,
      ),
      # Test not splitting out unknown characters.
      # (p and ! are unknown)
      dict(
          tokens=[[b"nap", b"hello!me"]],
          expected_subwords=[[[b"[UNK]"], [b"[UNK]"]]],
          unknown_token="[UNK]",
          vocab=_ENGLISH_VOCAB,
      ),
  ])
  def testWordPieceOpAndVerifyOffsets(self,
                                      tokens,
                                      expected_subwords,
                                      vocab,
                                      expected_start=None,
                                      expected_end=None,
                                      unknown_token="[UNK]",
                                      token_out_type=dtypes.string,
                                      max_bytes_per_word=100):
    tokens_t = ragged_factory_ops.constant(tokens)
    tokenizer = FastWordpieceTokenizer(
        vocab=vocab,
        unknown_token=unknown_token,
        token_out_type=token_out_type,
        max_bytes_per_word=max_bytes_per_word,
        no_pretokenization=True
    )
    subwords_t, begin_t, end_t = tokenizer.tokenize_with_offsets(tokens_t)
    self.assertAllEqual(subwords_t, expected_subwords)

    # Verify the indices by performing the following:
    # - Extract subwords and join them together to form the original tokens.
    # - Then compare the extracted tokens and original tokens.
    begin, end = (self.evaluate((begin_t, end_t)))

    # If expected start/end offsets were provided, check them explicitly.
    # Otherwise test the offsets by extracting subwords using token offsets
    # from the original 'tokens' input.
    if expected_start is None or expected_end is None:
      extracted_tokens = _GetTokensFromWordpieceOffsets(tokens, begin, end)
      self.assertAllEqual(extracted_tokens, tokens)
    else:
      self.assertAllEqual(begin, expected_start)
      self.assertAllEqual(end, expected_end)

  @parameterized.parameters([
      dict(
          tokens=[[[b"don't"], [b"treadness"],
                   [b"whatchamacallit?", b"you", b"hello"]], [[b"treadness"]]],
          expected_subwords=[
              [[[b"don", b"##'", b"##t"]], [[b"tread", b"##ness"]],
               [[b"what", b"##cha", b"##ma", b"##call", b"##it?"], [b"you"],
                [b"hel", b"##lo"]]], [[[b"tread", b"##ness"]]]
          ],
          vocab=_ENGLISH_VOCAB,
      ),
  ])
  def testWordPieceOpWithMultipleRaggedRank(self,
                                            tokens,
                                            expected_subwords,
                                            vocab,
                                            expected_start=None,
                                            expected_end=None,
                                            token_out_type=dtypes.string):
    for row_splits_dtype in (dtypes.int32, dtypes.int64):
      ragged_tokens = ragged_factory_ops.constant(
          tokens, row_splits_dtype=row_splits_dtype)
      tokenizer = FastWordpieceTokenizer(
          vocab=vocab, token_out_type=token_out_type,
          no_pretokenization=True)
      subwords = tokenizer.tokenize(ragged_tokens)
      self.assertAllEqual(subwords, expected_subwords)

  def testWordPieceOpWithIdReturned(self):
    """Let the table determine how to do a lookup on unknown tokens."""
    tokens = ragged_factory_ops.constant(
        [[b"don't", b"tread", b"cantfindme", b"treadcantfindme"]])
    tokenizer = FastWordpieceTokenizer(
        vocab=_ENGLISH_VOCAB, token_out_type=dtypes.int64,
        no_pretokenization=True)
    subwords, _, _ = tokenizer.tokenize_with_offsets(tokens)

    self.assertAllEqual(subwords, [[[0, 1, 2], [3], [21], [21]]])
    self.assertEqual(subwords.dtype, dtypes.int64)

  def testWordPieceOpWithInt32IdReturned(self):
    """Let the table determine how to do a lookup on unknown tokens."""
    tokens = ragged_factory_ops.constant(
        [[b"don't", b"tread", b"cantfindme", b"treadcantfindme"]])
    tokenizer = FastWordpieceTokenizer(
        vocab=_ENGLISH_VOCAB, token_out_type=dtypes.int32,
        no_pretokenization=True)
    subwords, _, _ = tokenizer.tokenize_with_offsets(tokens)

    self.assertAllEqual(subwords, [[[0, 1, 2], [3], [21], [21]]])
    self.assertEqual(subwords.dtype, dtypes.int32)

  # pyformat: disable
  @parameterized.parameters([
      dict(
          tokens=[[b"don't", b"treadness", b"whatchamacallit?"]],
          expected_subwords=[[[b"don", b"##'", b"##t"], [b"tread", b"##ness"],
                              [b"what", b"##cha", b"##ma", b"##call",
                               b"##it?"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=[[[b"don't"], [b"treadness"], [b"whatchamacallit?"]]],
          expected_subwords=[
              [[[b"don", b"##'", b"##t"]], [[b"tread", b"##ness"]],
               [[b"what", b"##cha", b"##ma", b"##call", b"##it?"]]]
          ],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=[[[b"don't", _Utf8(u"貿")],
                   [b"treadness", _Utf8(u"大")],
                   [b"whatchamacallit?", _Utf8(u"福")]]],
          expected_subwords=[[[[b"don", b"##'", b"##t"], [_Utf8(u"貿")]],
                              [[b"tread", b"##ness"], [_Utf8(u"大")]],
                              [[
                                  b"what", b"##cha", b"##ma", b"##call",
                                  b"##it?"
                              ], [_Utf8(u"福")]]]],
          vocab=_MIXED_LANG_VOCAB,
      ),
      # # Vector input
      dict(
          tokens=[_Utf8(u"купиха")],
          expected_subwords=[[
              _Utf8(u"к"),
              _Utf8(u"##уп"),
              _Utf8(u"##иха"),
          ]],
          vocab=_RUSSIAN_VOCAB,
      ),
      # # Scalar input
      dict(
          tokens=_Utf8(u"купиха"),
          expected_subwords=[
              _Utf8(u"к"),
              _Utf8(u"##уп"),
              _Utf8(u"##иха"),
          ],
          vocab=_RUSSIAN_VOCAB,
      ),
      # 3D input with 1 ragged dimension.
      dict(
          tokens=[[b"don't", b"treadness", b"whatchamacallit?"]],
          expected_subwords=[[[b"don", b"##'", b"##t"], [b"tread", b"##ness"],
                              [b"what", b"##cha", b"##ma", b"##call",
                               b"##it?"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=ragged_factory_ops.constant_value(
              [[[b"don't"], [b"treadness"], [b"whatchamacallit?"]]],
              ragged_rank=1),
          expected_subwords=[
              [[[b"don", b"##'", b"##t"]], [[b"tread", b"##ness"]],
               [[b"what", b"##cha", b"##ma", b"##call", b"##it?"]]]
          ],
          vocab=_ENGLISH_VOCAB,
      ),
      # Specifying max_chars_per_token.
      dict(
          tokens=[[b"don't", b"treadness"]],
          max_chars_per_token=5,
          expected_subwords=[[[b"don", b"##'", b"##t"], [b"tread", b"##ness"]]],
          vocab=_ENGLISH_VOCAB + [b"trea", b"##d"],
      ),
  ])
  # pyformat: enable
  def testTensors(self,
                  tokens,
                  expected_subwords,
                  vocab,
                  max_chars_per_token=None,
                  expected_start=None,
                  expected_end=None,
                  token_out_type=dtypes.string):
    tokenizer = FastWordpieceTokenizer(
        vocab=vocab,
        token_out_type=token_out_type,
        no_pretokenization=True
    )
    subwords = tokenizer.tokenize(tokens)
    self.assertAllEqual(subwords, expected_subwords)


# The following WordPiece setup is used in `FastWordpieceOpAdditionalTest` and
# `EndToEndFastWordpieceOpTest`.
_TEST_VOCAB = [
    "a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f", "##ghz",
    "<unk>", ","
]
_TEST_MAX_BYTES_PER_WORD = 100
_TEST_SUFFIX_INDICATOR = "##"
_TEST_UNKNOWN_TOKEN = "<unk>"

# The same WordPiece model but precompiled in buffer.
_TEST_MODEL_BUFFER_PATH = "tensorflow_text/python/ops/test_data/fast_wordpiece_tokenizer_model.fb"


def _LoadTestModelBuffer():
  return gfile.GFile(_TEST_MODEL_BUFFER_PATH, "rb").read()


@parameterized.parameters([
    # Test 0: Basic.
    dict(
        text_inputs=[u"", u"abcdefghz", u"abc", u"abcX"],
        expected_outputs=[[], [1, 3, 6, 7], [1], [8]],
    ),
    # Test 1: 2D input.
    dict(
        text_inputs=[[u"", u"abcdefghz", u"abc", u"abcX"]],
        expected_outputs=[[[], [1, 3, 6, 7], [1], [8]]],
    ),
    # Test 2: RaggedTensor input.
    dict(
        text_inputs=ragged_factory_ops.constant_value(
            [[u"", u"abcdefghz", u"abc"], [u"abcX"]]),
        expected_outputs=[[[], [1, 3, 6, 7], [1]], [[8]]],
    ),
])
class FastWordpieceOpAdditionalTest(test_base.DatasetTestBase,
                                    test_util.TensorFlowTestCase,
                                    parameterized.TestCase):
  """Some new tests, including tests on `tf.function`."""

  def testTokenizerBuiltFromConfig(self, text_inputs, expected_outputs):
    tokenizer = FastWordpieceTokenizer(
        vocab=_TEST_VOCAB,
        max_bytes_per_word=_TEST_MAX_BYTES_PER_WORD,
        suffix_indicator=_TEST_SUFFIX_INDICATOR,
        unknown_token=_TEST_UNKNOWN_TOKEN,
        no_pretokenization=True)

    self.assertAllEqual(tokenizer.tokenize(text_inputs), expected_outputs)

  def testTokenizerBuiltFromModel(self, text_inputs, expected_outputs):
    model_buffer = _LoadTestModelBuffer()
    tokenizer = FastWordpieceTokenizer(model_buffer=model_buffer)

    self.assertAllEqual(tokenizer.tokenize(text_inputs), expected_outputs)

  def testTokenizerBuiltFromModelInTensor(self, text_inputs, expected_outputs):
    model_buffer = _LoadTestModelBuffer()
    model_buffer = tf.constant(list(model_buffer), dtype=tf.uint8)
    tokenizer = FastWordpieceTokenizer(model_buffer=model_buffer)

    self.assertAllEqual(tokenizer.tokenize(text_inputs), expected_outputs)

  def testTokenizerBuiltInsideTfFunctionFromConfig(self, text_inputs,
                                                   expected_outputs):

    @def_function.function
    def Preprocess(text_input):
      tokenizer = FastWordpieceTokenizer(
          vocab=_TEST_VOCAB,
          max_bytes_per_word=_TEST_MAX_BYTES_PER_WORD,
          suffix_indicator=_TEST_SUFFIX_INDICATOR,
          unknown_token=_TEST_UNKNOWN_TOKEN,
          no_pretokenization=True)
      return tokenizer.tokenize(text_input)

    # Basic tests.
    self.assertAllEqual(Preprocess(text_inputs), expected_outputs)

    # Test with tf.data.DataSets.
    dataset = dataset_ops.Dataset.from_tensor_slices(text_inputs)
    self.assertDatasetProduces(dataset.map(Preprocess), expected_outputs)

  def testTokenizerBuiltInsideTfFunctionFromModel(self, text_inputs,
                                                  expected_outputs):

    @def_function.function
    def Preprocess(text_input):
      model_buffer = _LoadTestModelBuffer()
      tokenizer = FastWordpieceTokenizer(model_buffer=model_buffer)
      return tokenizer.tokenize(text_input)

    # Basic tests.
    self.assertAllEqual(Preprocess(text_inputs), expected_outputs)

    # Test with tf.data.DataSets.
    dataset = dataset_ops.Dataset.from_tensor_slices(text_inputs)
    self.assertDatasetProduces(dataset.map(Preprocess), expected_outputs)

  def testTokenizerBuiltOutsideTfFunctionFromConfig(self, text_inputs,
                                                    expected_outputs):
    tokenizer = FastWordpieceTokenizer(
        vocab=_TEST_VOCAB,
        max_bytes_per_word=_TEST_MAX_BYTES_PER_WORD,
        suffix_indicator=_TEST_SUFFIX_INDICATOR,
        unknown_token=_TEST_UNKNOWN_TOKEN,
        no_pretokenization=True)

    @def_function.function
    def Preprocess(text_input):
      return tokenizer.tokenize(text_input)

    # Basic tests.
    self.assertAllEqual(Preprocess(text_inputs), expected_outputs)

    # Test with tf.data.DataSets.
    dataset = dataset_ops.Dataset.from_tensor_slices(text_inputs)
    self.assertDatasetProduces(dataset.map(Preprocess), expected_outputs)

  def testTokenizerBuiltOutsideTfFunctionFromModel(self, text_inputs,
                                                   expected_outputs):
    model_buffer = _LoadTestModelBuffer()
    tokenizer = FastWordpieceTokenizer(model_buffer=model_buffer)

    @def_function.function
    def Preprocess(text_input):
      return tokenizer.tokenize(text_input)

    # Basic tests.
    self.assertAllEqual(Preprocess(text_inputs), expected_outputs)

    # Test with tf.data.DataSets.
    dataset = dataset_ops.Dataset.from_tensor_slices(text_inputs)
    self.assertDatasetProduces(dataset.map(Preprocess), expected_outputs)


@parameterized.parameters([
    # Test 0: Basic.
    dict(
        text_inputs=[u"abcdefghz abc, abcX"],
        expected_outputs=[[1, 3, 6, 7, 1, 9, 8]],
    ),
    # Test 1: 2D input.
    dict(
        text_inputs=[[u"abcdefghz abc abcX"]],
        expected_outputs=[[[1, 3, 6, 7, 1, 8]]],
    ),
    # Test 2: RaggedTensor input.
    dict(
        text_inputs=ragged_factory_ops.constant_value(
            [[u"", u"abcdefghz", u"abc"], [u"abcX"]]),
        expected_outputs=[[[], [1, 3, 6, 7], [1]], [[8]]],
    ),
])
class EndToEndFastWordpieceOpTest(test_base.DatasetTestBase,
                                  test_util.TensorFlowTestCase,
                                  parameterized.TestCase):
  """Test on end-to-end fast WordPiece when input is sentence."""

  def testTokenizerBuiltFromConfig(self, text_inputs, expected_outputs):
    tokenizer = FastWordpieceTokenizer(
        vocab=_TEST_VOCAB,
        max_bytes_per_word=_TEST_MAX_BYTES_PER_WORD,
        suffix_indicator=_TEST_SUFFIX_INDICATOR,
        unknown_token=_TEST_UNKNOWN_TOKEN)
    self.assertAllEqual(tokenizer.tokenize(text_inputs), expected_outputs)


@parameterized.parameters([
    # Test 0: Basic.
    dict(
        id_inputs=[[1, 3, 6, 7, 1, 9, 8]],  # Ids of [[u"abcdefghz abc, abcX"]].
        expected_outputs=[b"abcdefghz abc , <unk>"],
    ),
    # Test 1: 1D input.
    dict(
        id_inputs=[1, 3, 6, 7, 1, 9, 8],  # Ids of [u"abcdefghz abc, abcX"].
        expected_outputs=b"abcdefghz abc , <unk>",
    ),
    # Test 2: RaggedTensor input.
    dict(
        id_inputs=ragged_factory_ops.constant_value([[[], [1, 3, 6, 7, 1, 9, 8],
                                                      [1]], [[8]]]),
        expected_outputs=[[b"", b"abcdefghz abc , <unk>", b"abc"], [b"<unk>"]],
    ),
])
class FastWordpieceDetokenizeOpTest(test_base.DatasetTestBase,
                                    test_util.TensorFlowTestCase,
                                    parameterized.TestCase):
  """Test on end-to-end fast WordPiece when input is sentence."""

  def testTokenizerBuiltFromConfig(self, id_inputs, expected_outputs):
    tokenizer = FastWordpieceTokenizer(
        vocab=_TEST_VOCAB,
        max_bytes_per_word=_TEST_MAX_BYTES_PER_WORD,
        suffix_indicator=_TEST_SUFFIX_INDICATOR,
        unknown_token=_TEST_UNKNOWN_TOKEN,
        support_detokenization=True)
    results = tokenizer.detokenize(id_inputs)
    self.assertAllEqual(results, expected_outputs)


@parameterized.parameters([
    # Test 0: Single-word case.
    dict(
        no_pretokenization=True,
        text_inputs=[["", "abcdefghz", "abc", "abcX"]],
    ),
    # Test 1: End-to-end case.
    dict(
        no_pretokenization=False,
        text_inputs=[["", "abcdefghz, a", "abcdefghz abc abcX"]],
    ),
])
class FastWordpieceInKerasModelTest(test_util.TensorFlowTestCase,
                                    parameterized.TestCase):
  """Tests fast WordPiece when used in a Keras model."""

  def testTfLiteWordpieceTokenizer(
      self, no_pretokenization, text_inputs):
    """Checks TFLite conversion and inference."""

    class TokenizerModel(tf.keras.Model):

      def __init__(self,
                   vocab,
                   max_bytes_per_word=100,
                   suffix_indicator="##",
                   unknown_token="<unk>",
                   no_pretokenization=True,
                   support_detokenization=False,
                   **kwargs):
        super().__init__(**kwargs)
        self.wp = FastWordpieceTokenizer(
            vocab=vocab,
            max_bytes_per_word=max_bytes_per_word,
            suffix_indicator=suffix_indicator,
            unknown_token=unknown_token,
            no_pretokenization=no_pretokenization,
            support_detokenization=support_detokenization)

      def call(self, input_tensor, **kwargs):
        return self.wp.tokenize(input_tensor).flat_values

      @tf.function(
          input_signature=[tf.TensorSpec(shape=[None], dtype=dtypes.int64)])
      def detokenize(self, input_tensor):
        return self.wp.detokenize(input_tensor)

    # Test input data.
    input_data = np.array(text_inputs)

    # Define a Keras model.
    model = TokenizerModel(
        _TEST_VOCAB,
        _TEST_MAX_BYTES_PER_WORD,
        _TEST_SUFFIX_INDICATOR,
        _TEST_UNKNOWN_TOKEN,
        no_pretokenization,
        support_detokenization=True)
    # Test tokenization.
    # Do TF.Text inference.
    tf_result = model(input_data)

    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Do TFLite inference.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    interp.set_tensor(input_details[0]["index"], input_data)
    interp.invoke()
    output_details = interp.get_output_details()
    tflite_result = interp.get_tensor(output_details[0]["index"])

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)

    # Test detokenization.
    # Do TF.Text detokenization.
    tf_detokenization_result = model.detokenize(tf_result)
    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [model.detokenize.get_concrete_function()], model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Do TFLite detokenization.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    interp.allocate_tensors()
    detokenize = interp.get_signature_runner("serving_default")
    tflite_detokenization_result = detokenize(
        input_tensor=tf_result)["output_0"]

    # Assert the results are identical.
    self.assertAllEqual(tf_detokenization_result, tflite_detokenization_result)


if __name__ == "__main__":
  test.main()
