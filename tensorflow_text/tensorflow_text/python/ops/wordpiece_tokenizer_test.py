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
"""Tests for wordpiece_tokenized op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensorflow.python.compat import compat

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops.wordpiece_tokenizer import WordpieceTokenizer


def _Utf8(char):
  return char.encode("utf-8")


def _CreateTable(vocab, num_oov=1):
  size = array_ops.size(vocab, out_type=dtypes.int64)
  init = lookup_ops.KeyValueTensorInitializer(
      vocab,
      math_ops.range(size, dtype=dtypes.int64),
      key_dtype=dtypes.string,
      value_dtype=dtypes.int64)
  return lookup_ops.StaticVocabularyTableV1(
      init, num_oov, lookup_key_dtype=dtypes.string)


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
]

_RUSSIAN_VOCAB = [
    _Utf8(u"к"),
    _Utf8(u"##уп"),
    _Utf8(u"##иха"),
]

_DEATH_VOCAB = [
    _Utf8(u"क"),
    _Utf8(u"##र"),
    _Utf8(u"##े"),
    _Utf8(u"##ं"),
    b"##*",
    _Utf8(u"##👇"),
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


class WordpieceOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  _FORWARD_COMPATIBILITY_HORIZONS = [
      (2019, 7, 1),
      (2019, 10, 10),
      (2525, 1, 1),  # future behavior
  ]

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
      # Basic case w/o unknown token
      dict(
          tokens=[[b"don't", b"tread", b"cantfindme", b"treadcantfindme"]],
          expected_subwords=[[[b"don", b"##'", b"##t"], [b"tread"],
                              [b"cantfindme"], [b"treadcantfindme"]]],
          unknown_token=None,
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
      # Test token whose size is > max_bytes_per_word
      dict(
          tokens=[[b"don't", b"treadness"]],
          expected_subwords=[[[b"don", b"##'", b"##t"], [b"[UNK]"]]],
          vocab=_ENGLISH_VOCAB,
          max_bytes_per_word=5,
          # Explicitly specify the offsets here because the current way of
          # testing offsets would require '[UNK]' to be part of tokens.
          expected_start=[[[0, 3, 4], [0]]],
          expected_end=[[[3, 4, 5], [5]]],
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
      # Test splitting out unknown characters.
      dict(
          tokens=[[b"nap", b"hello!me"]],
          expected_subwords=[
              [[b"na", b"##[UNK]"], [b"hel", b"##lo", b"##[UNK]", b"##me"]]],
          unknown_token="[UNK]",
          vocab=_ENGLISH_VOCAB,
          split_unknown_characters=True,
      ),
      # Test splitting out unknown characters, with unknown_token set to None.
      dict(
          tokens=[[b"nap", b"hello!me"]],
          expected_subwords=[
              [[b"na", b"##p"], [b"hel", b"##lo", b"##!", b"##me"]]],
          unknown_token=None,
          vocab=_ENGLISH_VOCAB,
          split_unknown_characters=True,
      ),
  ])
  def testWordPieceOpAndVerifyOffsets(self,
                                      tokens,
                                      expected_subwords,
                                      vocab,
                                      expected_start=None,
                                      expected_end=None,
                                      use_unknown_token=True,
                                      unknown_token="[UNK]",
                                      token_out_type=dtypes.string,
                                      max_bytes_per_word=100,
                                      split_unknown_characters=False):
    for horizon in self._FORWARD_COMPATIBILITY_HORIZONS:
      with compat.forward_compatibility_horizon(*horizon):
        tokens_t = ragged_factory_ops.constant(tokens)
        vocab_table = _CreateTable(vocab)
        self.evaluate(vocab_table.initializer)
        tokenizer = WordpieceTokenizer(
            vocab_table,
            unknown_token=unknown_token,
            token_out_type=token_out_type,
            max_bytes_per_word=max_bytes_per_word,
            split_unknown_characters=split_unknown_characters,
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
                                            use_unknown_token=True,
                                            token_out_type=dtypes.string):
    for row_splits_dtype in (dtypes.int32, dtypes.int64):
      ragged_tokens = ragged_factory_ops.constant(
          tokens, row_splits_dtype=row_splits_dtype)
      vocab_table = _CreateTable(vocab)
      self.evaluate(vocab_table.initializer)
      tokenizer = WordpieceTokenizer(vocab_table, token_out_type=token_out_type)
      subwords = tokenizer.tokenize(ragged_tokens)
      self.assertAllEqual(subwords, expected_subwords)

  def testWordPieceOpWithIdReturned(self):
    """Let the table determine how to do a lookup on unknown tokens."""
    tokens = ragged_factory_ops.constant(
        [[b"don't", b"tread", b"cantfindme", b"treadcantfindme"]])
    vocab_table = _CreateTable(
        _ENGLISH_VOCAB,
        100  # OOV values
    )
    self.evaluate(vocab_table.initializer)
    tokenizer = WordpieceTokenizer(
        vocab_table, unknown_token=None, token_out_type=dtypes.int64)
    subwords, _, _ = tokenizer.tokenize_with_offsets(tokens)

    self.assertAllEqual(subwords, [[[0, 1, 2], [3], [96], [46]]])
    self.assertEqual(subwords.dtype, dtypes.int64)

  def testWordPieceOpWithInt32IdReturned(self):
    """Let the table determine how to do a lookup on unknown tokens."""
    tokens = ragged_factory_ops.constant(
        [[b"don't", b"tread", b"cantfindme", b"treadcantfindme"]])
    vocab_table = _CreateTable(
        _ENGLISH_VOCAB,
        100  # OOV values
    )
    self.evaluate(vocab_table.initializer)
    tokenizer = WordpieceTokenizer(
        vocab_table, unknown_token=None, token_out_type=dtypes.int32)
    subwords, _, _ = tokenizer.tokenize_with_offsets(tokens)

    self.assertAllEqual(subwords, [[[0, 1, 2], [3], [96], [46]]])
    self.assertEqual(subwords.dtype, dtypes.int32)

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
      # Vector input
      dict(
          tokens=[_Utf8(u"купиха")],
          expected_subwords=[[
              _Utf8(u"к"),
              _Utf8(u"##уп"),
              _Utf8(u"##иха"),
          ]],
          vocab=_RUSSIAN_VOCAB,
      ),
      # Scalar input
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
          expected_subwords=[
              [[b"don", b"##'", b"##t"], [b"tread", b"##ness"]]],
          vocab=_ENGLISH_VOCAB + [b"trea", b"##d"],
      ),
      # Specifying max_chars_per_token to 4, so that "tread" is not found, and
      # is split into "trea", "##d".
      dict(
          tokens=[[b"don't", b"treadness"]],
          max_chars_per_token=4,
          expected_subwords=[
              [[b"don", b"##'", b"##t"], [b"trea", b"##d", b"##ness"]]],
          vocab=_ENGLISH_VOCAB + [b"trea", b"##d"],
      ),
      # Specifying max_chars_per_token where characters are multiple bytes.
      dict(
          tokens=[[_Utf8(u"大"), _Utf8(u"易")]],
          max_chars_per_token=1,
          expected_subwords=[[[_Utf8(u"大")], [_Utf8(u"易")]]],
          vocab=_CHINESE_VOCAB,
      ),
  ])
  def testTensors(self,
                  tokens,
                  expected_subwords,
                  vocab,
                  max_chars_per_token=None,
                  expected_start=None,
                  expected_end=None,
                  use_unknown_token=True,
                  token_out_type=dtypes.string):
    vocab_table = _CreateTable(vocab)
    self.evaluate(vocab_table.initializer)
    tokenizer = WordpieceTokenizer(
        vocab_table, token_out_type=token_out_type,
        max_chars_per_token=max_chars_per_token,
    )
    subwords = tokenizer.tokenize(tokens)
    self.assertAllEqual(subwords, expected_subwords)

  @test_util.run_in_graph_and_eager_modes
  def testDetokenizeIsReversable(self):

    table = _CreateTable(_MIXED_LANG_VOCAB + [b""], 2)
    self.evaluate(table.initializer)

    tokenizer = WordpieceTokenizer(table)

    word_lists = [
        [b"hello", b"there", b"my", b"name", b"is", b"treadness"],
        [b"whatchamacallit?", b"you", b"said"],
        [_Utf8(u"大"), _Utf8(u"易")],
    ]
    words = ragged_factory_ops.constant(word_lists)

    subwords_ids = tokenizer.tokenize(words)

    # detokeinze input shape is (batch, ragged-words, ragged-wordpieces)
    words_output = tokenizer.detokenize(subwords_ids)
    words_output = array_ops.squeeze(words_output, axis=-1)

    self.assertAllEqual(words_output, words)

    # detokeinze input shape is (batch, ragged-wordpieces)
    subwords_id_seqs = subwords_ids.merge_dims(-2, -1)
    words_output = tokenizer.detokenize(subwords_id_seqs)
    self.assertAllEqual(words_output, words)

    # detokeinze input shape is a dense (batch, padded-wordpieces)
    words_output = tokenizer.detokenize(
        subwords_ids.merge_dims(-2, -1)
        # len(_MIXED_LANG_VOCAB) is ""
        .to_tensor(default_value=len(_MIXED_LANG_VOCAB)))

    self.assertAllEqual(words_output, words)

  @test_util.run_in_graph_and_eager_modes
  def testDetokenizeFailsForSparseVocab(self):
    vocab = ["a", "##b", "##c"]
    ids = [0, 10, 20]
    init = lookup_ops.KeyValueTensorInitializer(
        vocab, ids, key_dtype=dtypes.string, value_dtype=dtypes.int64)
    table = lookup_ops.StaticVocabularyTableV1(
        init, num_oov_buckets=1, lookup_key_dtype=dtypes.string)
    self.evaluate(table.initializer)

    tokenizer = WordpieceTokenizer(table)
    words = ragged_factory_ops.constant([["abb", "abc"], ["abcbc"]])
    subwords_ids = tokenizer.tokenize(words)

    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "detokenize.*?dense on the interval"):
      result = tokenizer.detokenize(subwords_ids)
      self.evaluate(result)

  def testVocabSize(self):
    vocab_table = _CreateTable(
        _ENGLISH_VOCAB,
        100  # OOV values
    )
    self.evaluate(vocab_table.initializer)
    tokenizer = WordpieceTokenizer(
        vocab_table, unknown_token=None)
    self.assertEqual(self.evaluate(tokenizer.vocab_size()), 121)

if __name__ == "__main__":
  test.main()
