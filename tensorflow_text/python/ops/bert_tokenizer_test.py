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
r"""Tests for BertTokenizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow_text.python.ops import bert_tokenizer


def _utf8(x):
  return x.encode('utf-8')


# TODO(thuang513): It appears there isn't a Ragged version of substr; consider
#               checking this into core TF.
def _ragged_substr(text_input, begin, end):
  text_input_flat = None
  if ragged_tensor.is_ragged(text_input):
    text_input_flat = text_input.flat_values
  else:
    text_input_flat = text_input

  def _ragged_tile(x):
    input_text, indices = x
    multiple = math_ops.reduce_sum(indices.row_lengths())
    return array_ops.tile([input_text], [multiple])

  broadcasted_text = ragged_map_ops.map_fn(
      _ragged_tile,
      (text_input_flat, begin),
      dtype=ragged_tensor.RaggedTensorType(dtype=dtypes.string, ragged_rank=1),
      infer_shape=False,
  )
  size = math_ops.sub(
      array_ops.squeeze(end.flat_values), array_ops.squeeze(begin.flat_values))
  new_tokens = string_ops.substr_v2(broadcasted_text,
                                    array_ops.squeeze(begin.flat_values), size)
  return begin.with_flat_values(new_tokens.flat_values)


_VOCAB = [
    b'[unused1]',
    b'[unused23]',
    b"'",
    b'##%',
    b'##af',
    b'##book',
    b'##c',
    b'##fr',
    b'##hey',
    b'##is',
    b'##o',
    b'##ost',
    b'##s',
    b'##tri',
    b'##y',
    b'$',
    b'%',
    b'&',
    b'(',
    b')',
    b'*',
    b'-',
    b'.',
    b'20',
    b':',
    b'?',
    b'[CLS]',
    b'[SEP]',
    _utf8(u'國'),
    _utf8(u'暐'),
    _utf8(u'瀚'),
    _utf8(u'韓'),
    _utf8(u'食'),
    _utf8(u'黃'),
    _utf8(u'🤔'),
    _utf8(u'🤣'),
    b'^',
    b'a',
    b'ago',
    b'among',
    b'an',
    b'and',
    b'are',
    b'aren',
    b'awesome',
    b'between',
    b'candy',
    b'china',
    b'companies',
    b'company',
    b'crushed',
    b'dug',
    b'earnings',
    b'engaged',
    b'even',
    b'few',
    b'forecast',
    b'getting',
    b'had',
    b'han',
    b'has',
    b'hers',
    b'high',
    b'hit',
    b'hs',
    b'hurting',
    b'in',
    b'indie',
    b'is',
    b'isn',
    b'ka',
    b'ku',
    b'major',
    b'maker',
    b'moth',
    b'nearly',
    b'new',
    b'now',
    b'president',
    b'record',
    b'regulators',
    b'reported',
    b'rift',
    b'rust',
    b'sales',
    b'shares',
    b'slightly',
    b'sprint',
    b'states',
    b'stock',
    b't',
    b'taste',
    b'tension',
    b'that',
    b'the',
    b'this',
    b'today',
    b'told',
    b'topped',
    b'trade',
    b'trump',
    b'united',
    b'up',
    b'weeks',
    b'what',
    b'why',
    b'with',
    b'year',
    b'yo',
    b'yu',
    _utf8(u'\u7231'),
    _utf8(u'\u4e0a'),
    _utf8(u'\u4e00'),
    _utf8(u'\u4e2a'),
    _utf8(u'\u4e0d'),
    _utf8(u'\u56de'),
    _utf8(u'\u5bb6'),
    _utf8(u'\u7684'),
    _utf8(u'\u4eba'),
]


def _create_table(vocab, num_oov=1):
  init = lookup_ops.KeyValueTensorInitializer(
      vocab,
      math_ops.range(
          array_ops.size(vocab, out_type=dtypes.int64), dtype=dtypes.int64),
      key_dtype=dtypes.string,
      value_dtype=dtypes.int64)
  return lookup_ops.StaticVocabularyTableV1(
      init, num_oov, lookup_key_dtype=dtypes.string)


class BertTokenizerTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def test_bert_tokenizer_outputs(self):
    text_inputs = constant_op.constant([_utf8('Test')])
    vocab = _VOCAB
    table = _create_table(vocab, 2)
    self.evaluate(table.initializer)
    tokenizer = bert_tokenizer.BertTokenizer(
        table,
        token_out_type=dtypes.int32)
    results = tokenizer.tokenize(text_inputs)
    self.assertAllEqual(results.dtype, dtypes.int32)

  @parameterized.parameters([
      dict(
          text_inputs=[
              _utf8(u'taste the rustisc indiefrost'),
              _utf8(u'Han Kuo-yu (韓國食)🤔'),
              _utf8(u'Añade la información del formulario y tus preguntas'),
          ],
          expected_tokens=[[b'taste', b'the', b'rustisc', b'indiefrost'],
                           [
                               b'Han', b'Kuo', b'-', b'yu', b'(',
                               b'\xe9\x9f\x93', b'\xe5\x9c\x8b',
                               b'\xe9\xa3\x9f', b')', b'\xf0\x9f\xa4\x94'
                           ],
                           [
                               b'A\xc3\xb1ade', b'la', b'informaci\xc3\xb3n',
                               b'del', b'formulario', b'y', b'tus', b'preguntas'
                           ]],
      ),
      dict(
          text_inputs=[
              _utf8(u'UNwant\u00E9d,running'),
              _utf8(u'Añade la información del formulario y tus preguntas'),
          ],
          expected_tokens=[[b'unwanted', b',', b'running'],
                           [
                               b'anade', b'la', b'informacion', b'del',
                               b'formulario', b'y', b'tus', b'preguntas'
                           ]],
          lower_case=True,
          # `lower_case` doesn't let you override the `normalization_form`
          normalization_form=None,
      ),
      dict(
          text_inputs=[
              _utf8(u'Añade la información del formulario y tus preguntas')
          ],
          expected_tokens=[[
              b'An\xcc\x83ade', b'la', b'informacio\xcc\x81n', b'del',
              b'formulario', b'y', b'tus', b'preguntas'
          ]],
          normalization_form='NFD',
      ),
      # Test CJK are tokenized by unicode characters
      dict(
          text_inputs=[
              _utf8(u'香港では４日'),
              _utf8(u'영어독해 자만심 왜 문제일까'),
              _utf8(u'據港媒《東網》報導')
          ],
          expected_tokens=[
              [_utf8(u'香'),
               _utf8(u'港'),
               _utf8(u'では４'),
               _utf8(u'日')],
              [
                  _utf8(u'영어독해'),
                  _utf8(u'자만심'),
                  _utf8(u'왜'),
                  _utf8(u'문제일까'),
              ],
              [
                  _utf8(u'據'),
                  _utf8(u'港'),
                  _utf8(u'媒'),
                  _utf8(u'《'),
                  _utf8(u'東'),
                  _utf8(u'網'),
                  _utf8(u'》'),
                  _utf8(u'報'),
                  _utf8(u'導')
              ],
          ],
          normalization_form=None,
      ),
      # Test Katakana followed by Hiragana.
      dict(
          text_inputs=[_utf8(u'のテキストとして')],
          expected_tokens=[
              [_utf8(u'のテキストとして')],
          ],
          normalization_form=None,
      ),
  ])
  @test_util.run_in_graph_and_eager_modes
  def test_basic_tokenize(self,
                          text_inputs,
                          expected_tokens,
                          lower_case=False,
                          normalization_form='NFC'):
    text_inputs = ragged_factory_ops.constant(text_inputs)
    tokenizer = bert_tokenizer.BasicTokenizer(
        lower_case=lower_case, normalization_form=normalization_form)
    tokens = tokenizer.tokenize(text_inputs)
    self.assertAllEqual(tokens, expected_tokens)

  @parameterized.parameters([
      dict(
          text_inputs=[
              b'taste the rustisc indiefrost',
              _utf8(u'Han Kuo-yu (韓國食)🤔'),
              _utf8(u'dugtrio had an awesome 🤣 dugbook'),
              b'yo^what$is*up?',
              b'mothaf*&%ka',
          ],
          expected=[[[b'taste'], [b'the'], [b'rust', b'##is', b'##c'],
                     [b'indie', b'##fr', b'##ost']],
                    [[b'han'], [b'ku', b'##o'], [b'-'], [b'yu'], [b'('],
                     [_utf8(u'韓')], [_utf8(u'國')], [_utf8(u'食')], [b')'],
                     [_utf8(u'🤔')]],
                    [[b'dug', b'##tri', b'##o'], [b'had'], [b'an'],
                     [b'awesome'], [_utf8(u'🤣')], [b'dug', b'##book']],
                    [[b'yo'], [b'^'], [b'what'], [b'$'], [b'is'], [b'*'],
                     [b'up'], [b'?']],
                    [[b'moth', b'##af'], [b'*'], [b'&'], [b'%'], [b'ka']]],
          expected_extracted=[[[b'taste'], [b'the'], [b'rust', b'is', b'c'],
                               [b'indie', b'fr', b'ost']],
                              [[b'Han'], [b'Ku', b'o'], [b'-'], [b'yu'], [b'('],
                               [_utf8(u'韓')], [_utf8(u'國')], [_utf8(u'食')],
                               [b')'], [_utf8(u'🤔')]],
                              [[b'dug', b'tri', b'o'], [b'had'], [b'an'],
                               [b'awesome'], [_utf8(u'🤣')], [b'dug', b'book']],
                              [[b'yo'], [b'^'], [b'what'], [b'$'], [b'is'],
                               [b'*'], [b'up'], [b'?']],
                              [[b'moth', b'af'], [b'*'], [b'&'], [b'%'],
                               [b'ka']]],
          lower_case=True,
      ),
      # Test when we are expecting multiple OOV vocab ids and tf.string just
      # maps out [UNK] token.
      dict(
          text_inputs=[
              b'mothaf*&%ka cantfindme whodis',
          ],
          expected=[[[b'moth', b'##af'], [b'*'], [b'&'], [b'%'], [b'ka'],
                     [b'[UNK]'], [b'[UNK]']]],
          expected_extracted=[[[b'moth', b'af'], [b'*'], [b'&'], [b'%'],
                               [b'ka'], [b'cantfindme'], [b'whodis']]],
          lower_case=True,
          num_oov=2,
      ),
      dict(
          text_inputs=[
              b'candy',
          ],
          expected=[[[b'candy']]],
          lower_case=True,
          num_oov=2,
      ),
      dict(
          text_inputs=[
              _utf8(u'爱上一个不回家的人'),
          ],
          expected=[[[_utf8(u'爱')], [_utf8(u'上')], [_utf8(u'一')], [_utf8(u'个')],
                     [_utf8(u'不')], [_utf8(u'回')], [_utf8(u'家')], [_utf8(u'的')],
                     [_utf8(u'人')]]],
          lower_case=True,
          num_oov=2,
      ),
      # Test 'preserve_unused_token' option
      dict(
          text_inputs=[
              b'taste the rustisc indiefrost [unused1]',
              _utf8(u'爱上一个不回家的人[unused23]'),
          ],
          expected=[[[b'taste'], [b'the'], [b'rust', b'##is', b'##c'],
                     [b'indie', b'##fr', b'##ost'], [b'[unused1]']],
                    [[_utf8(u'爱')], [_utf8(u'上')], [_utf8(u'一')], [_utf8(u'个')],
                     [_utf8(u'不')], [_utf8(u'回')], [_utf8(u'家')], [_utf8(u'的')],
                     [_utf8(u'人')], [b'[unused23]']]],
          preserve_unused_token=True,
      ),
  ])
  @test_util.run_in_graph_and_eager_modes
  def test_bert_tokenizer(self,
                          text_inputs,
                          expected,
                          vocab=None,
                          expected_extracted=None,
                          lower_case=True,
                          num_oov=1,
                          preserve_unused_token=False):
    text_inputs = constant_op.constant(text_inputs)
    if not vocab:
      vocab = _VOCAB
    table = _create_table(vocab, num_oov)
    self.evaluate(table.initializer)
    tokenizer = bert_tokenizer.BertTokenizer(
        table,
        token_out_type=dtypes.string,
        lower_case=lower_case,
        preserve_unused_token=preserve_unused_token)
    results = tokenizer.tokenize(text_inputs)
    self.assertAllEqual(results, expected)

    # Verify that the int ids are the same.
    expected_rt = ragged_factory_ops.constant(expected)
    expected_int = table.lookup(expected_rt.flat_values)
    expected_int_rt = ragged_tensor.RaggedTensor.from_nested_row_splits(
        expected_int, expected_rt.nested_row_splits)
    int_tokenizer = bert_tokenizer.BertTokenizer(
        vocab_lookup_table=table,
        token_out_type=dtypes.int64,
        lower_case=lower_case,
        preserve_unused_token=preserve_unused_token)
    results_int = int_tokenizer.tokenize(text_inputs)
    self.assertAllEqual(results_int, expected_int_rt)

    # Verify that the offsets can extract the expected tokens
    _, begin, end = tokenizer.tokenize_with_offsets(text_inputs)

    extracted_wordpieces = _ragged_substr(text_inputs, begin, end)
    if expected_extracted:
      self.assertAllEqual(extracted_wordpieces, expected_extracted)
    else:
      # The extracted won't have any wordpieces with '##' prefix. Strip them
      # out.
      stripped_prefix_flat = string_ops.regex_replace(expected_rt.flat_values,
                                                      '##', '')
      stripped_prefix = expected_rt.with_flat_values(stripped_prefix_flat)
      self.assertAllEqual(extracted_wordpieces, stripped_prefix)

  @parameterized.parameters(
      dict(row_splits_dtype=dtypes.int32),
      dict(row_splits_dtype=dtypes.int64))
  @test_util.run_in_graph_and_eager_modes
  def test_int32_row_splits(self, row_splits_dtype):
    table = _create_table(_VOCAB)
    self.evaluate(table.initializer)
    tokenizer = bert_tokenizer.BertTokenizer(
        table,
        token_out_type=dtypes.string,
        lower_case=True,
        preserve_unused_token=False)

    # Construct an input RaggedTensor [[b'candybook']]
    text_inputs = ragged_tensor.RaggedTensor.from_row_lengths(
        [b'candybook'], [1])
    text_inputs = text_inputs.with_row_splits_dtype(row_splits_dtype)

    # The input had shape [1, 1] so the output will have shape
    # [1, 1, num_word_pieces], where num_word_pieces=2.
    results = tokenizer.tokenize(text_inputs)

    # Check that the row_splits dtypes for the input and output tensors match.
    self.assertEqual(results.row_splits.dtype, row_splits_dtype)

    # Verify that the output values are correct (sanity check).
    self.assertAllEqual(results.flat_values, [b'candy', b'##book'])


if __name__ == '__main__':
  test.main()
