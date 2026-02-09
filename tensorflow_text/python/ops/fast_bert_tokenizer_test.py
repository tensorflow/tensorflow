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
r"""Tests for FastBertTokenizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow_text.python.ops import fast_bert_tokenizer


def _utf8(x):
  return x.encode('utf-8')


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
    b'un',
    b'##want',
    b'##ed',
    b'run',
    b'##ning',
    b',',
    b'[UNK]',
]


class FastBertTokenizerTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):

  def test_fast_bert_tokenizer_outputs(self):
    text_inputs = constant_op.constant([_utf8('Test')])
    vocab = _VOCAB
    tokenizer = fast_bert_tokenizer.FastBertTokenizer(
        vocab=vocab, token_out_type=dtypes.int32)
    results = tokenizer.tokenize(text_inputs)
    self.assertAllEqual(results.dtype, dtypes.int32)

  @parameterized.parameters([
      # Test 0.
      dict(
          text_inputs=[
              _utf8(u'taste the rustisc indiefrost'),
              _utf8(u'Han Kuo-yu (韓國食)🤔'),
              _utf8(u'Añade la información del formulario y tus preguntas'),
          ],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[[
              b'taste', b'the', b'rust', b'##is', b'##c', b'indie', b'##fr',
              b'##ost'
          ],
                           [
                               b'han', b'ku', b'##o', b'-', b'yu', b'(',
                               b'\xe9\x9f\x93', b'\xe5\x9c\x8b',
                               b'\xe9\xa3\x9f', b')', b'\xf0\x9f\xa4\x94'
                           ],
                           [
                               b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]',
                               b'[UNK]', b'[UNK]', b'[UNK]'
                           ]],
          expected_token_ids=[[91, 94, 83, 9, 6, 67, 7, 11],
                              [59, 71, 10, 21, 109, 18, 31, 28, 32, 19, 34],
                              [125, 125, 125, 125, 125, 125, 125, 125]],
          expected_starts=[[0, 6, 10, 14, 16, 18, 23, 25],
                           [0, 4, 6, 7, 8, 11, 12, 15, 18, 21, 22],
                           [0, 7, 10, 23, 27, 38, 40, 44]],
          expected_ends=[[5, 9, 14, 16, 17, 23, 25, 28],
                         [3, 6, 7, 8, 10, 12, 15, 18, 21, 22, 26],
                         [6, 9, 22, 26, 37, 39, 43, 53]],
      ),
      # Test 1. Nested tensor.
      dict(
          text_inputs=[[
              _utf8(u'UNwant\u00E9d,running'),
              _utf8(u'Añade la información del formulario y tus preguntas'),
          ]],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[[[
              b'un', b'##want', b'##ed', b',', b'run', b'##ning'
          ],
                            [
                                b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]',
                                b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]'
                            ]]],
          expected_token_ids=[[[119, 120, 121, 124, 122, 123],
                               [125, 125, 125, 125, 125, 125, 125, 125]]],
          expected_starts=[[[0, 2, 6, 9, 10, 13],
                            [0, 7, 10, 23, 27, 38, 40, 44]]],
          expected_ends=[[[2, 6, 9, 10, 13, 17], [6, 9, 22, 26, 37, 39, 43,
                                                  53]]],
      ),
      # Test 2. Tensor. No lower_case_nfd_strip_accents.
      dict(
          text_inputs=[
              _utf8(u'UNwant\u00E9d,running'),
              _utf8(u'Añade la información del formulario y tus preguntas'),
          ],
          lower_case_nfd_strip_accents=False,
          expected_tokens=[[b'[UNK]', b',', b'run', b'##ning'],
                           [
                               b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]',
                               b'[UNK]', b'[UNK]', b'[UNK]'
                           ]],
          expected_token_ids=[[125, 124, 122, 123],
                              [125, 125, 125, 125, 125, 125, 125, 125]],
          expected_starts=[[0, 9, 10, 13], [0, 7, 10, 23, 27, 38, 40, 44]],
          expected_ends=[[9, 10, 13, 17], [6, 9, 22, 26, 37, 39, 43, 53]],
      ),
      # Test 3.
      dict(
          text_inputs=[
              _utf8(u'Añade la información del formulario y tus preguntas')
          ],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[[
              b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]',
              b'[UNK]', b'[UNK]'
          ]],
          expected_token_ids=[[125, 125, 125, 125, 125, 125, 125, 125]],
          expected_starts=[[0, 7, 10, 23, 27, 38, 40, 44]],
          expected_ends=[[6, 9, 22, 26, 37, 39, 43, 53]],
      ),
      # Test 4. Test CJK are tokenized by unicode characters.
      dict(
          text_inputs=[
              _utf8(u'香港では４日'),
              _utf8(u'영어독해 자만심 왜 문제일까'),
              _utf8(u'據港媒《東網》報導')
          ],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[[b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]'],
                           [b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]'],
                           [
                               b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]',
                               b'[UNK]', b'[UNK]', b'[UNK]', b'[UNK]'
                           ]],
          expected_token_ids=[[125, 125, 125, 125], [125, 125, 125, 125],
                              [125, 125, 125, 125, 125, 125, 125, 125, 125]],
          expected_starts=[[0, 3, 6, 15], [0, 13, 23, 27],
                           [0, 3, 6, 9, 12, 15, 18, 21, 24]],
          expected_ends=[[3, 6, 15, 18], [12, 22, 26, 39],
                         [3, 6, 9, 12, 15, 18, 21, 24, 27]],
      ),
      # Test 5. Test Katakana followed by Hiragana.
      dict(
          text_inputs=[_utf8(u'のテキストとして')],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[[b'[UNK]']],
          expected_token_ids=[[125]],
          expected_starts=[[0]],
          expected_ends=[[24]],
      ),
      # Test 6.
      dict(
          text_inputs=[
              b'taste the rustisc indiefrost',
              _utf8(u'Han Kuo-yu (韓國食)🤔'),
              _utf8(u'dugtrio had an awesome 🤣 dugbook'),
              b'yo^what$is*up?',
              b'mothaf*&%ka',
          ],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[
              [
                  b'taste', b'the', b'rust', b'##is', b'##c', b'indie', b'##fr',
                  b'##ost'
              ],
              [
                  b'han', b'ku', b'##o', b'-', b'yu', b'(', b'\xe9\x9f\x93',
                  b'\xe5\x9c\x8b', b'\xe9\xa3\x9f', b')', b'\xf0\x9f\xa4\x94'
              ],
              [
                  b'dug', b'##tri', b'##o', b'had', b'an', b'awesome',
                  b'\xf0\x9f\xa4\xa3', b'dug', b'##book'
              ], [b'yo', b'^', b'what', b'$', b'is', b'*', b'up', b'?'],
              [b'moth', b'##af', b'*', b'&', b'%', b'ka']
          ],
          expected_token_ids=[[91, 94, 83, 9, 6, 67, 7, 11],
                              [59, 71, 10, 21, 109, 18, 31, 28, 32, 19, 34],
                              [51, 13, 10, 58, 40, 44, 35, 51, 5],
                              [108, 36, 104, 15, 68, 20, 102, 25],
                              [74, 4, 20, 17, 16, 70]],
          expected_starts=[[0, 6, 10, 14, 16, 18, 23, 25],
                           [0, 4, 6, 7, 8, 11, 12, 15, 18, 21, 22],
                           [0, 3, 6, 8, 12, 15, 23, 28, 31],
                           [0, 2, 3, 7, 8, 10, 11, 13], [0, 4, 6, 7, 8, 9]],
          expected_ends=[[5, 9, 14, 16, 17, 23, 25, 28],
                         [3, 6, 7, 8, 10, 12, 15, 18, 21, 22, 26],
                         [3, 6, 7, 11, 14, 22, 27, 31, 35],
                         [2, 3, 7, 8, 10, 11, 13, 14], [4, 6, 7, 8, 9, 11]],
      ),
      # Test 7.
      dict(
          text_inputs=[
              b'mothaf*&%ka cantfindme whodis',
          ],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[[
              b'moth', b'##af', b'*', b'&', b'%', b'ka', b'[UNK]', b'[UNK]'
          ]],
          expected_token_ids=[[74, 4, 20, 17, 16, 70, 125, 125]],
          expected_starts=[[0, 4, 6, 7, 8, 9, 12, 23]],
          expected_ends=[[4, 6, 7, 8, 9, 11, 22, 29]],
      ),
      # Test 8.
      dict(
          text_inputs=[
              b'candy',
          ],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[[b'candy']],
          expected_token_ids=[[46]],
          expected_starts=[[0]],
          expected_ends=[[5]],
      ),
      # Test 9.
      dict(
          text_inputs=[
              _utf8(u'爱上一个不回家的人'),
          ],
          lower_case_nfd_strip_accents=True,
          expected_tokens=[[
              b'\xe7\x88\xb1', b'\xe4\xb8\x8a', b'\xe4\xb8\x80',
              b'\xe4\xb8\xaa', b'\xe4\xb8\x8d', b'\xe5\x9b\x9e',
              b'\xe5\xae\xb6', b'\xe7\x9a\x84', b'\xe4\xba\xba'
          ]],
          expected_token_ids=[[110, 111, 112, 113, 114, 115, 116, 117, 118]],
          expected_starts=[[0, 3, 6, 9, 12, 15, 18, 21, 24]],
          expected_ends=[[3, 6, 9, 12, 15, 18, 21, 24, 27]],
      ),
  ])
  @test_util.run_in_graph_and_eager_modes
  def test_fast_bert_tokenizer(self,
                               text_inputs,
                               lower_case_nfd_strip_accents,
                               expected_tokens,
                               expected_token_ids,
                               expected_starts,
                               expected_ends,
                               vocab=None):
    text_inputs = constant_op.constant(text_inputs)
    if not vocab:
      vocab = _VOCAB
    # Verify that the tokens are the same.
    tokenizer = fast_bert_tokenizer.FastBertTokenizer(
        vocab=vocab,
        token_out_type=dtypes.string,
        lower_case_nfd_strip_accents=lower_case_nfd_strip_accents)
    result_tokens = tokenizer.tokenize(text_inputs)
    self.assertAllEqual(result_tokens, expected_tokens)

    # Verify that the token ids are the same.
    token_id_tokenizer = fast_bert_tokenizer.FastBertTokenizer(
        vocab=vocab,
        token_out_type=dtypes.int64,
        lower_case_nfd_strip_accents=lower_case_nfd_strip_accents)
    result_token_ids = token_id_tokenizer.tokenize(text_inputs)
    self.assertAllEqual(result_token_ids, expected_token_ids)

    # Verify that the offsets are the same
    _, starts, ends = token_id_tokenizer.tokenize_with_offsets(text_inputs)
    self.assertAllEqual(starts, expected_starts)
    self.assertAllEqual(ends, expected_ends)


@parameterized.parameters([
    # Test 0.
    dict(
        vocab=[b'[UNK]', b'low', b'##er', b'it', b'is', b'sa', b'##me', b'.'],
        lower_case_nfd_strip_accents=True,
        text_inputs=[_utf8(u'\u1e9b\u0323'), b'LOWer', b'It is same.'],
    ),
    # Test 1.
    dict(
        vocab=[b'[UNK]', b'low', b'##er', b'it', b'is', b'sa', b'##me', b'.'],
        lower_case_nfd_strip_accents=False,
        text_inputs=[_utf8(u'\u1e9b\u0323'), b'LOWer', b'It is same.'],
    ),
])
class FastBertTokenizerKerasModelTest(test.TestCase):

  def testKerasAndTflite(self, vocab, lower_case_nfd_strip_accents,
                         text_inputs):
    """Checks TFLite conversion and inference."""

    class TextTokenizerModel(tf.keras.Model):

      def __init__(self, vocab, lower_case_nfd_strip_accents=False, **kwargs):
        super().__init__(**kwargs)
        self.fast_bert_tokenizer = fast_bert_tokenizer.FastBertTokenizer(
            vocab=vocab,
            lower_case_nfd_strip_accents=lower_case_nfd_strip_accents)

      def call(self, input_tensor, **kwargs):
        tokens = self.fast_bert_tokenizer.tokenize(input_tensor)
        return tokens.flat_values

    # Define a Keras model.
    model = TextTokenizerModel(
        vocab=vocab, lower_case_nfd_strip_accents=lower_case_nfd_strip_accents)
    # Build the Keras model by doing an inference.
    model(np.array(text_inputs))

    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Setup the TFLite interpreter.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Do TF.Text and TFLite inference.
    # TFLite only supports batch_size=1. So we construct a single-element input
    # array.
    test_data = np.array([text_inputs[0]])
    tf_result = model(test_data)

    interp.set_tensor(input_details[0]['index'], test_data)
    interp.invoke()
    tflite_result = interp.get_tensor(output_details[0]['index'])

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)


if __name__ == '__main__':
  test.main()
