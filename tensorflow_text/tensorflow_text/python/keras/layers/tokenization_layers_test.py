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

"""Tests for Keras tokenization_layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_text.python.keras.layers import tokenization_layers


class TokenizationLayersTest(tf.test.TestCase, parameterized.TestCase):

  # The test below uses `keras_test_utils.layer_test` which is a private API,
  # so it has been disabled.
  # Please move the implementation of `layer_test` to this codebase to reenable
  # the test.
  # @parameterized.named_parameters(
  #     {
  #         'cls': tokenization_layers.UnicodeScriptTokenizer,
  #         'input_shape': (1,),
  #         'input_data': [b'I love Flume!'],
  #         'expected': [[b'I', b'love', b'Flume', b'!']],
  #         'kwargs': {
  #             'squeeze_token_dim': False,
  #         },
  #         'testcase_name': 'unicode_layer',
  #     },
  #     {
  #         'cls': tokenization_layers.WhitespaceTokenizer,
  #         'input_shape': (1,),
  #         'input_data': [b'I love Flume!'],
  #         'expected': [[b'I', b'love', b'Flume!']],
  #         # TODO(raw-pointer): layer test will fail if squeeze_token_dim option
  #         # is disabled. Not sure if it is layer_test limitaitons or the layer
  #         # itself. Fix it when layer_test is updated.
  #         'kwargs': {
  #             'squeeze_token_dim': False,
  #         },
  #         'testcase_name': 'whitespace_layer',
  #     },
  #     {
  #         'cls':
  #             tokenization_layers.WordpieceTokenizer,
  #         'input_shape': (
  #             1,
  #             None,
  #         ),
  #         'kwargs': {
  #             'vocabulary': [
  #                 b'don',
  #                 b"##'",
  #                 b'##t',
  #                 b'tread',
  #                 b'##ness',
  #                 b'hel',
  #                 b'##lo',
  #                 b'there',
  #                 b'my',
  #                 b'na',
  #                 b'##me',
  #                 b'is',
  #                 b'ter',
  #                 b'##ry',
  #                 b'what',
  #                 b'##cha',
  #                 b'##ma',
  #                 b'##call',
  #                 b'##it?',
  #                 b'you',
  #                 b'said',
  #             ],
  #             'merge_wordpiece_dim': False
  #         },
  #         'input_data':
  #             np.array([[b"don't", b'treadness', b'whatchamacallit?']]),
  #         'expected': [[[b'don', b"##'", b'##t'], [b'tread', b'##ness'],
  #                       [b'what', b'##cha', b'##ma', b'##call', b'##it?']]],
  #         'testcase_name':
  #             'wordpiece_layer',
  #     })
  # def test_tokenizer_layer_sequential(self,
  #                                     cls,
  #                                     input_shape,
  #                                     input_data=None,
  #                                     expected=None,
  #                                     kwargs=None):
  #   # TODO(raw-pointer): was there meant to be a wordpiece test that tests the
  #   # layers on an empty zero-value tensor? I think Keras doesn't support that
  #   #  in TF2. Or was it meant to test on an empty string?

  #   if not tf.executing_eagerly():
  #     # In TF1 list-of-lists-of-scalars need to be wrapped in an extra list
  #     # for single-io models, because it tries to disambiguate which
  #     # input to send an input to (which causes issues w/ single io models)
  #     input_data = [input_data]

  #   output_data = keras_test_utils.layer_test(
  #       cls,
  #       kwargs=kwargs,
  #       validate_training=False,
  #       input_shape=input_shape,
  #       input_dtype='string',
  #       input_data=input_data,
  #       test_harness=self,
  #   )
  #   self.assertAllEqual(output_data, expected)

  @parameterized.named_parameters(
      {
          'batch_input_shape': (None,),
          'batch_output_shape': (None, None),
          'testcase_name': 'basic_test',
      }, {
          'batch_input_shape': (2, 3),
          'batch_output_shape': (2, 3, None),
          'testcase_name': 'multi_dimensional',
      })
  def test_compute_output_signature(self,
                                    batch_input_shape,
                                    batch_output_shape):
    layer = tokenization_layers.WhitespaceTokenizer()
    self.assertEqual(
        layer.compute_output_signature(
            tf.TensorSpec(batch_input_shape, tf.string)),
        tf.TensorSpec(batch_output_shape, tf.string))

  @parameterized.named_parameters(
      {
          'input_data': [b'I love Flume!'],
          'expected_output': [[b'I', b'love', b'Flume!']],
          'input_shape': (1,),
          'testcase_name': 'basic_test',
      }, {
          'input_data': [[b'I love Flume!'], [b'Good day']],
          'expected_output': [[b'I', b'love', b'Flume!'], [b'Good', b'day']],
          'input_shape': (1,),
          'testcase_name': 'batch_of_2',
      }, {
          'input_data': [[b'  '], [b'  ']],
          'expected_output': [[], []],
          'input_shape': (1,),
          'testcase_name': 'batch_of_2_all_whitespace',
      }, {
          'input_data': np.array([['I love Flume!', 'Good day']]),
          'expected_output': [[[b'I', b'love', b'Flume!'], [b'Good', b'day']]],
          'input_shape': (None,),
          'squeeze_token_dim': False,
          'testcase_name': 'multi_dimensional',
      }, {
          'input_data': np.array([[b'I love Flume!', b'Good day']]),
          'expected_output': [[[b'I', b'love', b'Flume!'],
                               [b'Good', b'day', b'[PAD]']]],
          'pad_value': b'[PAD]',
          'input_shape': (None,),
          'squeeze_token_dim': False,
          'testcase_name': 'multi_dim_with_padding',
      })
  def test_whitespace_tokenization_layer(self,
                                         input_data,
                                         expected_output,
                                         input_shape=(None,),
                                         pad_value=None,
                                         squeeze_token_dim=True):
    if (not tf.executing_eagerly() and
        not isinstance(input_data, np.ndarray)):
      # In TF1 list-of-lists-of-scalars need to be wrapped in an extra list
      # for single-io models, because it tries to disambiguate which
      # input to send an input to (which causes issues w/ single io models)
      input_data = [input_data]

    # create a functional API model
    i = tf.keras.layers.Input(shape=input_shape, dtype=tf.string)
    layer = tokenization_layers.WhitespaceTokenizer(
        pad_value=pad_value, squeeze_token_dim=squeeze_token_dim)
    o = layer(i)
    model = tf.keras.models.Model(i, o)
    self.assertAllEqual(model.predict(input_data), expected_output)

  @parameterized.named_parameters(
      {
          'input_data': [[b'I love Flume!']],
          'expected_output': [[b'I', b'love', b'Flume', b'!']],
          'input_shape': (1,),
          'testcase_name': 'basic_test',
      }, {
          'input_data': [[b'I love Flume!'], [b'Good day']],
          'expected_output': [[b'I', b'love', b'Flume', b'!'],
                              [b'Good', b'day']],
          'input_shape': (1,),
          'testcase_name': 'batch_of_2',
      }, {
          'input_data': [[b'  '], [b'  ']],
          'expected_output': [[], []],
          'input_shape': (1,),
          'testcase_name': 'batch_of_2_all_whitespace',
      }, {
          'input_data': np.array([[b'I love Flume!', b'Good day']]),
          'expected_output': [[[b'I', b'love', b'Flume', b'!'],
                               [b'Good', b'day']]],
          'squeeze_token_dim': False,
          'testcase_name': 'multi_dimensional',
      }, {
          'input_data': np.array([[b'I love Flume!', b'Good day']]),
          'expected_output': [[[b'I', b'love', b'Flume', b'!'],
                               [b'Good', b'day', b'[PAD]', b'[PAD]']]],
          'pad_value': b'[PAD]',
          'input_shape': (None,),
          'squeeze_token_dim': False,
          'testcase_name': 'multi_dim_with_padding',
      })
  def test_unicode_tokenization_layer(self,
                                      input_data,
                                      expected_output,
                                      input_shape=(None,),
                                      pad_value=None,
                                      squeeze_token_dim=True):
    if (not tf.executing_eagerly() and
        not isinstance(input_data, np.ndarray)):
      # In TF1 list-of-lists-of-scalars need to be wrapped in an extra list
      # for single-io models, because it tries to disambiguate which
      # input to send an input to (which causes issues w/ single io models)
      input_data = [input_data]

    # create a functional API model
    i = tf.keras.layers.Input(shape=input_shape, dtype=tf.string)
    layer = tokenization_layers.UnicodeScriptTokenizer(
        pad_value=pad_value, squeeze_token_dim=squeeze_token_dim)
    o = layer(i)
    model = tf.keras.models.Model(i, o)
    self.assertAllEqual(model.predict(input_data), expected_output)

  def test_unicode_tokenization_in_text_vec(self):
    input_data = [[b'I love Flume!']]
    expected_output = [[b'i', b'love', b'flume']]
    if (not tf.executing_eagerly() and
        not isinstance(input_data, np.ndarray)):
      # In TF1 list-of-lists-of-scalars need to be wrapped in an extra list
      # for single-io models, because it tries to disambiguate which
      # input to send an input to (which causes issues w/ single io models)
      input_data = [input_data]

    # create a functional API model
    i = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    splitter = tokenization_layers.UnicodeScriptTokenizer()
    layer = tf.keras.layers.TextVectorization(
        split=splitter, output_mode=None)
    o = layer(i)
    model = tf.keras.models.Model(i, o)

    # evaluate the model
    self.assertAllEqual(model.predict(input_data), expected_output)

  @parameterized.named_parameters(
      {
          'input_data': [[b' ']],
          'expected_output': [[[b'[UNK]']]],
          'vocab_list': [
              b'don',
              b"##'",
              b'##t',
              b'tread',
              b'##ness',
              b'hel',
              b'##lo',
              b'there',
              b'my',
              b'na',
              b'##me',
              b'is',
              b'ter',
              b'##ry',
              b'what',
              b'##cha',
              b'##ma',
              b'##call',
              b'##it?',
              b'you',
              b'said',
          ],
          'input_shape': (1,),
          'merge_wordpiece_dim': False,
          'testcase_name': 'basic_test_whitespace',
      },
      {
          'input_data': [[b"don't"]],
          'expected_output': [[[b'don', b"##'", b'##t']]],
          'vocab_list': [
              b'don',
              b"##'",
              b'##t',
              b'tread',
              b'##ness',
              b'hel',
              b'##lo',
              b'there',
              b'my',
              b'na',
              b'##me',
              b'is',
              b'ter',
              b'##ry',
              b'what',
              b'##cha',
              b'##ma',
              b'##call',
              b'##it?',
              b'you',
              b'said',
          ],
          'input_shape': (1,),
          'merge_wordpiece_dim': False,
          'testcase_name': 'basic_test',
      },
      {
          'input_data':
              np.array([[b"don't", b'treadness', b'whatchamacallit?']]),
          'expected_output': [[[b'don', b"##'", b'##t'], [
              b'tread', b'##ness'
          ], [b'what', b'##cha', b'##ma', b'##call', b'##it?']]],
          'vocab_list': [
              b'don',
              b"##'",
              b'##t',
              b'tread',
              b'##ness',
              b'hel',
              b'##lo',
              b'there',
              b'my',
              b'na',
              b'##me',
              b'is',
              b'ter',
              b'##ry',
              b'what',
              b'##cha',
              b'##ma',
              b'##call',
              b'##it?',
              b'you',
              b'said',
          ],
          'input_shape': (3,),
          'merge_wordpiece_dim':
              False,
          'testcase_name':
              'multi_dimensional',
      },
      {
          'input_data':
              np.array([[b"don't", b'treadness', b'whatchamacallit?']]),
          'expected_output': [[[
              b'don', b"##'", b'##t', b'[PAD]', b'[PAD]'
          ], [b'tread', b'##ness', b'[PAD]', b'[PAD]', b'[PAD]'
             ], [b'what', b'##cha', b'##ma', b'##call', b'##it?']]],
          'vocab_list': [
              b'don',
              b"##'",
              b'##t',
              b'tread',
              b'##ness',
              b'hel',
              b'##lo',
              b'there',
              b'my',
              b'na',
              b'##me',
              b'is',
              b'ter',
              b'##ry',
              b'what',
              b'##cha',
              b'##ma',
              b'##call',
              b'##it?',
              b'you',
              b'said',
          ],
          'pad_value':
              b'[PAD]',
          'input_shape': (3,),
          'merge_wordpiece_dim':
              False,
          'testcase_name':
              'multi_dim_with_padding',
      },
  )
  def test_wordpiece_tokenization_layer(self,
                                        input_data,
                                        expected_output,
                                        vocab_list,
                                        input_shape=(None,),
                                        pad_value=None,
                                        merge_wordpiece_dim=False):
    if (not tf.executing_eagerly() and
        not isinstance(input_data, np.ndarray)):
      # In TF1 list-of-lists-of-scalars need to be wrapped in an extra list
      # for single-io models, because it tries to disambiguate which
      # input to send an input to (which causes issues w/ single io models)
      input_data = [input_data]

    # create a functional API model
    i = tf.keras.layers.Input(shape=input_shape, dtype=tf.string)
    layer = tokenization_layers.WordpieceTokenizer(
        vocabulary=vocab_list,
        pad_value=pad_value,
        merge_wordpiece_dim=merge_wordpiece_dim)
    o = layer(i)
    model = tf.keras.models.Model(i, o)
    self.assertAllEqual(model.predict(input_data), expected_output)

  @parameterized.named_parameters({
      'input_data': [[b'I love Flume!'], [b'Good day']],
      'expected_output': [[b'I', b'love', b'Flume!'], [b'Good', b'day']],
      'input_shape': (1,),
      'testcase_name': 'batch_of_2',
  })
  def test_whitespace_tokenization_multi_layer(self,
                                               input_data,
                                               expected_output,
                                               input_shape=(None,),
                                               pad_value=None,
                                               squeeze_token_dim=True):
    if not tf.executing_eagerly():
      # In TF1 list-of-lists-of-scalars need to be wrapped in an extra list
      # for single-io models, because it tries to disambiguate which
      # input to send an input to (which causes issues w/ single io models)
      input_data = [input_data]

    # create a functional API model
    i = tf.keras.layers.Input(shape=input_shape, dtype=tf.string)
    layer1 = tokenization_layers.WhitespaceTokenizer(
        pad_value=pad_value, squeeze_token_dim=squeeze_token_dim)
    z = layer1(i)
    layer2 = tokenization_layers.WhitespaceTokenizer(
        pad_value=pad_value, squeeze_token_dim=squeeze_token_dim)
    o = layer2(i)
    model = tf.keras.models.Model(i, [z, o])
    out1, out2 = model.predict(input_data)
    self.assertAllEqual(out1, expected_output)
    self.assertAllEqual(out2, expected_output)

  @parameterized.named_parameters({
      'input_data': [[b'I love Flume!'], [b'Good day']],
      'expected_output': [[b'I', b'love', b'Flume', b'!'], [b'Good', b'day']],
      'input_shape': (1,),
      'testcase_name': 'batch_of_2',
  })
  def test_unicode_tokenization_multi_layer(self,
                                            input_data,
                                            expected_output,
                                            input_shape=(None,),
                                            pad_value=None,
                                            squeeze_token_dim=True):
    if not tf.executing_eagerly():
      # In TF1 list-of-lists-of-scalars need to be wrapped in an extra list
      # for single-io models, because it tries to disambiguate which
      # input to send an input to (which causes issues w/ single io models)
      input_data = [input_data]

    # create a functional API model
    i = tf.keras.layers.Input(shape=input_shape, dtype=tf.string)
    layer1 = tokenization_layers.UnicodeScriptTokenizer(
        pad_value=pad_value, squeeze_token_dim=squeeze_token_dim)
    z = layer1(i)
    layer2 = tokenization_layers.UnicodeScriptTokenizer(
        pad_value=pad_value, squeeze_token_dim=squeeze_token_dim)
    o = layer2(i)
    model = tf.keras.models.Model(i, [z, o])
    out1, out2 = model.predict(input_data)
    self.assertAllEqual(out1, expected_output)
    self.assertAllEqual(out2, expected_output)

  @parameterized.named_parameters(
      {
          'input_data':
              np.array([[b"don't", b'treadness', b'whatchamacallit?']]),
          'expected_output': [[[b'don', b"##'", b'##t'], [
              b'tread', b'##ness'
          ], [b'what', b'##cha', b'##ma', b'##call', b'##it?']]],
          'vocab_list': [
              b'don',
              b"##'",
              b'##t',
              b'tread',
              b'##ness',
              b'hel',
              b'##lo',
              b'there',
              b'my',
              b'na',
              b'##me',
              b'is',
              b'ter',
              b'##ry',
              b'what',
              b'##cha',
              b'##ma',
              b'##call',
              b'##it?',
              b'you',
              b'said',
          ],
          'input_shape': (3,),
          'merge_wordpiece_dim':
              False,
          'testcase_name':
              'multi_dimensional',
      },
      {
          'input_data':
              np.array([[b"don't", b'treadness', b'whatchamacallit?']]),
          'expected_output': [[[b'don', b"##'", b'##t'], [
              b'tread', b'##ness'
          ], [b'what', b'##cha', b'##ma', b'##call', b'##it?']]],
          'vocab_list': [
              b'don',
              b"##'",
              b'##t',
              b'tread',
              b'##ness',
              b'hel',
              b'##lo',
              b'there',
              b'my',
              b'na',
              b'##me',
              b'is',
              b'ter',
              b'##ry',
              b'what',
              b'##cha',
              b'##ma',
              b'##call',
              b'##it?',
              b'you',
              b'said',
          ],
          'input_shape': (3,),
          'merge_wordpiece_dim':
              True,
          'testcase_name':
              'merge',
      },
  )
  def test_wordpiece_tokenization_multi_layer(self,
                                              input_data,
                                              expected_output,
                                              vocab_list,
                                              input_shape=(None,),
                                              pad_value=None,
                                              merge_wordpiece_dim=False):
    # create a functional API model
    i = tf.keras.layers.Input(shape=input_shape, dtype=tf.string)
    layer1 = tokenization_layers.WordpieceTokenizer(
        vocabulary=vocab_list,
        pad_value=pad_value,
        merge_wordpiece_dim=merge_wordpiece_dim)
    o1 = layer1(i)
    layer2 = tokenization_layers.WordpieceTokenizer(
        vocabulary=vocab_list,
        pad_value=pad_value,
        merge_wordpiece_dim=merge_wordpiece_dim)
    o2 = layer2(i)
    model = tf.keras.models.Model(i, [o1, o2])

    out1, out2 = model.predict(input_data)
    self.assertAllEqual(out1, expected_output)
    self.assertAllEqual(out2, expected_output)

if __name__ == '__main__':
  tf.test.main()
