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
"""Tests for whitespace_tokenizer_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops.whitespace_tokenizer import WhitespaceTokenizer


@test_util.run_all_in_graph_and_eager_modes
class WhitespaceTokenizerOpTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(WhitespaceTokenizerOpTest, self).setUp()
    self.whitespace_tokenizer = WhitespaceTokenizer()

  def testScalar(self):
    test_value = constant_op.constant(b'I love Flume!')
    expected_tokens = [b'I', b'love', b'Flume!']
    expected_offset_starts = [0, 2, 7]
    expected_offset_ends = [1, 6, 13]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testScalarWithSplit(self):
    # Similar to testScalar, but using split() calls (instead of tokenize()).
    # Should produce the same results as before.  This tests that a
    # WhitespaceTokenizer is a valid Splitter.
    test_value = constant_op.constant(b'I love Flume!')
    expected_tokens = [b'I', b'love', b'Flume!']
    expected_offset_starts = [0, 2, 7]
    expected_offset_ends = [1, 6, 13]
    tokens = self.whitespace_tokenizer.split(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.split_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testVectorSingleValue(self):
    test_value = constant_op.constant([b'I love Flume!'])
    expected_tokens = [[b'I', b'love', b'Flume!']]
    expected_offset_starts = [[0, 2, 7]]
    expected_offset_ends = [[1, 6, 13]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testVector(self):
    test_value = constant_op.constant([b'I love Flume!', b'Good day'])
    expected_tokens = [[b'I', b'love', b'Flume!'], [b'Good', b'day']]
    expected_offset_starts = [[0, 2, 7], [0, 5]]
    expected_offset_ends = [[1, 6, 13], [4, 8]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testMatrix(self):
    test_value = constant_op.constant([[b'I love Flume!', b'Good day'],
                                       [b'I don\'t want', b'no scrubs']])
    expected_tokens = [[[b'I', b'love', b'Flume!'], [b'Good', b'day']],
                       [[b'I', b'don\'t', b'want'], [b'no', b'scrubs']]]
    expected_offset_starts = [[[0, 2, 7], [0, 5]], [[0, 2, 8], [0, 3]]]
    expected_offset_ends = [[[1, 6, 13], [4, 8]], [[1, 7, 12], [2, 9]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testMatrixRagged(self):
    test_value = ragged_factory_ops.constant([[b'I love Flume!'],
                                              [b'I don\'t want', b'no scrubs']])
    expected_tokens = [[[b'I', b'love', b'Flume!']],
                       [[b'I', b'don\'t', b'want'], [b'no', b'scrubs']]]
    expected_offset_starts = [[[0, 2, 7]], [[0, 2, 8], [0, 3]]]
    expected_offset_ends = [[[1, 6, 13]], [[1, 7, 12], [2, 9]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def test3DimMatrix(self):
    test_value = constant_op.constant([[[b'I love Flume!', b'Good day'],
                                        [b'I don\'t want', b'no scrubs']],
                                       [[b'I love Zhu!', b'Good night'],
                                        [b'A scrub is', b'a guy']]])
    expected_tokens = [[[[b'I', b'love', b'Flume!'], [b'Good', b'day']],
                        [[b'I', b'don\'t', b'want'], [b'no', b'scrubs']]],
                       [[[b'I', b'love', b'Zhu!'], [b'Good', b'night']],
                        [[b'A', b'scrub', b'is'], [b'a', b'guy']]]]
    expected_offset_starts = [[[[0, 2, 7], [0, 5]], [[0, 2, 8], [0, 3]]],
                              [[[0, 2, 7], [0, 5]], [[0, 2, 8], [0, 2]]]]
    expected_offset_ends = [[[[1, 6, 13], [4, 8]], [[1, 7, 12], [2, 9]]],
                            [[[1, 6, 11], [4, 10]], [[1, 7, 10], [1, 5]]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def test3DimMatrixRagged(self):
    test_value = ragged_factory_ops.constant([[[b'I love Flume!'],
                                               [b'I don\'t want',
                                                b'no scrubs']],
                                              [[b'I love Zhu!',
                                                b'Good night']]])
    expected_tokens = [[[[b'I', b'love', b'Flume!']],
                        [[b'I', b'don\'t', b'want'], [b'no', b'scrubs']]],
                       [[[b'I', b'love', b'Zhu!'], [b'Good', b'night']]]]
    expected_offset_starts = [[[[0, 2, 7]], [[0, 2, 8], [0, 3]]],
                              [[[0, 2, 7], [0, 5]]]]
    expected_offset_ends = [[[[1, 6, 13]], [[1, 7, 12], [2, 9]]],
                            [[[1, 6, 11], [4, 10]]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testInternationalization(self):
    test_value = constant_op.constant(
        [u"J'adore la灯".encode('utf8'), u'¡Escríbeme!'.encode('utf8')])
    expected_tokens = [[b'J\'adore', u'la灯'.encode('utf8')],
                       [u'¡Escríbeme!'.encode('utf8')]]
    expected_offset_starts = [[0, 8], [0]]
    expected_offset_ends = [[7, 13], [13]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testSpaceBoundaries(self):
    test_value = constant_op.constant([b' Hook em! ', b' .Ok.   Go  '])
    expected_tokens = [[b'Hook', b'em!'], [b'.Ok.', b'Go']]
    expected_offset_starts = [[1, 6], [1, 8]]
    expected_offset_ends = [[5, 9], [5, 10]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testOnlySpaces(self):
    test_value = constant_op.constant([b' ', b'     ', b' \t\r\n'])
    expected_tokens = [[], [], []]
    expected_offset_starts = [[], [], []]
    expected_offset_ends = [[], [], []]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testWhitespaceCharacters(self):
    test_value = constant_op.constant([b'things:\tcarpet\rdesk\nlamp\r\nlove'])
    expected_tokens = [[b'things:', b'carpet', b'desk', b'lamp', b'love']]
    expected_offset_starts = [[0, 8, 15, 20, 26]]
    expected_offset_ends = [[7, 14, 19, 24, 30]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyStringSingle(self):
    test_value = constant_op.constant([b''])
    expected_tokens = [[]]
    expected_offset_starts = [[]]
    expected_offset_ends = [[]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyString(self):
    test_value = constant_op.constant(
        [b'', b'I love Flume!', b'', b'O hai', b''])
    expected_tokens = [[], [b'I', b'love', b'Flume!'], [], [b'O', b'hai'], []]
    expected_offset_starts = [[], [0, 2, 7], [], [0, 2], []]
    expected_offset_ends = [[], [1, 6, 13], [], [1, 5], []]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyDimensions(self):
    test_value = ragged_factory_ops.constant(
        [[[b'I love Flume!', b'Good day. . .'], []], [],
         [[b'I love Zhu!', b'Good night'], [b'A scrub is', b'a guy']]])
    expected_tokens = [[[[b'I', b'love', b'Flume!'],
                         [b'Good', b'day.', b'.', b'.']], []], [],
                       [[[b'I', b'love', b'Zhu!'], [b'Good', b'night']],
                        [[b'A', b'scrub', b'is'], [b'a', b'guy']]]]
    expected_offset_starts = [[[[0, 2, 7], [0, 5, 10, 12]], []], [],
                              [[[0, 2, 7], [0, 5]], [[0, 2, 8], [0, 2]]]]
    expected_offset_ends = [[[[1, 6, 13], [4, 9, 11, 13]], []], [],
                            [[[1, 6, 11], [4, 10]], [[1, 7, 10], [1, 5]]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertAllEqual(tokens, expected_tokens)
    (tokens, starts, ends) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testTfLite(self):
    """Checks TFLite conversion and inference."""

    class TokenizerModel(tf.keras.Model):

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = WhitespaceTokenizer()

      def call(self, input_tensor, **kwargs):
        return self.tokenizer.tokenize(input_tensor).flat_values

    # Test input data.
    input_data = np.array(['Some minds are better kept apart'])

    # Define a Keras model.
    model = TokenizerModel()
    # Do TF.Text inference.
    tf_result = model(tf.constant(input_data))

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
    interp.set_tensor(input_details[0]['index'], input_data)
    interp.invoke()
    output_details = interp.get_output_details()
    tflite_result = interp.get_tensor(output_details[0]['index'])

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)


if __name__ == '__main__':
  test.main()
