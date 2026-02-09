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

"""Tests for byte_splitter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def _split(s):
  return list(s.encode())


def _split_by_offsets(s, starts, ends):
  result = list()
  for start, end in zip(starts, ends):
    result.append(bytes(s, 'utf-8')[start:end])
  return result


@test_util.run_all_in_graph_and_eager_modes
class ByteSplitterTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(ByteSplitterTest, self).setUp()
    self.byte_splitter = tf_text.ByteSplitter()

  def testScalar(self):
    test_value = tf.constant('hello')
    expected_bytes = _split('hello')
    expected_start_offsets = range(5)
    expected_end_offsets = range(1, 6)
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytes)
    (bytez, start_offsets, end_offsets) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytes)
    self.assertAllEqual(start_offsets, expected_start_offsets)
    self.assertAllEqual(end_offsets, expected_end_offsets)

  def testVectorSingleValue(self):
    test_value = tf.constant(['hello'])
    expected_bytez = [_split('hello')]
    expected_offset_starts = [range(5)]
    expected_offset_ends = [range(1, 6)]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testVector(self):
    test_value = tf.constant(['hello', 'muñdʓ'])
    expected_bytez = [_split('hello'), _split('muñdʓ')]
    expected_offset_starts = [[*range(5)], [*range(7)]]
    expected_offset_ends = [[*range(1, 6)], [*range(1, 8)]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testMatrix(self):
    test_value = tf.constant([['hello', 'hola'],
                              ['goodbye', 'muñdʓ']])
    expected_bytez = [[_split('hello'), _split('hola')],
                      [_split('goodbye'), _split('muñdʓ')]]
    expected_offset_starts = [[[*range(5)], [*range(4)]],
                              [[*range(7)], [*range(7)]]]
    expected_offset_ends = [[[*range(1, 6)], [*range(1, 5)]],
                            [[*range(1, 8)], [*range(1, 8)]]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testMatrixRagged(self):
    test_value = tf.ragged.constant([['hello', 'hola'], ['muñdʓ']])
    expected_bytez = [[_split('hello'), _split('hola')], [_split('muñdʓ')]]
    expected_offset_starts = [[[*range(5)], [*range(4)]], [[*range(7)]]]
    expected_offset_ends = [[[*range(1, 6)], [*range(1, 5)]], [[*range(1, 8)]]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def test3DimMatrix(self):
    test_value = tf.constant([[['hello', 'hola'],
                               ['lol', 'ha']],
                              [['goodbye', 'muñdʓ'],
                               ['bye', 'mudʓ']]])
    expected_bytez = [[[_split('hello'), _split('hola')],
                       [_split('lol'), _split('ha')]],
                      [[_split('goodbye'), _split('muñdʓ')],
                       [_split('bye'), _split('mudʓ')]]]
    expected_offset_starts = [[[[*range(5)], [*range(4)]],
                               [[*range(3)], [*range(2)]]],
                              [[[*range(7)], [*range(7)]],
                               [[*range(3)], [*range(5)]]]]
    expected_offset_ends = [[[[*range(1, 6)], [*range(1, 5)]],
                             [[*range(1, 4)], [*range(1, 3)]]],
                            [[[*range(1, 8)], [*range(1, 8)]],
                             [[*range(1, 4)], [*range(1, 6)]]]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def test3DimMatrixRagged(self):
    test_value = tf.ragged.constant([[['hello'], ['lol', 'ha']],
                                     [['bye', 'mudʓ']]])
    expected_bytez = [[[_split('hello')], [_split('lol'), _split('ha')]],
                      [[_split('bye'), _split('mudʓ')]]]
    expected_offset_starts = [[[[*range(5)]],
                               [[*range(3)], [*range(2)]]],
                              [[[*range(3)], [*range(5)]]]]
    expected_offset_ends = [[[[*range(1, 6)]],
                             [[*range(1, 4)], [*range(1, 3)]]],
                            [[[*range(1, 4)], [*range(1, 6)]]]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyStringSingle(self):
    test_value = tf.constant('')
    expected_bytez = []
    expected_offset_starts = []
    expected_offset_ends = []
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyStrings(self):
    test_value = tf.constant(['', 'hello', '', 'muñdʓ', ''])
    expected_bytez = [[], _split('hello'), [], _split('muñdʓ'), []]
    expected_offset_starts = [[], [*range(5)], [], [*range(7)], []]
    expected_offset_ends = [[], [*range(1, 6)], [], [*range(1, 8)], []]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyDimensions(self):
    test_value = tf.ragged.constant([[[], ['lol', 'ha']], []])
    expected_bytez = [[[], [_split('lol'), _split('ha')]], []]
    expected_offset_starts = [[[], [[*range(3)], [*range(2)]]], []]
    expected_offset_ends = [[[], [[*range(1, 4)], [*range(1, 3)]]], []]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testTfLite(self):
    """Checks TFLite conversion and inference."""

    class Model(tf.keras.Model):

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bs = tf_text.ByteSplitter()

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='input')
      ])
      def call(self, input_tensor):
        return {'result': self.bs.split(input_tensor).flat_values}

    # Test input data.
    input_data = np.array(['Some minds are better kept apart'])

    # Define a model.
    model = Model()
    # Do TF inference.
    tf_result = model(tf.constant(input_data))['result']

    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Do TFLite inference.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    print(interp.get_signature_list())
    split = interp.get_signature_runner('serving_default')
    output = split(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output['result']
    else:
      tflite_result = output['output_1']

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)

    input_data = np.array(['Some minds are better kept apart oh'])
    tf_result = model(tf.constant(input_data))['result']

    output = split(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output['result']
    else:
      tflite_result = output['output_1']
    self.assertAllEqual(tflite_result, tf_result)

  def testSplitByOffsetsScalar(self):
    test_value = 'hello'
    starts = [0, 4]
    ends = [4, 5]
    vals = self.byte_splitter.split_by_offsets(test_value, starts, ends)
    self.assertAllEqual(vals, _split_by_offsets(test_value, starts, ends))

  def testSplitByOffsetsVectorSingleValue(self):
    test_value = ['hello']
    starts = [0, 4]
    ends = [4, 5]
    vals = self.byte_splitter.split_by_offsets(test_value, [starts], [ends])
    self.assertAllEqual(vals, [_split_by_offsets(test_value[0], starts, ends)])

  def testSplitByOffsetsVector(self):
    test_value = ['hello', 'muñdʓ']
    starts = [[0, 4], [0, 2]]
    ends = [[4, 5], [2, 7]]
    expected = [_split_by_offsets(test_value[0], starts[0], ends[0]),
                _split_by_offsets(test_value[1], starts[1], ends[1])]
    vals = self.byte_splitter.split_by_offsets(test_value, starts, ends)
    self.assertAllEqual(vals, expected)

  def testSplitByOffsetsMatrix(self):
    test_value = [['hello', 'hola'], ['goodbye', 'muñdʓ']]
    starts = [[[0, 4], [0, 2]], [[0, 4], [0, 2]]]
    ends = [[[4, 5], [2, 4]], [[3, 7], [2, 7]]]
    expected = []
    for i in range(2):
      expected.append([_split_by_offsets(
          test_value[i][j], starts[i][j], ends[i][j])
                       for j in range(len(ends[i]))])
    vals = self.byte_splitter.split_by_offsets(test_value, starts, ends)
    self.assertAllEqual(vals, expected)

  def testSplitByOffsetsMatrixRagged(self):
    test_value = [['hello', 'hola'], ['goodbye']]
    starts = [[[0], [0, 2]], [[0, 4]]]
    ends = [[[4], [2, 4]], [[3, 7]]]
    expected = []
    expected.append([_split_by_offsets(
        test_value[0][j], starts[0][j], ends[0][j]) for j in range(2)])
    expected.append([_split_by_offsets(
        test_value[1][j], starts[1][j], ends[1][j]) for j in range(1)])
    vals = self.byte_splitter.split_by_offsets(
        tf.ragged.constant(test_value),
        tf.ragged.constant(starts),
        tf.ragged.constant(ends))
    self.assertAllEqual(vals, expected)

  def testSplitByOffsets3DimMatrix(self):
    test_value = [[['hello', 'hola'],
                   ['lolz', 'haha']],
                  [['goodbye', 'muñdʓ'],
                   ['bye', 'mudʓ']]]
    starts = [[[[0, 4], [0, 2]],
               [[0, 3], [0, 2]]],
              [[[0, 4], [0, 2]],
               [[0, 3], [0, 2]]]]
    ends = [[[[4, 5], [2, 4]],
             [[3, 4], [2, 4]]],
            [[[3, 7], [2, 7]],
             [[3, 3], [2, 5]]]]
    expected = []
    for i in range(2):
      row = []
      for j in range(2):
        row.append([_split_by_offsets(
            test_value[i][j][k], starts[i][j][k], ends[i][j][k]) for k in
                    range(2)])
      expected.append(row)
    vals = self.byte_splitter.split_by_offsets(test_value, starts, ends)
    self.assertAllEqual(vals, expected)

  def testSplitByOffsets3DimMatrixRagged(self):
    test_value = [[['hello', 'hola'],
                   ['lolz', 'ha']],
                  [['bye', 'mudʓ']]]
    starts = [[[[0, 4], [0, 2]],
               [[0], [0, 1]]],
              [[[0, 0], [0, 1, 2, 3, 4]]]]
    ends = [[[[4, 5], [2, 4]],
             [[3], [1, 2]]],
            [[[0, 3], [1, 2, 3, 4, 5]]]]
    expected = []
    row = []
    for j in range(2):
      row.append([_split_by_offsets(
          test_value[0][j][k], starts[0][j][k], ends[0][j][k]) for k in
                  range(2)])
    expected.append(row)
    expected.append([[_split_by_offsets(
        test_value[1][0][k], starts[1][0][k], ends[1][0][k]) for k in
                      range(2)]])
    vals = self.byte_splitter.split_by_offsets(
        tf.ragged.constant(test_value),
        tf.ragged.constant(starts),
        tf.ragged.constant(ends))
    self.assertAllEqual(vals, expected)

  def testSplitByOffsetsEmptyString(self):
    test_value = ['', 'hello', '']
    starts = [[], [0, 4], []]
    ends = [[], [4, 5], []]
    expected = [_split_by_offsets(test_value[0], starts[0], ends[0]),
                _split_by_offsets(test_value[1], starts[1], ends[1]),
                _split_by_offsets(test_value[2], starts[2], ends[2])]
    vals = self.byte_splitter.split_by_offsets(test_value,
                                               tf.ragged.constant(starts),
                                               tf.ragged.constant(ends))
    self.assertAllEqual(vals, expected)

  def testSplitByOffsetsEmptyOffsets(self):
    test_value = ['hello']
    starts = [[]]
    ends = [[]]
    vals = self.byte_splitter.split_by_offsets(test_value,
                                               tf.constant(starts, tf.int32),
                                               tf.constant(ends, tf.int32))
    self.assertAllEqual(vals,
                        [_split_by_offsets(test_value[0], starts[0], ends[0])])

  def testSplitByOffsetsTfLite(self):
    """Checks TFLite conversion and inference."""

    class Model(tf.keras.Model):

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bs = tf_text.ByteSplitter()

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='input')
      ])
      def call(self, input_tensor):
        return {'result': self.bs.split(input_tensor).flat_values}

    # Test input data.
    input_data = np.array(['Some minds are better kept apart'])

    # Define a model.
    model = Model()
    # Do TF inference.
    tf_result = model(tf.constant(input_data))['result']

    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Do TFLite inference.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    print(interp.get_signature_list())
    split = interp.get_signature_runner('serving_default')
    output = split(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output['result']
    else:
      tflite_result = output['output_1']

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)

    input_data = np.array(['Some minds are better kept apart oh'])
    tf_result = model(tf.constant(input_data))['result']

    output = split(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output['result']
    else:
      tflite_result = output['output_1']
    self.assertAllEqual(tflite_result, tf_result)


if __name__ == '__main__':
  test.main()
