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

"""Tests for the UTF-8 binarization op."""

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class Utf8BinarizeTest(test_util.TensorFlowTestCase):

  def _lowest_bits_little_endian(self, char, n_bits):
    binary = '000000000000000000000' + bin(ord(char))[2:]
    return [float(bit) for bit in reversed(binary)][:n_bits]

  def testUtf8BinarizeScalar(self):
    test_value = '爱上一个不回'
    expected_values = sum([self._lowest_bits_little_endian(c, n_bits=4)
                           for c in test_value[:3]], [])
    values = tf_text.utf8_binarize(test_value, word_length=3, bits_per_char=4)
    self.assertAllEqual(values, expected_values)

  def testUtf8BinarizeVector(self):
    test_value = ['爱上一个不回', 'foo']
    expected_values = []
    for s in test_value:
      expected_values.append(
          sum([self._lowest_bits_little_endian(c, n_bits=4)
               for c in s[:3]], []))
    values = tf_text.utf8_binarize(test_value, word_length=3, bits_per_char=4)
    self.assertAllEqual(values, expected_values)

  def testUtf8BinarizeMatrix(self):
    test_value = [['爱上一个不回'], ['foo']]
    expected_values = []
    for row in test_value:
      expected_values.append([])
      for s in row:
        expected_values[-1].append(
            sum([self._lowest_bits_little_endian(c, n_bits=4)
                 for c in s[:3]], []))
    values = tf_text.utf8_binarize(test_value, word_length=3, bits_per_char=4)
    self.assertAllEqual(values, expected_values)

  def testUtf8BinarizeTfLite(self):
    """Checks TFLite conversion and inference."""

    class Model(tf.keras.Model):

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='input')
      ])
      def call(self, input_tensor):
        return {'result': tf_text.utf8_binarize(input_tensor, word_length=3,
                                                bits_per_char=4)}

    # Test input data.
    input_data = np.array(['爱上一个不回'])

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


if __name__ == '__main__':
  test.main()
