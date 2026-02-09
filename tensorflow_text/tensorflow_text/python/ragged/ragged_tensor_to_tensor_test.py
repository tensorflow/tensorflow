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
"""Tests for ragged_tensor_to_tensor op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToTensorTest(test_util.TensorFlowTestCase):

  def testTfLite(self):
    """Checks TFLite conversion and inference."""

    class TokenizerModel(tf.keras.Model):

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fwp = tf_text.FastWordpieceTokenizer(['minds', 'apart', '[UNK]'])

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='input')
      ])
      def call(self, input_tensor):
        return {'tokens': self.fwp.tokenize(input_tensor).to_tensor()}

    # Test input data.
    input_data = np.array(['Some minds are better kept apart'])

    # Define a model.
    model = TokenizerModel()
    # Do TF inference.
    tf_result = model(tf.constant(input_data))['tokens']

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
    tokenize = interp.get_signature_runner('serving_default')
    output = tokenize(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output['tokens']
    else:
      tflite_result = output['output_1']

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)


if __name__ == '__main__':
  test.main()
