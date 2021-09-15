# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for auto_quant."""

import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.lite.tools.optimize.auto_quant import auto_quant
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class AutoQuantTest(test_util.TensorFlowTestCase):

  def test_quant_single_layers(self):

    class Multiplier(tf.Module):

      @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
      def multiply(self, x):
        return x * x

    model = Multiplier()
    model_path = tempfile.mkdtemp()
    tf.saved_model.save(model, model_path)

    def representative_dataset():
      for i in range(5):
        yield [np.array([i], dtype=np.float32)]

    quanted_models = auto_quant.quant_single_layers(model_path,
                                                    representative_dataset)

    for flatbuffer, _ in quanted_models:
      model = flatbuffer_utils.convert_bytearray_to_object(flatbuffer)
      quant_index = -1
      for i, opcode in enumerate(model.operatorCodes):
        if opcode.builtinCode == schema_fb.BuiltinOperator.QUANTIZE:
          quant_index = i
          break
      self.assertNotEqual(quant_index, -1)

      quant_cnt = 0
      for graph in model.subgraphs:
        for op in graph.operators:
          if op.opcodeIndex == quant_index:
            quant_cnt += 1
      self.assertEqual(quant_cnt, 1)


if __name__ == '__main__':
  test.main()
