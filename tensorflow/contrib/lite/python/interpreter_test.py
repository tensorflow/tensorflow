# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite Python Interface: Sanity check."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class InterpreterTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    interpreter = interpreter_wrapper.Interpreter(
        resource_loader.get_path_to_datafile('testdata/permute_float.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 4] == input_details[0]['shape']).all())

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())

    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    expected_output = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testUint8(self):
    interpreter = interpreter_wrapper.Interpreter(
        resource_loader.get_path_to_datafile('testdata/permute_uint8.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 4] == input_details[0]['shape']).all())

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())

    test_input = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    expected_output = np.array([[4, 3, 2, 1]], dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())


if __name__ == '__main__':
  test.main()
