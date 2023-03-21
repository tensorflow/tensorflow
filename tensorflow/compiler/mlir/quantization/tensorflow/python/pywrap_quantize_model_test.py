# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for pywrap_quantize_model.

These test cases are mostly for validation checks. Tests for functionalities
are at `quantize_model_test.py`.
"""
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model
from tensorflow.python.platform import test


class PywrapQuantizeModelTest(test.TestCase):
  """Test cases for quantize_model python wrappers."""

  def test_quantize_model_fails_when_invalid_quant_options_serialization(self):
    saved_model_path = self.create_tempdir('saved_model').full_path
    signature_def_keys = ['serving_default']
    tags = {'serve'}
    quant_opts_serialized = 'invalid protobuf serialization string'

    with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
      pywrap_quantize_model.quantize_ptq_model_pre_calibration(
          saved_model_path, signature_def_keys, tags, quant_opts_serialized
      )

  def test_quantize_model_fails_when_invalid_quant_options_type(self):
    saved_model_path = self.create_tempdir('saved_model').full_path
    signature_def_keys = ['serving_default']
    tags = {'serve'}
    invalid_quant_opts_object = ('a', 'b', 'c')

    with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
      pywrap_quantize_model.quantize_ptq_model_pre_calibration(
          saved_model_path, signature_def_keys, tags, invalid_quant_opts_object
      )


if __name__ == '__main__':
  test.main()
