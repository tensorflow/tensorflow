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
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model
from tensorflow.python.platform import test


class PywrapQuantizeModelTest(test.TestCase):
  """Test cases for quantize_model python wrappers."""

  def test_quantize_model_fails_when_invalid_quant_options_serialization(self):
    src_saved_model_path = self.create_tempdir().full_path
    dst_saved_model_path = self.create_tempdir().full_path
    signature_def_keys = ['serving_default']
    quant_opts_serialized = 'invalid proto serialization string'.encode('utf-8')

    with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
      pywrap_quantize_model.quantize_ptq_static_range(
          src_saved_model_path,
          dst_saved_model_path,
          quant_opts_serialized,
          signature_keys=signature_def_keys,
          signature_def_map_serialized={},
          function_aliases={},
          py_function_library=py_function_lib.PyFunctionLibrary(),
          representative_dataset_file_map_serialized=None,
      )

  def test_quantize_model_fails_when_invalid_quant_options_type(self):
    src_saved_model_path = self.create_tempdir().full_path
    dst_saved_model_path = self.create_tempdir().full_path
    signature_def_keys = ['serving_default']
    invalid_quant_opts_object = ('a', 'b', 'c')

    with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
      pywrap_quantize_model.quantize_ptq_static_range(
          src_saved_model_path,
          dst_saved_model_path,
          invalid_quant_opts_object,
          signature_keys=signature_def_keys,
          signature_def_map_serialized={},
          function_aliases={},
          py_function_library=py_function_lib.PyFunctionLibrary(),
          representative_dataset_file_map_serialized=None,
      )


if __name__ == '__main__':
  test.main()
