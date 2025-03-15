# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python module for MLIR functions exported by pybind11."""

# pylint: disable=invalid-import-order, g-bad-import-order, wildcard-import, unused-import, undefined-variable
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *

# Helper function to reduce repetitive encoding
def _encode(s):
  return str(s).encode('utf-8')

def import_graphdef(
    graphdef,
    pass_pipeline,
    show_debug_info,
    input_names=None,
    input_data_types=None,
    input_data_shapes=None,
    output_names=[],
):
  if input_names is not None:
    return ImportGraphDef(
        _encode(graphdef),
        _encode(pass_pipeline),
        show_debug_info,
        _encode(','.join(input_names)),
        _encode(','.join(input_data_types)),
        _encode(':'.join(input_data_shapes)),
        _encode(','.join(output_names)),
    )
  return ImportGraphDef(
      _encode(graphdef),
      _encode(pass_pipeline),
      show_debug_info,
  )

def import_function(concrete_function, pass_pipeline, show_debug_info):
  ctxt = context.context()
  ctxt.ensure_initialized()
  return ImportFunction(
      ctxt._handle,
      _encode(concrete_function.function_def),
      _encode(pass_pipeline),
      show_debug_info,
  )

def experimental_convert_saved_model_to_mlir(
    saved_model_path, exported_names, show_debug_info
):
  return ExperimentalConvertSavedModelToMlir(
      _encode(saved_model_path),
      _encode(exported_names),
      show_debug_info,
  )

def experimental_convert_saved_model_v1_to_mlir_lite(
    saved_model_path, exported_names, tags, upgrade_legacy, show_debug_info
):
  return ExperimentalConvertSavedModelV1ToMlirLite(
      _encode(saved_model_path),
      _encode(exported_names),
      _encode(tags),
      upgrade_legacy,
      show_debug_info,
  )

def experimental_convert_saved_model_v1_to_mlir(
    saved_model_path,
    exported_names,
    tags,
    lift_variables,
    include_variables_in_initializers,
    upgrade_legacy,
    show_debug_info,
):
  return ExperimentalConvertSavedModelV1ToMlir(
      _encode(saved_model_path),
      _encode(exported_names),
      _encode(tags),
      lift_variables,
      include_variables_in_initializers,
      upgrade_legacy,
      show_debug_info,
  )

def experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info):
  return ExperimentalRunPassPipeline(
      _encode(mlir_txt), _encode(pass_pipeline), show_debug_info
  )

def experimental_write_bytecode(filename, mlir_txt):
  return ExperimentalWriteBytecode(_encode(filename), _encode(mlir_txt))

def experimental_tflite_to_tosa_bytecode(
    flatbuffer,
    bytecode,
    use_external_constant=False,
    ordered_input_arrays=[],
    ordered_output_arrays=[],
):
  return _pywrap_mlir.ExperimentalTFLiteToTosaBytecode(
      _encode(flatbuffer),
      _encode(bytecode),
      use_external_constant,
      ordered_input_arrays,
      ordered_output_arrays,
  )