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
        str(graphdef).encode('utf-8'),
        pass_pipeline.encode('utf-8'),
        show_debug_info,
        ','.join(input_names).encode('utf-8'),
        ','.join(input_data_types).encode('utf-8'),
        ':'.join(input_data_shapes).encode('utf-8'),
        ','.join(output_names).encode('utf-8'),
    )
  return ImportGraphDef(
      str(graphdef).encode('utf-8'),
      pass_pipeline.encode('utf-8'),
      show_debug_info,
  )


def import_function(concrete_function, pass_pipeline, show_debug_info):
  ctxt = context.context()
  ctxt.ensure_initialized()
  return ImportFunction(
      ctxt._handle,
      str(concrete_function.function_def).encode('utf-8'),
      pass_pipeline.encode('utf-8'),
      show_debug_info,
  )


def experimental_convert_saved_model_to_mlir(
    saved_model_path, exported_names, show_debug_info
):
  return ExperimentalConvertSavedModelToMlir(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'),
      show_debug_info,
  )


def experimental_convert_saved_model_v1_to_mlir_lite(
    saved_model_path, exported_names, tags, upgrade_legacy, show_debug_info
):
  return ExperimentalConvertSavedModelV1ToMlirLite(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'),
      str(tags).encode('utf-8'),
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
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'),
      str(tags).encode('utf-8'),
      lift_variables,
      include_variables_in_initializers,
      upgrade_legacy,
      show_debug_info,
  )


def experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info):
  return ExperimentalRunPassPipeline(
      mlir_txt.encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info
  )


def experimental_write_bytecode(filename, mlir_txt):
  return ExperimentalWriteBytecode(filename.encode('utf-8'), mlir_txt.encode())
