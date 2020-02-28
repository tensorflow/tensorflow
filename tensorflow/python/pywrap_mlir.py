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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-import-order, g-bad-import-order, wildcard-import, unused-import, undefined-variable
from tensorflow.python import pywrap_tensorflow
from tensorflow.python._pywrap_mlir import *


def import_graphdef(graphdef, pass_pipeline):
  return ImportGraphDef(
      str(graphdef).encode('utf-8'),
      pass_pipeline.encode('utf-8'))


def experimental_convert_saved_model_to_mlir(saved_model_path, exported_names,
                                             show_debug_info):
  return ExperimentalConvertSavedModelToMlir(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'), show_debug_info)


def experimental_convert_saved_model_v1_to_mlir(saved_model_path, tags,
                                                show_debug_info):
  return ExperimentalConvertSavedModelV1ToMlir(
      str(saved_model_path).encode('utf-8'),
      str(tags).encode('utf-8'), show_debug_info)


def experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info):
  return ExperimentalRunPassPipeline(
      mlir_txt.encode('utf-8'), pass_pipeline.encode('utf-8'),
      show_debug_info)
