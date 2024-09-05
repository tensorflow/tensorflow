/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERTER_PYTHON_API_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERTER_PYTHON_API_H_

#include <Python.h>

#include <string>
#include <vector>

#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"

namespace tflite {

// Convert a model represented in `input_contents`. `model_flags_proto`
// describes model parameters. `flags_proto` describes conversion
// parameters (see relevant .protos for more information). Returns a string
// representing the contents of the converted model. When extended_return
// flag is set to true returns a dictionary that contains string representation
// of the converted model and some statistics like arithmetic ops count.
// `debug_info_str` contains the `GraphDebugInfo` proto. When
// `enable_mlir_converter` is True, use MLIR-based conversion instead of
// TOCO conversion.
PyObject* Convert(PyObject* model_flags_proto_txt_raw,
                  PyObject* toco_flags_proto_txt_raw,
                  PyObject* input_contents_txt_raw,
                  bool extended_return = false,
                  PyObject* debug_info_txt_raw = nullptr,
                  bool enable_mlir_converter = false,
                  const tensorflow::quantization::PyFunctionLibrary*
                      quantization_py_function_library = nullptr);

// Quantize the model with calibration data. Throw errors if `fully_quantize`
// is specified by the calibration data are not sufficient to quantize the
// model.
PyObject* MlirQuantizeModel(PyObject* data, bool disable_per_channel,
                            bool fully_quantize, int inference_type,
                            int input_data_type, int output_data_type,
                            bool enable_numeric_verify = false,
                            bool enable_whole_model_verify = false,
                            PyObject* op_denylist = nullptr,
                            PyObject* node_denylist = nullptr,
                            bool enable_variable_quantization = false,
                            bool disable_per_channel_for_dense_layers = false,
                            PyObject* debug_options_proto_txt_raw = nullptr);

// Sparsifies model to encode sparse tensors with proper format. Throws error if
// sparsification fails.
PyObject* MlirSparsifyModel(PyObject* data);

// Registers the given custom opdefs to TensorFlow global op registry.
PyObject* RegisterCustomOpdefs(PyObject* list);

// Returns the collected TFLite conversion errors.
std::vector<std::string> RetrieveCollectedErrors();

// Returns MLIR string dump of the given Flatbuffer model.
std::string FlatBufferFileToMlir(const std::string& model,
                                 bool input_is_filepath);

// All the exported functions should be listed in
// tensorflow/tools/def_file_filter/symbols_pybind.txt for the Windows build.
}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_CONVERTER_PYTHON_API_H_
