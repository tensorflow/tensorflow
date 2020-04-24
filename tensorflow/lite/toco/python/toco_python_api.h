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
#ifndef TENSORFLOW_LITE_TOCO_PYTHON_TOCO_PYTHON_API_H_
#define TENSORFLOW_LITE_TOCO_PYTHON_TOCO_PYTHON_API_H_

#include <Python.h>

#include <string>

namespace toco {

// Convert a model represented in `input_contents`. `model_flags_proto`
// describes model parameters. `toco_flags_proto` describes conversion
// parameters (see relevant .protos for more information). Returns a string
// representing the contents of the converted model. When extended_return
// flag is set to true returns a dictionary that contains string representation
// of the converted model and some statistics like arithmetic ops count.
// `debug_info_str` contains the `GraphDebugInfo` proto. When
// `enable_mlir_converter` is True, use MLIR-based conversion instead of
// TOCO conversion.
PyObject* TocoConvert(PyObject* model_flags_proto_txt_raw,
                      PyObject* toco_flags_proto_txt_raw,
                      PyObject* input_contents_txt_raw,
                      bool extended_return = false,
                      PyObject* debug_info_txt_raw = nullptr,
                      bool enable_mlir_converter = false);

// Returns a list of names of all ops potentially supported by tflite.
PyObject* TocoGetPotentiallySupportedOps();

// Quantize the model with calibration data. Throw errors if `fully_quantize`
// is specified by the calibration data are not sufficient to quantize the
// model.
PyObject* MlirQuantizeModel(PyObject* data, bool fully_quantize);

// Sparsifies model to encode sparse tensors with proper format. Throws error if
// sparsification fails.
PyObject* MlirSparsifyModel(PyObject* data);
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_PYTHON_TOCO_PYTHON_API_H_
