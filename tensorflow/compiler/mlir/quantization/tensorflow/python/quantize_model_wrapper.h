/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_

#include <pybind11/stl.h>

#include <string>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace tensorflow {
namespace quantization {

PyObject* QuantizeQATModel(const absl::string_view saved_model_path,
                           const absl::string_view exported_names_str,
                           const absl::string_view tags,
                           const std::string& quant_opts_serialized);

PyObject* QuantizePTQDynamicRange(const absl::string_view saved_model_path,
                                  const absl::string_view exported_names_str,
                                  const absl::string_view tags,
                                  const std::string& quant_opts_serialized);

PyObject* QuantizePTQModelPreCalibration(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags);

PyObject* QuantizePTQModelPostCalibration(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const std::string& quant_opts_serialized);

void ClearCollectedInformationFromCalibrator();

void ClearDataFromCalibrator(absl::string_view id);

float GetMinFromCalibrator(absl::string_view id);

float GetMaxFromCalibrator(absl::string_view id);

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_
