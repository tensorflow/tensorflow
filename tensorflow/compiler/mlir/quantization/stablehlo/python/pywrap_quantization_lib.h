/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PYTHON_PYWRAP_QUANTIZATION_LIB_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PYTHON_PYWRAP_QUANTIZATION_LIB_H_

// Contains mirror functions from StableHLO Quantizer to be exposed to python
// via `pywrap_quantization`.

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace stablehlo::quantization::pywrap {

// Function used by the pywrap_quantization module to mirror
// `::mlir::quant::stablehlo::QuantizeStaticRangePtq`.
absl::Status PywrapQuantizeStaticRangePtq(
    absl::string_view src_saved_model_path,
    absl::string_view dst_saved_model_path, const QuantizationConfig& config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, tensorflow::SignatureDef>&
        signature_def_map,
    const tensorflow::quantization::PyFunctionLibrary& py_function_library);

// Function used by the pywrap_quantization module to mirror
// `::mlir::quant::stablehlo::QuantizeWeightOnlyPtq`.
absl::Status PywrapQuantizeWeightOnlyPtq(
    absl::string_view src_saved_model_path,
    absl::string_view dst_saved_model_path, const QuantizationConfig& config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, tensorflow::SignatureDef>&
        signature_def_map,
    const tensorflow::quantization::PyFunctionLibrary& py_function_library);

// Function used by the pywrap_quantization module to mirror
// `::stablehlo::quantization::PopulateDefaults`.
QuantizationConfig PywrapPopulateDefaults(
    const QuantizationConfig& user_provided_config);

// Function used by the pywrap_quantization module to mirror
// `::stablehlo::quantization::ExpandPresets`.
QuantizationConfig PywrapExpandPresets(const QuantizationConfig& config);

}  // namespace stablehlo::quantization::pywrap

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PYTHON_PYWRAP_QUANTIZATION_LIB_H_
