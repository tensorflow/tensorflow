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
#include "tensorflow/compiler/mlir/quantization/stablehlo/python/pywrap_quantization_lib.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_static_range_ptq.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_weight_only_ptq.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace stablehlo::quantization::pywrap {

using ::mlir::tf_quant::stablehlo::QuantizeStaticRangePtq;
using ::mlir::tf_quant::stablehlo::QuantizeWeightOnlyPtq;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::PyFunctionLibrary;

// Note for maintainers: the definitions should ONLY mirror existing functions
// defined in different targets. Do not include any extra business logic that
// causes divergence from the semantics of mirrored functions.

absl::Status PywrapQuantizeStaticRangePtq(
    absl::string_view src_saved_model_path,
    absl::string_view dst_saved_model_path, const QuantizationConfig& config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map,
    const PyFunctionLibrary& py_function_library) {
  return QuantizeStaticRangePtq(src_saved_model_path, dst_saved_model_path,
                                config, signature_keys, signature_def_map,
                                py_function_library);
}

absl::Status PywrapQuantizeWeightOnlyPtq(
    absl::string_view src_saved_model_path,
    absl::string_view dst_saved_model_path, const QuantizationConfig& config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map,
    const PyFunctionLibrary& py_function_library) {
  return QuantizeWeightOnlyPtq(src_saved_model_path, dst_saved_model_path,
                               config, signature_keys, signature_def_map,
                               py_function_library);
}

QuantizationConfig PywrapPopulateDefaults(
    const QuantizationConfig& user_provided_config) {
  return PopulateDefaults(user_provided_config);
}

QuantizationConfig PywrapExpandPresets(const QuantizationConfig& config) {
  return ExpandPresets(config);
}

}  // namespace stablehlo::quantization::pywrap
