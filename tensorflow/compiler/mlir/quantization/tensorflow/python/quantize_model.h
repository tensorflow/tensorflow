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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace tensorflow {
namespace quantization {

// Names of the TensorFlow Quantization steps. These names are used primarily
// for debugging.
inline constexpr absl::string_view kTfQuantPtqPreCalibrationStepName =
    "tf_quant_ptq_pre_calibration";
inline constexpr absl::string_view kTfQuantPtqPostCalibrationStepName =
    "tf_quant_ptq_post_calibration";
inline constexpr absl::string_view kTfQuantQatStepName = "tf_quant_qat";
inline constexpr absl::string_view kTfQuantPtqDynamicRangeStepName =
    "tf_quant_ptq_dynamic_range";
inline constexpr absl::string_view kTfQuantWeightOnlyStepName =
    "tf_quant_weight_only";

absl::StatusOr<ExportedModel> QuantizeQatModel(
    absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationOptions& quantization_options);

// Applies post-training dynamic-range quantization to the model.
absl::StatusOr<ExportedModel> QuantizeDynamicRangePtq(
    absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationOptions& quantization_options);

// Applies post-training static-range weight-only quantization to the model.
absl::StatusOr<ExportedModel> QuantizeWeightOnly(
    absl::string_view saved_model_path,
    const QuantizationOptions& quantization_options);

// Applies post-training static-range quantization to the model.
absl::StatusOr<ExportedModel> QuantizeStaticRangePtq(
    absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationOptions& quantization_options,
    const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map,
    const PyFunctionLibrary& py_function_library,
    const absl::flat_hash_map<std::string, RepresentativeDatasetFile>&
        representative_dataset_file_map_serialized);

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_H_
