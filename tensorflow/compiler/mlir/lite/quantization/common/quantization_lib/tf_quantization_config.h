/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Conversion from `tensorflow::DataType` to the TensorFlow-independent
// quantization config in `quantization_config.h`, plus helpers that take
// `tensorflow::DataType`. Only TensorFlow-facing code (e.g. the TensorFlow to
// TFLite converter) should depend on this library.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_COMMON_QUANTIZATION_LIB_TF_QUANTIZATION_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_COMMON_QUANTIZATION_LIB_TF_QUANTIZATION_CONFIG_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace TFL {

// Converts `type` to the corresponding `InferenceDataType`. Returns `kInvalid`
// for data types that have no `InferenceDataType` counterpart.
InferenceDataType ConvertDataTypeToInferenceDataType(tensorflow::DataType type);

// Parses the command line flag strings to the quantization specification for
// input arrays of a graph. `inference_type` is a `tensorflow::DataType` name,
// e.g. "DT_QINT8". The array names are not stored in the spec, and will be
// matched by position. Returns true if failed.
bool ParseInputNodeQuantSpecs(absl::string_view node_names,
                              absl::string_view min_values,
                              absl::string_view max_values,
                              absl::string_view inference_type,
                              QuantizationSpecs* quant_specs);

// Same as the `InferenceDataType` overload in `quantization_config.h`, but
// takes a `tensorflow::DataType`. Returns true if failed.
bool GetInputNodeQuantSpecs(const std::vector<std::string>& node_names,
                            const std::vector<std::optional<double>>& node_mins,
                            const std::vector<std::optional<double>>& node_maxs,
                            tensorflow::DataType inference_type,
                            QuantizationSpecs* quant_specs);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_COMMON_QUANTIZATION_LIB_TF_QUANTIZATION_CONFIG_H_
