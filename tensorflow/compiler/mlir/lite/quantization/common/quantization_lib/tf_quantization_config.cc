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

#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/tf_quantization_config.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace TFL {

InferenceDataType ConvertDataTypeToInferenceDataType(
    const tensorflow::DataType type) {
  switch (type) {
    case tensorflow::DT_FLOAT:
      return InferenceDataType::kFloat;
    case tensorflow::DT_HALF:
      return InferenceDataType::kHalf;
    case tensorflow::DT_INT8:
      return InferenceDataType::kInt8;
    case tensorflow::DT_UINT8:
      return InferenceDataType::kUInt8;
    case tensorflow::DT_INT16:
      return InferenceDataType::kInt16;
    case tensorflow::DT_UINT16:
      return InferenceDataType::kUInt16;
    case tensorflow::DT_INT32:
      return InferenceDataType::kInt32;
    case tensorflow::DT_QINT8:
      return InferenceDataType::kQInt8;
    case tensorflow::DT_QUINT8:
      return InferenceDataType::kQUInt8;
    case tensorflow::DT_QINT16:
      return InferenceDataType::kQInt16;
    case tensorflow::DT_QUINT16:
      return InferenceDataType::kQUInt16;
    case tensorflow::DT_QINT32:
      return InferenceDataType::kQInt32;
    default:
      return InferenceDataType::kInvalid;
  }
}

bool ParseInputNodeQuantSpecs(const absl::string_view node_names,
                              const absl::string_view min_values,
                              const absl::string_view max_values,
                              const absl::string_view inference_type,
                              QuantizationSpecs* quant_specs) {
  const std::vector<std::string> input_nodes = absl::StrSplit(node_names, ',');
  std::vector<std::optional<double>> node_mins;
  if (!min_values.empty()) {
    std::vector<std::string> node_mins_str = absl::StrSplit(min_values, ',');
    for (const std::string& node_mins_str : node_mins_str) {
      double value;
      if (!absl::SimpleAtod(node_mins_str, &value)) {
        llvm::errs() << "Unexpected mins: " << node_mins_str << "\n";
        return true;
      }
      node_mins.push_back(value);
    }
  }

  std::vector<std::optional<double>> node_maxs;
  if (!max_values.empty()) {
    const std::vector<std::string> node_maxs_str =
        absl::StrSplit(max_values, ',');
    for (const std::string& node_maxs_str : node_maxs_str) {
      double value;
      if (!absl::SimpleAtod(node_maxs_str, &value)) {
        llvm::errs() << "Unexpected maxs: " << node_maxs_str << "\n";
        return true;
      }
      node_maxs.push_back(value);
    }
  }

  tensorflow::DataType final_type = tensorflow::DT_FLOAT;
  if (!inference_type.empty() && !DataType_Parse(inference_type, &final_type)) {
    return true;
  }
  return GetInputNodeQuantSpecs(input_nodes, node_mins, node_maxs, final_type,
                                quant_specs);
}

bool GetInputNodeQuantSpecs(const std::vector<std::string>& node_names,
                            const std::vector<std::optional<double>>& node_mins,
                            const std::vector<std::optional<double>>& node_maxs,
                            const tensorflow::DataType inference_type,
                            QuantizationSpecs* quant_specs) {
  return GetInputNodeQuantSpecs(
      node_names, node_mins, node_maxs,
      ConvertDataTypeToInferenceDataType(inference_type), quant_specs);
}

}  // namespace TFL
}  // namespace mlir
