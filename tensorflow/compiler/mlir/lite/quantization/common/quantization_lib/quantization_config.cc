/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"

#include <ios>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace TFL {
namespace {
bool GetBooleanSpecs(const std::string& bool_val) {
  bool result;
  std::stringstream iss(bool_val);
  iss >> std::boolalpha >> result;
  return result;
}
}  // namespace

absl::string_view InferenceDataTypeName(const InferenceDataType type) {
  switch (type) {
    case InferenceDataType::kInvalid:
      return "DT_INVALID";
    case InferenceDataType::kFloat:
      return "DT_FLOAT";
    case InferenceDataType::kHalf:
      return "DT_HALF";
    case InferenceDataType::kInt8:
      return "DT_INT8";
    case InferenceDataType::kUInt8:
      return "DT_UINT8";
    case InferenceDataType::kInt16:
      return "DT_INT16";
    case InferenceDataType::kUInt16:
      return "DT_UINT16";
    case InferenceDataType::kInt32:
      return "DT_INT32";
    case InferenceDataType::kQInt8:
      return "DT_QINT8";
    case InferenceDataType::kQUInt8:
      return "DT_QUINT8";
    case InferenceDataType::kQInt16:
      return "DT_QINT16";
    case InferenceDataType::kQUInt16:
      return "DT_QUINT16";
    case InferenceDataType::kQInt32:
      return "DT_QINT32";
  }
  return "DT_INVALID";
}

bool IsQuantizedInferenceDataType(const InferenceDataType type) {
  switch (type) {
    case InferenceDataType::kQInt8:
    case InferenceDataType::kQUInt8:
    case InferenceDataType::kQInt16:
    case InferenceDataType::kQUInt16:
    case InferenceDataType::kQInt32:
      return true;
    default:
      return false;
  }
}

void ParseCustomOpSpecs(const absl::string_view node_names,
                        const CustomOpUpdateOptions& update_option,
                        CustomOpMap& custom_op_map) {
  if (node_names.empty()) return;

  const std::vector<std::string> custom_nodes = absl::StrSplit(node_names, ',');

  for (const std::string& cur_node : custom_nodes) {
    const std::vector<std::string> node_infos = absl::StrSplit(cur_node, '=');
    const std::string& node_name = node_infos[0];
    const std::string& node_specification = node_infos[1];
    CustomOpInfo new_node_info;
    switch (update_option) {
      case CustomOpUpdateOptions::kInputIndices: {
        const std::vector<std::string> indices =
            absl::StrSplit(node_specification, '-');
        for (const std::string& cur_index : indices) {
          custom_op_map[node_name].quantizable_input_indices.push_back(
              std::stoi(cur_index));
        }
        break;
      }
      case CustomOpUpdateOptions::kWeightOnly:
        custom_op_map[node_name].is_weight_only =
            GetBooleanSpecs(node_specification);
        break;
      case CustomOpUpdateOptions::kNoSideEffect:
        custom_op_map[node_name].no_side_effect =
            GetBooleanSpecs(node_specification);
        break;
    }
  }
}

bool GetInputNodeQuantSpecs(const std::vector<std::string>& node_names,
                            const std::vector<std::optional<double>>& node_mins,
                            const std::vector<std::optional<double>>& node_maxs,
                            const InferenceDataType inference_type,
                            QuantizationSpecs* quant_specs) {
  quant_specs->inference_type = inference_type;

  // If min/max are not specified, just return;
  if (node_mins.empty() || node_maxs.empty()) return false;

  // Otherwise make sure min/max has the same size as inputs.
  if (IsQuantizedInferenceDataType(inference_type)) {
    // min/max should have same size as inputs, or shouldn't be specified.
    if (node_names.size() != node_mins.size() ||
        node_names.size() != node_maxs.size()) {
      return true;
    }
    for (int i = 0; i < node_names.size(); ++i) {
      quant_specs->input_ranges.push_back({node_mins[i], node_maxs[i]});
    }
    return false;
  }
  if (!node_mins.empty()) {
    llvm::dbgs() << "Ignored input_min_values.";
  }
  if (!node_maxs.empty()) {
    llvm::dbgs() << "Ignored input_max_values.";
  }
  return false;
}

std::string GetQDQQuantModeString(const QDQConversionMode mode) {
  switch (mode) {
    case QDQConversionMode::kQDQStatic:
      return "Static";
    case QDQConversionMode::kQDQDynamic:
      return "Dynamic";
    case QDQConversionMode::kQDQStrict:
      return "Strict";
    default:
      return "NoQDQ";
  }
}

QDQConversionMode GetQDQQuantModeFromString(const std::string& mode_str) {
  if (mode_str == "Static") return QDQConversionMode::kQDQStatic;
  if (mode_str == "Dynamic") return QDQConversionMode::kQDQDynamic;
  if (mode_str == "Strict") return QDQConversionMode::kQDQStrict;
  return QDQConversionMode::kQDQNone;
}

}  // namespace TFL
}  // namespace mlir
