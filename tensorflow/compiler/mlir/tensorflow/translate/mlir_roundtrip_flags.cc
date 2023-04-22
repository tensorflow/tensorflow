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

#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"

#include <ostream>
#include <sstream>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/Optional.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

std::string GraphImportConfig::str() const {
  std::ostringstream ss;

  ss << "graph_func_name: " << graph_func_name;
  InputArrays inputs;
  ss << "\ninputs: ";
  for (auto& it : inputs) {
    ss << "\n\t" << it.first << " -> "
       << DataTypeString(it.second.imported_dtype) << " "
       << it.second.shape.DebugString();
  }
  ss << "\noutputs:";
  for (auto& output : outputs) ss << " " << output;
  ss << "\ncontrol_outputs:";
  for (auto& output : control_outputs) ss << " " << output;
  ss << "\nprune_unused_nodes: " << prune_unused_nodes;
  ss << "\nconvert_legacy_fed_inputs: " << convert_legacy_fed_inputs;
  ss << "\ngraph_as_function: " << graph_as_function;
  ss << "\nupgrade_legacy: " << upgrade_legacy;
  ss << "\nrestrict_functionalization_to_tpu_nodes: "
     << restrict_functionalization_to_tpu_nodes;
  ss << "\nenable_shape_inference: " << enable_shape_inference;

  return ss.str();
}

Status ParseOutputArrayInfo(absl::string_view array_names,
                            std::vector<string>* outputs) {
  TF_RETURN_IF_ERROR(ParseNodeNames(array_names, *outputs));
  return Status::OK();
}

Status ParseOutputArrayInfo(const std::vector<string>& output_names,
                            std::vector<string>* outputs) {
  for (auto& output_name : output_names) {
    if (output_name.empty()) continue;
    outputs->push_back(output_name);
  }
  return Status::OK();
}

Status ParseInputArrayInfo(absl::string_view array_names,
                           absl::string_view data_types,
                           absl::string_view shapes,
                           GraphImportConfig::InputArrays* inputs) {
  std::vector<string> node_names;
  std::vector<string> node_dtypes;
  std::vector<llvm::Optional<std::vector<int>>> node_shapes;
  TF_RETURN_IF_ERROR(ParseNodeNames(array_names, node_names));
  TF_RETURN_IF_ERROR(ParseNodeDataTypes(data_types, node_dtypes));
  TF_RETURN_IF_ERROR(ParseNodeShapes(shapes, node_shapes));
  return ParseInputArrayInfo(node_names, node_dtypes, node_shapes, inputs);
}

Status ParseInputArrayInfo(
    const std::vector<string>& node_names,
    const std::vector<string>& node_dtypes,
    const std::vector<llvm::Optional<std::vector<int>>>& node_shapes,
    GraphImportConfig::InputArrays* inputs) {
  std::vector<std::string> used_node_dtypes;
  if (node_dtypes.empty()) {
    // Mark all the node dtypes Invalid, so the importer can handle them by
    // using the type from the graph.
    used_node_dtypes.resize(node_names.size(), DataType_Name(DT_INVALID));
  } else if (node_names.size() == node_dtypes.size()) {
    for (const auto& dtype : node_dtypes) {
      if (dtype.empty()) {
        used_node_dtypes.push_back(DataType_Name(DT_INVALID));
      } else if (dtype != DataType_Name(DT_INVALID)) {
        used_node_dtypes.push_back(dtype);
      } else {
        return errors::FailedPrecondition(
            "Use '' if want to use the type from graph.");
      }
    }
  } else {
    return errors::FailedPrecondition(absl::StrCat(
        "Unmatched node array and data type numbers (#arrays ",
        node_names.size(), ", #data_types ", node_dtypes.size(), ")"));
  }

  if (!node_shapes.empty() && node_names.size() != node_shapes.size()) {
    return errors::FailedPrecondition(absl::StrCat(
        "Unmatched node array and shape numbers (#arrays ", node_names.size(),
        ", #input_shapes ", node_shapes.size(), ")"));
  }

  // StringMap doesn't support reserve else reserve input map size here.
  for (int i = 0, end = node_names.size(); i < end; i++) {
    auto& name = node_names[i];
    if (name.empty()) continue;

    auto it_inserted_pair = inputs->insert({name, {}});
    if (!it_inserted_pair.second)
      return errors::FailedPrecondition(
          absl::StrCat("tensor ", name, " is repeated in the arrays flag"));

    ArrayInfo& info = it_inserted_pair.first->second;
    if (!DataType_Parse(used_node_dtypes[i], &info.imported_dtype)) {
      return errors::FailedPrecondition(
          absl::StrCat("Invalid node type '", node_dtypes[i], "'"));
    }

    if (!node_shapes.empty()) {
      if (!node_shapes[i].hasValue()) {
        info.shape.set_unknown_rank(true);
        continue;
      }
      for (auto& dim : node_shapes[i].getValue()) {
        info.shape.add_dim()->set_size(dim);
      }
    }
  }
  return Status::OK();
}

Status ParseNodeShapes(
    absl::string_view shapes_str,
    std::vector<llvm::Optional<std::vector<int>>>& shapes_vector) {
  shapes_vector.clear();
  if (!shapes_str.empty()) {
    std::vector<string> node_shapes_str = absl::StrSplit(shapes_str, ':');
    for (int i = 0; i < node_shapes_str.size(); i++) {
      if (node_shapes_str[i] == "*") {
        shapes_vector.push_back(llvm::None);
        continue;
      }
      std::vector<int> dims;
      for (const absl::string_view dim_str :
           absl::StrSplit(node_shapes_str[i], ',')) {
        // Treats empty input shape as scalar
        if (dim_str.empty()) continue;
        if (dim_str == "?") {
          dims.push_back(-1);
          continue;
        }
        int size;
        TF_RET_CHECK(absl::SimpleAtoi(dim_str, &size));
        dims.push_back(size);
      }
      shapes_vector.push_back(dims);
    }
  }
  return Status::OK();
}

Status ParseNodeNames(absl::string_view names_str,
                      std::vector<std::string>& names_vector) {
  names_vector = absl::StrSplit(names_str, ',', absl::SkipEmpty());
  return Status::OK();
}

Status ParseNodeDataTypes(absl::string_view data_types_str,
                          std::vector<std::string>& data_type_vector) {
  data_type_vector.clear();
  if (!data_types_str.empty()) {
    data_type_vector = absl::StrSplit(data_types_str, ',');
  }
  return Status::OK();
}

}  // namespace tensorflow
