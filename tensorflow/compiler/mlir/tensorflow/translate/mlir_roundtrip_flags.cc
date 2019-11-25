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

#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

Status ParseOutputArrayInfo(absl::string_view array_names,
                            absl::flat_hash_set<string>* array,
                            std::vector<string>* order) {
  std::vector<string> output_names = absl::StrSplit(array_names, ',');
  return ParseOutputArrayInfo(output_names, array, order);
}

Status ParseOutputArrayInfo(const std::vector<string>& output_names,
                            absl::flat_hash_set<string>* array,
                            std::vector<string>* order) {
  for (auto& output_name : output_names) {
    if (output_name.empty()) continue;
    array->insert(string(*absl::StrSplit(output_name, ':').begin()));
    order->push_back(output_name);
  }
  return Status::OK();
}

Status ParseInputArrayInfo(absl::string_view array_names,
                           absl::string_view data_types,
                           absl::string_view shapes,
                           GraphImportConfig::InputArrays* inputs) {
  std::vector<string> node_names = absl::StrSplit(array_names, ',');
  std::vector<string> node_dtypes = absl::StrSplit(data_types, ',');

  std::vector<string> node_shapes_str = absl::StrSplit(shapes, ':');
  std::vector<std::vector<int>> node_shapes;
  for (int i = 0; i < node_shapes_str.size(); i++) {
    std::vector<int> dims;
    for (auto& dim_str : absl::StrSplit(node_shapes_str[i], ',')) {
      // Treats empty input shape as scalar
      if (dim_str.empty()) continue;
      int size;
      TF_RET_CHECK(absl::SimpleAtoi(dim_str, &size));
      dims.push_back(size);
    }
    node_shapes.push_back(dims);
  }
  return ParseInputArrayInfo(node_names, node_dtypes, node_shapes, inputs);
}

Status ParseInputArrayInfo(const std::vector<string>& node_names,
                           const std::vector<string>& node_dtypes,
                           const std::vector<std::vector<int>>& node_shapes,
                           GraphImportConfig::InputArrays* inputs) {
  std::vector<std::string> used_node_dtypes;
  if (node_dtypes.empty() ||
      (node_dtypes.size() == 1 && node_dtypes[0].empty())) {
    // Mark all the node dtypes Invalid, so the importer can handle them by
    // using the type from the graph.
    used_node_dtypes.resize(node_names.size(), DataType_Name(DT_INVALID));
  } else if (node_names.size() == node_dtypes.size()) {
    for (auto dtype : node_dtypes) {
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

  if (node_names.size() != node_shapes.size()) {
    return errors::FailedPrecondition(absl::StrCat(
        "Unmatched node array and data type numbers (#arrays ",
        node_names.size(), ", #input_shapes ", node_shapes.size(), ")"));
  }

  // StringMap doesn't support reserve else reserve input map size here.
  for (int i = 0; i < node_names.size(); i++) {
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

    for (auto& dim : node_shapes[i]) {
      info.shape.add_dim()->set_size(dim);
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
