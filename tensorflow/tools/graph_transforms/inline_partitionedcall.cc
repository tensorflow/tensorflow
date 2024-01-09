/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

constexpr char kPartitionedCallOpName[] = "PartitionedCall";
constexpr char kFunctionAttrName[] = "f";

namespace {
absl::optional<FunctionDef> GetFunctionByNameFromLibrary(
    const GraphDef& graph, absl::string_view function_name) {
  for (const auto& fct : graph.library().function()) {
    if (fct.signature().name() == function_name) {
      return fct;
    }
  }
  return {};
}

std::string NormalizeNodeDefInput(const std::string& input_name) {
  std::vector<std::string> name_parts =
      absl::StrSplit(input_name, absl::ByChar(':'));
  if (name_parts.size() > 2) {
    return absl::StrCat(name_parts[0], ":", name_parts.back());
  }
  return input_name;
}

}  // namespace

Status InlinePartitionedCall(const GraphDef& input_graph_def,
                             const TransformFuncContext& context,
                             GraphDef* output_graph_def) {
  output_graph_def->Clear();
  absl::flat_hash_map<std::string, std::string> remap_input;

  for (const NodeDef& node : input_graph_def.node()) {
    if (node.op() == kPartitionedCallOpName) {
      if (node.attr().count(kFunctionAttrName) == 0) {
        return Status(
            absl::StatusCode::kNotFound,
            "Node " + node.name() + " has no attribute: " + kFunctionAttrName);
      }

      if (!node.attr().at(kFunctionAttrName).has_func()) {
        return Status(absl::StatusCode::kNotFound,
                      "Cannot figure out function name");
      }
      const std::string function_name =
          node.attr().at(kFunctionAttrName).func().name();
      absl::optional<FunctionDef> function =
          GetFunctionByNameFromLibrary(input_graph_def, function_name);
      if (!function.has_value()) {
        return Status(absl::StatusCode::kNotFound,
                      "function " + function_name + " Not found");
      }

      const std::string prefix = node.name();

      const int kOutputArgumentCount =
          function->signature().output_arg().size();
      for (int k = 0; k < kOutputArgumentCount; ++k) {
        const std::string function_arg_output_name =
            function->ret().at(function->signature().output_arg()[k].name());
        remap_input.insert_or_assign(
            CanonicalInputName(absl::StrCat(node.name(), ":", k)),
            absl::StrCat(prefix, "/",
                         NormalizeNodeDefInput(function_arg_output_name)));
      }

      const int kInputArgumentCount = function->signature().input_arg().size();
      if (node.input().size() != kInputArgumentCount) {
        return Status(absl::StatusCode::kInvalidArgument,
                      "Called function  " + function_name +
                          " has invalid input signature.");
      }
      absl::flat_hash_map<std::string, std::string> input_argument_map;
      for (int k = 0; k < kInputArgumentCount; ++k) {
        const std::string canonical_name =
            CanonicalInputName(function->signature().input_arg()[k].name());
        input_argument_map.insert_or_assign(canonical_name, node.input()[k]);
      }

      for (const NodeDef& function_node : function->node_def()) {
        NodeDef* new_node = output_graph_def->mutable_node()->Add();
        *new_node = function_node;
        new_node->set_name(absl::StrCat(prefix, "/", function_node.name()));
        absl::c_transform(
            *new_node->mutable_input(), new_node->mutable_input()->begin(),
            [prefix, input_argument_map](const std::string& input_name) {
              const std::string canonical_input_name =
                  CanonicalInputName(input_name);
              if (input_argument_map.find(canonical_input_name) !=
                  input_argument_map.end()) {
                return input_argument_map.at(canonical_input_name);
              }
              return absl::StrCat(prefix, "/",
                                  NormalizeNodeDefInput(input_name));
            });
      }
    } else {
      NodeDef* new_node = output_graph_def->mutable_node()->Add();
      *new_node = node;
    }
  }

  // Remap PartitionCall outputs to correct nodes.
  for (NodeDef& node : *output_graph_def->mutable_node()) {
    absl::c_transform(
        *node.mutable_input(), node.mutable_input()->begin(),
        [remap_input](const std::string& input_name) {
          const std::string canonical_input_name =
              CanonicalInputName(input_name);
          if (remap_input.find(canonical_input_name) != remap_input.end()) {
            return remap_input.at(canonical_input_name);
          }
          return input_name;
        });
  }
  return OkStatus();
}

REGISTER_GRAPH_TRANSFORM("inline_partitionedcall", InlinePartitionedCall);
}  // namespace graph_transforms
}  // namespace tensorflow
