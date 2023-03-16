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

#include "tensorflow/core/graph/regularization/simple_delete.h"

#include <string>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/regularization/util.h"
#include "tensorflow/core/grappler/op_types.h"

namespace tensorflow::graph_regularization {

namespace {

// This function mutates `graph_def`, changing the names and config_proto's
// of the Function nodes.
void RegularizeNodes(GraphDef* graph_def) {
  for (NodeDef& node : *graph_def->mutable_node()) {
    // Check if this is a function call.
    if (grappler::IsPartitionedCall(node) ||
        grappler::IsStatefulPartitionedCall(node)) {
      // Regularize "f" attribute, the function name for PartitionedCall and
      // and StatefulPartitionedCall ops, by stripping the suffix UID if it
      // has one.
      std::string function_name = node.attr().find("f")->second.func().name();
      StatusOr<int> uid = GetSuffixUID(function_name);
      if (uid.ok()) {
        node.mutable_attr()->find("f")->second.mutable_func()->set_name(
            std::string(
                absl::StripSuffix(function_name, std::to_string(*uid))));
      }
      // Erase the "config_proto" attribute which contains device-specific
      // information.
      auto node_config_proto = node.mutable_attr()->find("config_proto");
      if (node_config_proto != node.attr().end()) {
        node_config_proto->second.mutable_s()->erase();
      }
    }
    // Erase the value of string constants, which can vary based on platform.
    if (grappler::IsConstant(node)) {
      if (node.attr().at("dtype").type() == DT_STRING) {
        node.mutable_attr()->find("value")->second.clear_value();
      }
    }
  }
}
}  // namespace

void SimpleDelete(GraphDef& graph_def) {
  // The GraphDef contains two main sections: a list of nodes and the
  // FunctionDefLibrary. Regularization treats these two sections separately.
  RegularizeNodes(&graph_def);
  // TODO(b/240173815): Complete canonicalization of the FunctionDefLibrary.
  // For now, we just completely clear the FunctionDefLibrary.
  graph_def.mutable_library()->Clear();
  graph_def.mutable_versions()->Clear();
}

}  // namespace tensorflow::graph_regularization
