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

#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"

#include "llvm/ADT/StringSet.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace {

constexpr char kTpuReplicateAttr[] = "_tpu_replicate";

// Returns the ops that should use node name if shared_name is empty.
const llvm::StringSet<>& GetOpsUsingNodeName() {
  static auto* const ops =
      new llvm::StringSet<>({"VariableV2", "Variable", "BatchFunction"});
  return *ops;
}

// Returns the set of ops that we want to generate shared_names for them if
// empty.
const llvm::StringSet<>& GetSharedNameGenerationCompatibleOps() {
  return GetOpsUsingNodeName();
}

}  // namespace

Status GenerateResourceSharedNameIfEmpty(
    GraphDef& gdef, const OpRegistryInterface* default_registry) {
  auto is_resource_op_with_empty_shared_name = [](const NodeDef& node_def,
                                                  const OpDef& op_def) {
    if (!GetSharedNameGenerationCompatibleOps().contains(op_def.name())) {
      // If this op is not in the allowlist, then it is likely a custom op.
      // Currently for these ops, we are relying on its "use_node_name_sharing"
      // to decide whether it is valid to generate shared_names. If the OpDef
      // has "use_node_name_sharing" field, then it is valid to use node names
      // as shared names.
      if (!std::any_of(op_def.attr().begin(), op_def.attr().end(),
                       [](const auto& attr_def) {
                         return attr_def.name() == "use_node_name_sharing" &&
                                attr_def.type() == "bool";
                       }))
        return false;
    }

    if (!std::any_of(op_def.attr().begin(), op_def.attr().end(),
                     [](const auto& attr_def) {
                       return attr_def.name() == "shared_name" &&
                              attr_def.type() == "string";
                     }))
      return false;

    auto iter = node_def.attr().find("shared_name");
    if (iter == node_def.attr().end()) return true;
    return iter->second.s().empty();
  };

  FunctionDefLibrary* library = gdef.mutable_library();
  auto flib_def = library ? std::make_unique<FunctionLibraryDefinition>(
                                default_registry, *library)
                          : std::make_unique<FunctionLibraryDefinition>(
                                default_registry, FunctionDefLibrary());

  if (library) {
    // Upgrade nodes in the functions.
    for (FunctionDef& fdef : *library->mutable_function()) {
      auto func_name = fdef.signature().name();
      for (auto& node_def : *fdef.mutable_node_def()) {
        const OpDef* op_def = nullptr;
        // With lazy loading, some functions might not be executed, thus we skip
        // the node if the op is not registered.
        if (flib_def->LookUpOpDef(node_def.op(), &op_def).ok() &&
            is_resource_op_with_empty_shared_name(node_def, *op_def)) {
          // TODO(b/197144710): improve the shared_name attr, each op may use
          // the shared_name differently.
          if (GetOpsUsingNodeName().contains(op_def->name())) {
            // Use the node name for such ops as the shared_name according to
            // the document of variable ops.
            (*node_def.mutable_attr())["shared_name"].set_s(node_def.name());
          } else {
            // Use the concat of function name and node name for such ops in a
            // function as the shared_name. "@" is used as the separator because
            // it is not allowed in the function name or the node name.
            (*node_def.mutable_attr())["shared_name"].set_s(
                absl::StrCat(node_def.name(), "@", func_name));
          }
        }
      }
    }
  }

  // Upgrade nodes in the GraphDef.
  for (auto& node_def : *gdef.mutable_node()) {
    const OpDef* op_def = nullptr;
    TF_RETURN_IF_ERROR(flib_def->LookUpOpDef(node_def.op(), &op_def));
    // TODO(b/197144710): improve the shared_name attr, each op may use the
    // shared_name differently.
    if (is_resource_op_with_empty_shared_name(node_def, *op_def)) {
      (*node_def.mutable_attr())["shared_name"].set_s(node_def.name());
    }
  }

  return tensorflow::Status::OK();
}

Status UpgradeLegacyGraph(Graph* graph, FunctionLibraryDefinition* flib_def,
                          bool restrict_functionalization_to_tpu_nodes) {
  // If `restrict_functionalization_to_tpu_nodes` is true let filter function
  // return true for `_tpu_replicate` nodes, otherwise don't set filter.
  NodeFilter node_filter =
      restrict_functionalization_to_tpu_nodes
          ? [](const Node* n) { return n->attrs().Find(kTpuReplicateAttr); }
          : NodeFilter{};
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      FunctionalizeControlFlow(graph, flib_def, node_filter,
                               /*include_functions=*/true),
      "Failed to functionalize Control Flow V1 ops. Consider using Control "
      "Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/"
      "compat/v1/enable_control_flow_v2.");
  return Status::OK();
}

}  // namespace tensorflow
