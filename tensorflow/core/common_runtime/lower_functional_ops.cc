/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/lower_functional_ops.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/lower_function_call_op.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"
#include "tensorflow/core/common_runtime/lower_while_op.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

constexpr const char* const kLowerUsingSwitchMergeAttr =
    LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr;
constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
    LowerFunctionalOpsPass::kLowerAsMultiDeviceFunctionAttr;

constexpr const char* const kTpuReplicateAttr = "_tpu_replicate";
constexpr const char* const kXlaClusterAttr = "_xla_compile_id";

// Checks if boolean attribute is defined and it's value is 'true'.
bool CheckBoolAttr(const Node* n, absl::string_view attr_name) {
  bool match;
  Status s = GetNodeAttr(n->attrs(), attr_name, &match);
  return s.ok() && match;
}

// Checks if string attribute is defined and it's not empty.
bool CheckStringAttr(const Node* n, absl::string_view attr_name) {
  string match;
  Status s = GetNodeAttr(n->attrs(), attr_name, &match);
  return s.ok() && !match.empty();
}

bool LowerUsingSwitchMergeIsOn(const Node* n) {
  return CheckBoolAttr(n, kLowerUsingSwitchMergeAttr);
}

bool LowerAsMultiDeviceFunctionIsOn(const Node* n) {
  return CheckBoolAttr(n, kLowerAsMultiDeviceFunctionAttr);
}

bool MarkedForTpuCompilation(const Node* n) {
  return CheckStringAttr(n, kTpuReplicateAttr);
}

bool MarkedForXlaCompilation(const Node* n) {
  return CheckStringAttr(n, kXlaClusterAttr);
}

bool HasArgsOrRetvals(const Graph& g) {
  for (const Node* n : g.op_nodes()) {
    if (n->IsArg() || n->IsRetval()) return true;
  }
  return false;
}

}  // namespace

Status LowerFunctionalOpsPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (options.partition_graphs != nullptr) {
    return errors::Internal(
        "Lowering If/While ops should happen before partitioning.");
  }
  if (options.graph == nullptr) {
    return Status::OK();
  }

  Graph* g = options.graph->get();
  if (g == nullptr) {
    return errors::Internal(
        "Lowering While op requires a graph to be available.");
  }

  FunctionLibraryDefinition* flib_def = options.flib_def;
  if (flib_def == nullptr) {
    return errors::Internal(
        "Lowering If op requires a FunctionLibraryDefinition to be available.");
  }

  // Lower function calls only if it's explicitly enabled in session options.
  const bool lower_function_calls =
      options.session_options && options.session_options->config.graph_options()
                                     .optimizer_options()
                                     .do_function_inlining();

  // If graph is a function instantiation, it will have `_Arg` and `_Retval`
  // nodes for input and output tensors. Otherwise it's unsafe to remove any of
  // the nodes, because they might be later used as fetches.
  //
  // When we do not keep lowered nodes fetchable, we still add a NoOp node to
  // the graph with the same name as lowered node, because it might be used as a
  // control output source, and it's currently not expressed in a graph.
  bool keep_lowered_nodes_fetchable = keep_lowered_nodes_fetchable_.has_value()
                                          ? *keep_lowered_nodes_fetchable_
                                          : !HasArgsOrRetvals(*g);

  // Lower all If and While ops that have the `kLowerUsingSwitchMergeAttr` attr
  // set and inlines all function calls into the graph.
  // We start at `i` = 2 to skip the source and sink nodes.
  // Note that `g->num_node_ids()` may change in the for body if a matching If
  // or While node is lowered. Since new graph nodes are always added to the
  // end of the list of nodes it is ensured that nested If/While nodes will be
  // lowered as well.
  for (int i = 2; i < g->num_node_ids(); ++i) {
    Node* n = g->FindNodeId(i);
    if (n == nullptr) continue;  // deleted node
    if (MarkedForTpuCompilation(n)) continue;
    if (MarkedForXlaCompilation(n)) continue;

    // Always lower function calls produces by lowering If/While nodes.
    if (IsFunctionCall(*flib_def, *n) &&
        (lower_function_calls || LowerAsMultiDeviceFunctionIsOn(n))) {
      TF_RETURN_IF_ERROR(RewriteFunctionCallNode(n, g, *flib_def,
                                                 keep_lowered_nodes_fetchable));
      continue;
    }

    if (LowerUsingSwitchMergeIsOn(n)) {
      if (n->type_string() == "If") {
        TF_RETURN_IF_ERROR(
            RewriteIfNode(n, g, *flib_def, keep_lowered_nodes_fetchable));
      } else if (n->type_string() == "While") {
        TF_RETURN_IF_ERROR(
            RewriteWhileNode(n, g, *flib_def, keep_lowered_nodes_fetchable));
      } else {
        return errors::Internal(
            "Node ", FormatNodeForError(*n), " of type ", n->type_string(),
            " has '", LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr,
            "' attr set but it does not support lowering.\n");
      }
    }
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      LowerFunctionalOpsPass);

}  // namespace tensorflow
