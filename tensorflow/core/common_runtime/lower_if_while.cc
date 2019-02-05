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

#include "tensorflow/core/common_runtime/lower_if_while.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"
#include "tensorflow/core/common_runtime/lower_while_op.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

#if defined(_MSC_VER)
constexpr char* LowerIfWhilePass::kLowerUsingSwitchMergeAttr;
#else
constexpr char LowerIfWhilePass::kLowerUsingSwitchMergeAttr[];
#endif

namespace {

bool HasLoweringAttr(const AttrSlice& attrs) {
  bool match;
  Status s =
      GetNodeAttr(attrs, LowerIfWhilePass::kLowerUsingSwitchMergeAttr, &match);
  return s.ok() && match;
}

}  // namespace

Status LowerIfWhilePass::Run(const GraphOptimizationPassOptions& options) {
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

  FunctionLibraryDefinition* flib = options.flib_def;
  if (flib == nullptr) {
    return errors::Internal(
        "Lowering If op requires a FunctionLibraryDefinition to be available.");
  }

  // Lower all If and While ops that have the `kLowerUsingSwitchMergeAttr` attr
  // set.
  // We start at `i` = 2 to skip the source and sink nodes.
  // Note that `g->num_node_ids()` may change in the for body if a matching If
  // or While node is lowered. Since new graph nodes are always added to the
  // end of the list of nodes it is ensured that nested If/While nodes will be
  // lowered as well.
  for (int i = 2; i < g->num_node_ids(); ++i) {
    Node* n = g->FindNodeId(i);
    if (n == nullptr) continue;  // deleted node
    if (HasLoweringAttr(n->attrs())) {
      if (n->type_string() == "If") {
        TF_RETURN_IF_ERROR(RewriteIfNode(n, g, *flib));
      } else if (n->type_string() == "While") {
        TF_RETURN_IF_ERROR(RewriteWhileNode(n, g, *flib));
      } else {
        return errors::Internal(
            "Node ", FormatNodeForError(*n), " of type ", n->type_string(),
            " has '", LowerIfWhilePass::kLowerUsingSwitchMergeAttr,
            "' attr set but it does not support lowering.\n");
      }
    }
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      LowerIfWhilePass);

}  // namespace tensorflow
