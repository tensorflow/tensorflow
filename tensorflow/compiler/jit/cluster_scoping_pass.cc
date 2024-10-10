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

#include "tensorflow/compiler/jit/cluster_scoping_pass.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {
namespace {

class ClusterScopingPassImpl {
 public:
  ClusterScopingPassImpl(Graph* graph,
                         OptimizerOptions::GlobalJitLevel global_jit_level)
      : graph_(graph),
        global_jit_level_(global_jit_level),
        unique_scope_id_(0) {}

  absl::Status Run();

 private:
  absl::Status ScopingForPipelineStages();

  size_t GetUniqueScopeId() { return unique_scope_id_++; }

  void AddScopeToAllTransitivePredecessors(Node* start);

  void AddScopeToAllTransitiveSuccessors(Node* start);

 private:
  Graph* graph_;
  OptimizerOptions::GlobalJitLevel global_jit_level_;
  size_t unique_scope_id_;
};

std::optional<string> GetXlaInternalScope(Node* node) {
  string scope;
  if (GetNodeAttr(node->attrs(), kXlaInternalScopeAttr, &scope).ok()) {
    return scope;
  }

  return std::nullopt;
}

void SetXlaInternalScope(Node* node, StringPiece scope) {
  node->AddAttr(kXlaInternalScopeAttr, scope);
}

// NB! We append a new scope as suffix to the _XlaInternalScope attribute
// instead of overriding the old value.  In other words, appending scope B to
// scope A creates the conjunction of the scopes A and B (i.e, A & B) and,
// in effect, the node gets both the old and new scopes.  As a unique scope
// disallows a node being merged with nodes in other scopes, the scope
// conjunction preserves the semantic of the old scope (i.e., the node still
// cannot be merged with the previously incompatible nodes.)
//
// For example, the below case should be rare in practice but can serve for the
// purpose of discussion.  After adding scopes for both Stage and Unstage,
// Node_Y will receive both scopes "unstage" and "stage", while Node_X receives
// only scope "stage".  The semantic of scope "unstage" is preserved although
// scope "stage" is later appended.  As a result, Node_X and Node_Y will be put
// into different clusters.
//
//                Unstage -> Node_Y (scope "unstage & stage")
//                              |
//                              V
//  Node_X (scope "stage") -> Stage
//
void AddOrAppendXlaInternalScope(Node* node, absl::string_view suffix) {
  string updated_scope;
  std::optional<string> cur_scope = GetXlaInternalScope(node);
  if (cur_scope == std::nullopt) {
    updated_scope = std::string(suffix);
  } else {
    updated_scope = absl::StrCat(cur_scope.value(), "&", suffix);
  }
  SetXlaInternalScope(node, updated_scope);
}

void ClusterScopingPassImpl::AddScopeToAllTransitivePredecessors(Node* start) {
  const string unique_suffix = absl::StrCat("_", GetUniqueScopeId());

  std::vector<Node*> starts;
  starts.push_back(start);
  auto enter = [&](Node* n) { AddOrAppendXlaInternalScope(n, unique_suffix); };
  ReverseDFSFrom(*graph_, starts, enter, /*leave=*/nullptr,
                 /*stable_comparator=*/NodeComparatorName());
}

void ClusterScopingPassImpl::AddScopeToAllTransitiveSuccessors(Node* start) {
  const string unique_suffix = absl::StrCat("_", GetUniqueScopeId());

  std::vector<Node*> starts;
  starts.push_back(start);
  auto enter = [&](Node* n) { AddOrAppendXlaInternalScope(n, unique_suffix); };
  DFSFrom(*graph_, starts, enter, /*leave=*/nullptr,
          /*stable_comparator=*/NodeComparatorName(),
          // Do not filter any edges to better capture the semantics of
          // transitive closure of successors.  We may revisit this when
          // we see more cases needing cluster scoping in the future.
          /*edge_filter=*/nullptr);
}

// This preserves the parallelism between pipeline stages.  For example, below
// is a typical pattern of input pipelining in Tensorflow and this heuristic
// ensures Node_X and Node_Y are put into different clusters.  Without the
// heuristic, they may be put into the same cluster and it can introduce
// artificial dependencies and incur great performance loss.  In this example,
// Node_Y becomes dependent on IteratorGetNext and the latencies add up if
// Node_X and Node_Y are in the same cluster.
//
// IteratorGetNext -> Node_X -> Stage
//
// Unstage -> Node_Y
//
absl::Status ClusterScopingPassImpl::ScopingForPipelineStages() {
  for (Node* n : graph_->nodes()) {
    DCHECK(n);
    if (n->type_string() == "Unstage") {
      AddScopeToAllTransitiveSuccessors(n);
    }
    if (n->type_string() == "Stage") {
      AddScopeToAllTransitivePredecessors(n);
    }
  }

  return absl::OkStatus();
}

absl::Status ClusterScopingPassImpl::Run() {
  if (global_jit_level_ == OptimizerOptions::OFF) {
    return absl::OkStatus();
  }

  return ScopingForPipelineStages();
}
}  // namespace

absl::Status ClusterScopingPass::Run(
    const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();

  return ClusterScopingPassImpl{graph, GetGlobalJitLevelForGraph(options)}
      .Run();
}
}  // namespace tensorflow
