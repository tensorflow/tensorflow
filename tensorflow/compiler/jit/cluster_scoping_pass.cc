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

  Status Run();

 private:
  Status ScopingForPipelineStages();

  size_t GetUniqueScopeId() { return unique_scope_id_++; }

  void AddScopeToAllPredecessors(Node* start);

  void AddScopeToAllSuccessors(Node* start);

 private:
  Graph* graph_;
  OptimizerOptions::GlobalJitLevel global_jit_level_;
  size_t unique_scope_id_;
};

absl::optional<string> GetXlaScope(Node* node) {
  string scope;
  if (GetNodeAttr(node->attrs(), kXlaScopeAttr, &scope).ok()) {
    return scope;
  }

  return absl::nullopt;
}

void SetXlaScope(Node* node, StringPiece scope) {
  node->AddAttr(kXlaScopeAttr, scope);
}

// NB! We append new scope as suffix to the XlaScope attribute instead of
// overriding the old value.  In this way, we respect the original scopes.
// In other words, appending X to Y creates the conjunction of the scopes X
// and Y (i.e, X & Y in effect).
void AddOrAppendScope(Node* node, absl::string_view suffix) {
  string updated_scope;
  absl::optional<string> cur_scope = GetXlaScope(node);
  if (cur_scope == absl::nullopt) {
    updated_scope = std::string(suffix);
  } else {
    updated_scope = absl::StrCat(cur_scope.value(), "&", suffix);
  }
  SetXlaScope(node, updated_scope);
}

void ClusterScopingPassImpl::AddScopeToAllPredecessors(Node* start) {
  const string unique_suffix = absl::StrCat("_", GetUniqueScopeId());

  std::vector<Node*> starts;
  starts.push_back(start);
  auto enter = [&](Node* n) { AddOrAppendScope(n, unique_suffix); };
  ReverseDFSFrom(*graph_, starts, enter, /*leave=*/nullptr,
                 /*stable_comparator=*/NodeComparatorName());
}

void ClusterScopingPassImpl::AddScopeToAllSuccessors(Node* start) {
  const string unique_suffix = absl::StrCat("_", GetUniqueScopeId());

  std::vector<Node*> starts;
  starts.push_back(start);
  auto enter = [&](Node* n) { AddOrAppendScope(n, unique_suffix); };
  auto not_back_edge = [](const Edge& edge) -> bool {
    return !edge.src()->IsNextIteration();
  };
  DFSFrom(*graph_, starts, enter, /*leave=*/nullptr,
          /*stable_comparator=*/NodeComparatorName(),
          /*edge_filter=*/not_back_edge);
}

Status ClusterScopingPassImpl::ScopingForPipelineStages() {
  for (Node* n : graph_->nodes()) {
    DCHECK(n);
    if (n->type_string() == "Unstage") {
      AddScopeToAllSuccessors(n);
    }
    if (n->type_string() == "Stage") {
      AddScopeToAllPredecessors(n);
    }
  }

  return Status::OK();
}

Status ClusterScopingPassImpl::Run() {
  if (global_jit_level_ == OptimizerOptions::OFF) {
    return Status::OK();
  }

  // This preserves the parallelism between pipeline stages.  For example,
  // below is a typical pattern of input pipelining in Tensorflow and this
  // heuristic ensures Node_X and Node_Y are put into different clusters.
  // Without the heuristic, they may be put into the same cluster and it
  // can introduce artificial dependencies and incur great performance loss.
  // In this example, Node_Y becomes dependent on IteratorGetNext and the
  // latencies add up.
  //
  // IteratorGetNext -> Node_X -> Stage
  //
  // Unstage -> Node_Y
  //
  return ScopingForPipelineStages();
}
}  // namespace

Status ClusterScopingPass::Run(const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();

  return ClusterScopingPassImpl{graph, GetGlobalJitLevelForGraph(options)}
      .Run();
}
}  // namespace tensorflow
