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

#include "tensorflow/compiler/tf2xla/resource_util.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

using stream_executor::port::StatusOr;

const char kIdentityNOp[] = "IdentityN";
const char kIfOp[] = "If";
const char kWhileOp[] = "While";

bool IsControlFlowV1Node(const Node* n) {
  return (n->IsEnter() || n->IsExit() || n->IsSwitch() || n->IsMerge() ||
          n->IsNextIteration());
}

// Given an output edge, find the corresponding input edge if given edge is
// coming from a pass-through node. Otherwise, return nullptr.
StatusOr<const Edge*> WalkBackPassThroughEdge(const Edge* e) {
  const Node* n = e->src();

  if (n->IsIdentity()) {
    const Edge* ret;
    TF_RETURN_IF_ERROR(n->input_edge(0, &ret));
    return ret;
  }

  if (n->type_string() == kIdentityNOp) {
    const Edge* ret;
    TF_RETURN_IF_ERROR(n->input_edge(e->src_output(), &ret));
    return ret;
  }

  // TODO(ycao): Support pass-through function calls and functional while/if
  // nodes.

  // Reaching here means e is not coming from a pass through node, return empty
  // vector to indicate we can no longer trace back.
  return nullptr;
}

bool IsStackOrTensorArraySource(const Node* n) {
  const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(n->type_string());

  if (!op_info) return false;
  if (op_info->resource_kind() != XlaResourceKind::kStack &&
      op_info->resource_kind() != XlaResourceKind::kTensorArray)
    return false;
  return n->num_outputs() > 0 && n->output_type(0) == DataType::DT_RESOURCE;
}

}  // anonymous namespace

Status AnalyzeResourceOpSourcePath(
    const Graph* graph,
    absl::flat_hash_map<const Node*, absl::flat_hash_set<const Node*>>*
        sources_paths) {
  sources_paths->clear();

  std::vector<Node*> reverse_post_order;
  GetReversePostOrder(*graph, &reverse_post_order, NodeComparatorName{});

  // user_to_source maps from an edge carrying a Stack or TensorArray resource
  // to the node that created this resource.
  absl::flat_hash_map<const Edge*, const Node*> user_to_source;
  for (const Node* n : reverse_post_order) {
    if (IsControlFlowV1Node(n)) {
      return errors::InvalidArgument(
          "AnalyzeResourceOpSourcePath does not support control flow v1 node: ",
          n->DebugString());
    }

    if (n->type_string() == kIfOp || n->type_string() == kWhileOp) {
      return errors::InvalidArgument(
          "AnalyzeResourceOpSourcePath does not yet support control flow v2 "
          "node: ",
          n->DebugString());
    }

    // Record a resource source edge.
    if (IsStackOrTensorArraySource(n)) {
      for (const Edge* o : n->out_edges()) {
        if (o->IsControlEdge()) continue;
        if (o->dst()->input_type(o->dst_input()) != DataType::DT_RESOURCE)
          continue;
        user_to_source[o] = n;
      }
      continue;
    }

    for (const Edge* o : n->out_edges()) {
      if (o->IsControlEdge()) continue;
      TF_ASSIGN_OR_RETURN(const Edge* e, WalkBackPassThroughEdge(o));
      if (!e || !user_to_source.contains(e)) continue;
      user_to_source[o] = user_to_source[e];
    }
  }

  for (auto it : user_to_source) {
    (*sources_paths)[it.second].emplace(it.first->dst());
  }

  return Status::OK();
}

}  // namespace tensorflow
