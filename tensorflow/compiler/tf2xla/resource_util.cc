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
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

using stream_executor::port::StatusOr;

const char kIdentityNOp[] = "IdentityN";
const char kIfOp[] = "If";
const char kWhileOp[] = "While";
const char kArgOp[] = "_Arg";
const char kRetvalOp[] = "_Retval";

const int kMaxCallDepth = 100;

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

  // Reaching here means e is not coming from a pass through node, return empty
  // vector to indicate we can no longer trace back.
  return nullptr;
}

// TODO(ycao): Add this as Tensorflow Node method.
StatusOr<absl::InlinedVector<const Edge*, 1>> OutputEdgesByIndex(const Node* n,
                                                                 int idx) {
  absl::InlinedVector<const Edge*, 1> res;
  if (idx >= n->num_outputs()) {
    return errors::InvalidArgument("Invalid out_edge index: ", idx, ", Node ",
                                   n->name(), " only has ", n->num_outputs(),
                                   " outputs.");
  }

  for (const Edge* o : n->out_edges()) {
    if (o->src_output() == idx) res.emplace_back(o);
  }
  return res;
}

bool IsStackOrTensorArraySource(const Node* n) {
  const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(n->type_string());

  if (!op_info) return false;
  if (op_info->resource_kind() != XlaResourceKind::kStack &&
      op_info->resource_kind() != XlaResourceKind::kTensorArray)
    return false;
  return n->num_outputs() > 0 && n->output_type(0) == DataType::DT_RESOURCE;
}

Status AnalyzeResourceUsage(
    const Graph* graph, FunctionLibraryRuntime* lib_runtime,
    const absl::optional<std::string>& function_name, const int call_depth,
    const absl::flat_hash_set<int>& resource_arg_indices,
    absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                        absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>*
        source_to_path) {
  source_to_path->clear();

  std::vector<Node*> reverse_post_order;
  GetReversePostOrder(*graph, &reverse_post_order, NodeComparatorName{});

  // user_to_source maps from an edge carrying a Stack or TensorArray resource
  // to the node that created this resource.
  absl::flat_hash_map<const Edge*, ResourceUsageAnalysis::NodeInfo>
      user_to_source;
  for (const Node* n : reverse_post_order) {
    if (IsControlFlowV1Node(n)) {
      return errors::InvalidArgument(
          "AnalyzeResourceUsage does not support control flow v1 node: ",
          n->DebugString());
    }

    // TODO(ycao): Support pass-through functional while/if nodes.
    if (n->type_string() == kIfOp || n->type_string() == kWhileOp) {
      return errors::InvalidArgument(
          "AnalyzeResourceUsage does not yet support control flow v2 "
          "node: ",
          n->DebugString());
    }

    // Record a resource source edge.
    if (IsStackOrTensorArraySource(n)) {
      ResourceUsageAnalysis::NodeInfo src_node_info(function_name, n->name(),
                                                    n->type_string());
      for (const Edge* o : n->out_edges()) {
        if (o->IsControlEdge()) continue;
        if (o->dst()->input_type(o->dst_input()) != DataType::DT_RESOURCE) {
          continue;
        }
        user_to_source[o] = src_node_info;
      }
      continue;
    }

    // Arguments that are listed in resource_arg_indices are also considered as
    // resource sources.
    if (n->IsArg()) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      if (!resource_arg_indices.contains(index)) continue;

      TF_RET_CHECK(function_name.has_value())
          << "ResourceUsageAnalysis does not support analyzing _Arg nodes "
             "carrying Stack/TensorArray resource in given graph unless they "
             "are in function calls.";

      const ResourceUsageAnalysis::NodeInfo src_node_info(
          function_name, n->name(), n->type_string());

      for (const Edge* o : n->out_edges()) {
        if (o->IsControlEdge()) continue;
        if (o->dst()->input_type(o->dst_input()) != DataType::DT_RESOURCE) {
          continue;
        }
        user_to_source[o] = src_node_info;
      }
      continue;
    }

    if (IsFunctionCall(*lib_runtime->GetFunctionLibraryDefinition(), *n)) {
      if (call_depth > kMaxCallDepth) {
        return errors::InvalidArgument(
            "Function call stack in given graph is too deep, last function ",
            "name is: ", function_name.value());
      }
      // resource_arg_indices_for_call contains all indices of the input
      // arguments that carry Stack/TensorArray resource handles.
      absl::flat_hash_set<int> resource_arg_indices_for_call;
      for (const Edge* e : n->in_edges()) {
        if (!user_to_source.contains(e)) continue;
        resource_arg_indices_for_call.emplace(e->dst_input());
      }

      absl::string_view called_function_name = n->type_string();
      FunctionLibraryRuntime::Handle handle;
      TF_RETURN_IF_ERROR(
          InstantiateFunctionCall(n->def(), lib_runtime, &handle));
      auto release_handle_on_return = gtl::MakeCleanup(
          [&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });
      const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);

      // Recursively analyze called function for resource sources and users.
      absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                          absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>
          called_function_source_to_path;
      TF_RETURN_IF_ERROR(AnalyzeResourceUsage(
          fbody->graph, lib_runtime,
          absl::optional<std::string>(called_function_name), call_depth + 1,
          resource_arg_indices_for_call, &called_function_source_to_path));

      std::unordered_map<std::string, Node*> node_name_index =
          fbody->graph->BuildNodeNameIndex();

      for (auto it : called_function_source_to_path) {
        ResourceUsageAnalysis::NodeInfo src_node_info = it.first;

        // If source is an _Arg, then the true source is actually corresponding
        // edge that feeds into function call node with the same index.
        if (src_node_info.op_ == kArgOp) {
          const Node* arg_src = node_name_index[src_node_info.node_name_];
          int index;
          TF_RETURN_IF_ERROR(GetNodeAttr(arg_src->attrs(), "index", &index));

          const Edge* e;
          TF_RETURN_IF_ERROR(n->input_edge(index, &e));
          const Node* true_src = e->src();
          src_node_info.function_name_ = function_name;
          src_node_info.node_name_ = true_src->name();
          src_node_info.op_ = true_src->type_string();
        }

        for (const auto& dst_node_info : it.second) {
          // If user is an _Retval, then the true user is actually corresponding
          // edge of that _Retval.
          if (dst_node_info.op_ == kRetvalOp) {
            const Node* ret_user = node_name_index[dst_node_info.node_name_];
            int index;
            TF_RETURN_IF_ERROR(GetNodeAttr(ret_user->attrs(), "index", &index));

            absl::InlinedVector<const Edge*, 1> outs;
            TF_ASSIGN_OR_RETURN(outs, OutputEdgesByIndex(n, index));
            for (const Edge* o : outs) user_to_source[o] = src_node_info;
          } else {
            (*source_to_path)[src_node_info].emplace(dst_node_info);
          }
        }
      }
      continue;
    }

    for (const Edge* o : n->out_edges()) {
      if (o->IsControlEdge()) continue;
      TF_ASSIGN_OR_RETURN(const Edge* e, WalkBackPassThroughEdge(o));
      if (!e || !user_to_source.contains(e)) continue;
      user_to_source.emplace(std::make_pair(o, user_to_source[e]));
    }
  }

  for (auto it : user_to_source) {
    ResourceUsageAnalysis::NodeInfo dst_node_info(
        function_name, it.first->dst()->name(), it.first->dst()->type_string());
    (*source_to_path)[it.second].emplace(dst_node_info);
  }

  return Status::OK();
}

}  // anonymous namespace

/*Static*/ Status ResourceUsageAnalysis::Analyze(
    const Graph* graph, FunctionLibraryRuntime* lib_runtime,
    absl::flat_hash_map<NodeInfo, absl::flat_hash_set<NodeInfo>>*
        source_to_path) {
  return AnalyzeResourceUsage(
      graph, lib_runtime, /*function_name=*/{}, /*call_depth=*/0,
      /*resource_arg_indices=*/absl::flat_hash_set<int>(), source_to_path);
}

}  // namespace tensorflow
