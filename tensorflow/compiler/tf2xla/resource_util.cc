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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

using tsl::StatusOr;

const char kIdentityNOp[] = "IdentityN";
const char kIfOp[] = "If";
const char kWhileOp[] = "While";
const char kArgOp[] = "_Arg";
const char kRetvalOp[] = "_Retval";

const int kMaxCallDepth = 100;

Status AnalyzeResourceUsage(
    const Graph* graph, const std::optional<std::string>& function_name,
    const int call_depth, const absl::flat_hash_set<int>& resource_arg_indices,
    FunctionLibraryRuntime* lib_runtime,
    absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                        absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>*
        source_to_path);

bool IsControlFlowV1Node(const Node* n) {
  return (n->IsEnter() || n->IsExit() || n->IsSwitch() || n->IsMerge() ||
          n->IsNextIteration());
}

// TODO(ycao): Add this as Tensorflow Node method.
StatusOr<absl::InlinedVector<const Edge*, 1>> OutputEdgesByIndex(const Node& n,
                                                                 int idx) {
  absl::InlinedVector<const Edge*, 1> res;
  if (idx >= n.num_outputs()) {
    return errors::InvalidArgument("Invalid out_edge index: ", idx, ", Node ",
                                   n.name(), " only has ", n.num_outputs(),
                                   " outputs.");
  }

  for (const Edge* o : n.out_edges()) {
    if (o->src_output() == idx) res.emplace_back(o);
  }
  return res;
}

bool IsStackOrTensorArraySource(const Node& n) {
  const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(n.type_string());

  if (!op_info) return false;
  if (op_info->resource_kind() != XlaResourceKind::kStack &&
      op_info->resource_kind() != XlaResourceKind::kTensorArray)
    return false;
  return n.num_outputs() > 0 && n.output_type(0) == DataType::DT_RESOURCE;
}

void PropagateFromStackOrTensorArraySourceOp(
    const Node& n, const std::optional<std::string>& function_name,
    absl::flat_hash_map<const Edge*, ResourceUsageAnalysis::NodeInfo>*
        user_to_source) {
  ResourceUsageAnalysis::NodeInfo src_node_info(function_name, n.name(),
                                                n.type_string());
  for (const Edge* o : n.out_edges()) {
    if (o->IsControlEdge()) continue;
    if (o->dst()->input_type(o->dst_input()) != DataType::DT_RESOURCE) {
      continue;
    }
    (*user_to_source)[o] = src_node_info;
  }
}

Status PropagateFromArgOp(
    const Node& n, const std::optional<std::string>& function_name,
    const absl::flat_hash_set<int>& resource_arg_indices,
    absl::flat_hash_map<const Edge*, ResourceUsageAnalysis::NodeInfo>*
        user_to_source) {
  TF_RET_CHECK(n.type_string() == kArgOp);

  int index;
  TF_RETURN_IF_ERROR(GetNodeAttr(n.attrs(), "index", &index));
  if (!resource_arg_indices.contains(index)) return OkStatus();

  TF_RET_CHECK(function_name.has_value())
      << "ResourceUsageAnalysis does not support analyzing _Arg nodes "
         "carrying Stack/TensorArray resource in given graph unless they "
         "are in function calls.";

  const ResourceUsageAnalysis::NodeInfo src_node_info(function_name, n.name(),
                                                      n.type_string());

  for (const Edge* o : n.out_edges()) {
    if (o->IsControlEdge()) continue;
    if (o->dst()->input_type(o->dst_input()) != DataType::DT_RESOURCE) {
      continue;
    }
    (*user_to_source)[o] = src_node_info;
  }

  return OkStatus();
}

Status UpdateResourceUsageFromFunctionBodyAnalysis(
    const Node& call_node,
    const std::optional<absl::string_view>& caller_function_name,
    const FunctionBody& fbody,
    const absl::flat_hash_map<
        ResourceUsageAnalysis::NodeInfo,
        absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>&
        called_function_source_to_path,
    absl::flat_hash_map<const Edge*, ResourceUsageAnalysis::NodeInfo>*
        user_to_source,
    absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                        absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>*
        caller_source_to_path) {
  std::unordered_map<std::string, Node*> node_name_index =
      fbody.graph->BuildNodeNameIndex();
  for (const auto& it : called_function_source_to_path) {
    ResourceUsageAnalysis::NodeInfo src_node_info = it.first;

    // If source is an _Arg, then the true source is actually corresponding
    // edge that feeds into function call node with the same index.
    if (src_node_info.op_ == kArgOp) {
      const Node* arg_src = node_name_index[src_node_info.node_name_];
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(arg_src->attrs(), "index", &index));

      const Edge* e;
      // TODO(ycao): Allow overriding input_edge to _Arg index mapping. This is
      // needed for cond function of while nodes.
      TF_RETURN_IF_ERROR(call_node.input_edge(index, &e));
      src_node_info = (*user_to_source)[e];
    }

    for (const auto& dst_node_info : it.second) {
      // If user is an _Retval, then the true user is actually corresponding
      // edge of that _Retval.
      if (dst_node_info.op_ == kRetvalOp) {
        const Node* ret_user = node_name_index[dst_node_info.node_name_];
        int index;
        TF_RETURN_IF_ERROR(GetNodeAttr(ret_user->attrs(), "index", &index));

        absl::InlinedVector<const Edge*, 1> outs;
        // TODO(ycao): Allow overriding _Retval index to call node output edge
        // mapping. This is needed for cond function of while nodes.
        TF_ASSIGN_OR_RETURN(outs, OutputEdgesByIndex(call_node, index));
        for (const Edge* o : outs) (*user_to_source)[o] = src_node_info;
      } else {
        (*caller_source_to_path)[src_node_info].emplace(dst_node_info);
      }
    }
  }

  return OkStatus();
}

Status PropagateThroughCallOp(
    const Node& n, const std::optional<std::string>& function_name,
    const int call_depth, FunctionLibraryRuntime* lib_runtime,
    absl::flat_hash_map<const Edge*, ResourceUsageAnalysis::NodeInfo>*
        user_to_source,
    absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                        absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>*
        source_to_path) {
  if (call_depth > kMaxCallDepth) {
    return errors::InvalidArgument(
        "Function call stack in given graph is too deep, last function ",
        "name is: ", function_name.value());
  }
  // resource_arg_indices contains all indices of the input
  // arguments that carry Stack/TensorArray resource handles.
  absl::flat_hash_set<int> resource_arg_indices;
  for (const Edge* e : n.in_edges()) {
    if (user_to_source->contains(e)) {
      resource_arg_indices.emplace(e->dst_input());
    }
  }

  // Instantiate associated function to get function body.
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(InstantiateFunctionCall(n.def(), lib_runtime, &handle));
  auto release_handle_on_return = gtl::MakeCleanup(
      [&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });
  const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);

  // Recursively analyze called function for resource sources and users.
  absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                      absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>
      called_function_source_to_path;
  TF_RETURN_IF_ERROR(AnalyzeResourceUsage(
      fbody->graph, n.type_string(), call_depth + 1, resource_arg_indices,
      lib_runtime, &called_function_source_to_path));

  TF_RETURN_IF_ERROR(UpdateResourceUsageFromFunctionBodyAnalysis(
      n, function_name, *fbody, called_function_source_to_path, user_to_source,
      source_to_path));
  return OkStatus();
}

// Analyzes pass through values for Identity and IdentityN ops.
Status PropagateThroughIdentityOp(
    const Node& n,
    absl::flat_hash_map<const Edge*, ResourceUsageAnalysis::NodeInfo>*
        user_to_source) {
  TF_RET_CHECK(n.IsIdentity() || n.type_string() == kIdentityNOp);
  if (n.IsIdentity()) {
    for (const Edge* o : n.out_edges()) {
      if (o->IsControlEdge()) continue;
      const Edge* in;
      TF_RETURN_IF_ERROR(n.input_edge(0, &in));
      if (!user_to_source->contains(in)) continue;
      user_to_source->emplace(std::make_pair(o, (*user_to_source)[in]));
    }
  } else {
    for (const Edge* o : n.out_edges()) {
      if (o->IsControlEdge()) continue;
      const Edge* in;
      TF_RETURN_IF_ERROR(n.input_edge(o->src_output(), &in));
      if (!user_to_source->contains(in)) continue;
      user_to_source->emplace(std::make_pair(o, (*user_to_source)[in]));
    }
  }

  return OkStatus();
}

Status AnalyzeResourceUsage(
    const Graph* graph, const std::optional<std::string>& function_name,
    const int call_depth, const absl::flat_hash_set<int>& resource_arg_indices,
    FunctionLibraryRuntime* lib_runtime,
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
    if (IsStackOrTensorArraySource(*n)) {
      PropagateFromStackOrTensorArraySourceOp(*n, function_name,
                                              &user_to_source);
      continue;
    }

    // Arguments that are listed in resource_arg_indices are also considered as
    // resource sources.
    if (n->IsArg()) {
      TF_RETURN_IF_ERROR(PropagateFromArgOp(
          *n, function_name, resource_arg_indices, &user_to_source));
      continue;
    }

    // Recursively analyze function call ops.
    if (IsFunctionCall(*lib_runtime->GetFunctionLibraryDefinition(), *n)) {
      TF_RETURN_IF_ERROR(PropagateThroughCallOp(*n, function_name, call_depth,
                                                lib_runtime, &user_to_source,
                                                source_to_path));
      continue;
    }

    if (n->IsIdentity() || n->type_string() == kIdentityNOp) {
      TF_RETURN_IF_ERROR(PropagateThroughIdentityOp(*n, &user_to_source));
    }
  }

  for (const auto& it : user_to_source) {
    (*source_to_path)[it.second].emplace(function_name, it.first->dst()->name(),
                                         it.first->dst()->type_string());
  }

  return OkStatus();
}

}  // anonymous namespace

/*Static*/ Status ResourceUsageAnalysis::Analyze(
    const Graph* graph, FunctionLibraryRuntime* lib_runtime,
    absl::flat_hash_map<NodeInfo, absl::flat_hash_set<NodeInfo>>*
        source_to_path) {
  return AnalyzeResourceUsage(
      graph, /*function_name=*/{}, /*call_depth=*/0,
      /*resource_arg_indices=*/absl::flat_hash_set<int>(), lib_runtime,
      source_to_path);
}

}  // namespace tensorflow
