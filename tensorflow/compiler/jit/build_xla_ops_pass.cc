/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/build_xla_ops_pass.h"
#include "absl/algorithm/container.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_jit_ops.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {
void MoveOutgoingEdges(Graph* g, Node* old_node, Node* new_node) {
  std::vector<const Edge*> out_edges(old_node->out_edges().begin(),
                                     old_node->out_edges().end());
  for (const Edge* edge : out_edges) {
    // TODO(sanjoy): This does not update NodeDef inputs.  To be able to update
    // NodeDef inputs we first need to fix encapsulate_subgraphs_pass to fix up
    // the NodeDef inputs to the function call nodes.
    g->AddEdge(new_node, edge->src_output(), edge->dst(), edge->dst_input());
    g->RemoveEdge(edge);
  }
}

struct XlaClusterInfo {
  std::vector<Output> constant_inputs;
  std::vector<Output> non_constant_inputs;
  std::vector<Output> resource_inputs;
  NameAttrList function;
};

Output IncomingEdgeAsOutput(const Edge* e) {
  return Output(e->src(), e->src_output());
}

Status GetXlaClusterInfo(Node* n, XlaClusterInfo* result) {
  int num_constant_inputs, num_resource_inputs;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaNumConstantArgsAttr, &num_constant_inputs));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaNumResourceArgsAttr, &num_resource_inputs));

  if (num_constant_inputs < 0 || num_resource_inputs < 0 ||
      num_constant_inputs + num_resource_inputs > n->num_inputs()) {
    return errors::InvalidArgument(
        "Invalid number of constant/resource arguments to XLA kernel.");
  }

  int num_non_constant_inputs =
      n->num_inputs() - num_constant_inputs - num_resource_inputs;

  std::vector<const Edge*> input_edges_vector;
  TF_RETURN_IF_ERROR(n->input_edges(&input_edges_vector));
  absl::Span<const Edge*> input_edges(input_edges_vector);

  absl::c_transform(input_edges.subspan(0, num_constant_inputs),
                    std::back_inserter(result->constant_inputs),
                    IncomingEdgeAsOutput);

  absl::c_transform(
      input_edges.subspan(num_constant_inputs, num_non_constant_inputs),
      std::back_inserter(result->non_constant_inputs), IncomingEdgeAsOutput);

  absl::c_transform(
      input_edges.subspan(num_constant_inputs + num_non_constant_inputs,
                          num_resource_inputs),
      std::back_inserter(result->resource_inputs), IncomingEdgeAsOutput);

  result->function.set_name(n->type_string());
  *result->function.mutable_attr() = n->def().attr();
  return Status::OK();
}

Status CopyIncomingControlEdges(Graph* g, Node* from, Node* to) {
  for (const Edge* e : from->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), to);
    }
  }

  return Status::OK();
}

Status ReplaceNodeWithXlaCompileAndXlaRun(Graph* g, Node* n) {
  Status status;
  Scope root = NewInternalScope(g, &status, /*refiner=*/nullptr)
                   .NewSubScope(n->name())
                   .WithDevice(n->requested_device())
                   .WithAssignedDevice(n->assigned_device_name());

  XlaClusterInfo cluster_info;
  TF_RETURN_IF_ERROR(GetXlaClusterInfo(n, &cluster_info));

  ops::_XlaCompile xla_compile(root.WithOpName("xla_compile"),
                               /*constants=*/cluster_info.constant_inputs,
                               /*args=*/cluster_info.non_constant_inputs,
                               /*resources=*/cluster_info.resource_inputs,
                               cluster_info.function);
  TF_RETURN_IF_ERROR(
      CopyIncomingControlEdges(g, /*from=*/n, /*to=*/xla_compile.key.node()));

  std::vector<Output> xla_run_args = cluster_info.non_constant_inputs;
  absl::c_copy(cluster_info.resource_inputs, std::back_inserter(xla_run_args));
  ops::_XlaRun xla_run(root.WithOpName("xla_run"), xla_run_args,
                       xla_compile.key, n->output_types());

  MoveOutgoingEdges(g, /*old_node=*/n,
                    /*new_node=*/xla_run.operation.node());
  g->RemoveNode(n);

  return Status::OK();
}
}  // namespace

Status BuildXlaOpsPass::Run(const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();

  for (Node* n : graph->op_nodes()) {
    // In all cases, only try to compile computational nodes.
    if (n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
      continue;
    }

    // Only compile nodes that are marked for compilation by the
    // compilation-marking pass (via 'attr_name').
    if (IsXlaCompiledKernel(*n)) {
      TF_RETURN_IF_ERROR(ReplaceNodeWithXlaCompileAndXlaRun(graph, n));
    }
  }

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("build_xla_ops", *graph, options.flib_def);
  }
  return Status::OK();
}
}  // namespace tensorflow
