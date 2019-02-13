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
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
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

// Returns a data value that is dead iff `control` is dead.
Output ControlToData(const Scope& scope, Node* control) {
  Output data = ops::Const(scope.WithOpName("ctrl_as_data"),
                           Tensor(DT_BOOL, TensorShape({0})));
  scope.graph()->AddControlEdge(control, data.node());
  return Output(data.node());
}

// Returns an operation that can be control-depended on that is dead iff `data`
// is dead.
Operation DataToControl(const Scope& scope, Output data) {
  return Operation(
      ops::Identity(scope.WithOpName("data_as_ctrl"), data).node());
}

// Replaces each outgoing edge from `old_node` with a merge node that merges in
// the corresponding output from `new_node`.
void MergeOutgoingDataEdges(const Scope& s, Node* old_node, Node* new_node) {
  if (!s.status().ok()) {
    return;
  }

  std::vector<Output> merged_outputs(old_node->num_outputs(), Output(nullptr));

  std::vector<const Edge*> data_edges;
  absl::c_copy_if(old_node->out_edges(), std::back_inserter(data_edges),
                  [](const Edge* e) { return !e->IsControlEdge(); });

  for (const Edge* e : data_edges) {
    int oidx = e->src_output();
    Output merged_output = merged_outputs[oidx];
    if (merged_output.node() == nullptr) {
      ops::Merge merge_op(s.WithOpName(absl::StrCat("merge_oidx_", oidx)),
                          {Output(old_node, oidx), Output(new_node, oidx)});
      merged_output = merged_outputs[oidx] = merge_op.output;
    }

    Node* dst = e->dst();
    int dst_idx = e->dst_input();

    s.graph()->RemoveEdge(e);
    s.graph()->AddEdge(merged_output.node(), merged_output.index(), dst,
                       dst_idx);
  }
}

// Replaces each control successor of `old_node` to execute whenever either
// `old_node` or `new_node` is executed.
void MergeOutgoingControlEdges(const Scope& s, Node* old_node, Node* new_node) {
  if (!s.status().ok()) {
    return;
  }

  std::vector<const Edge*> ctrl_edges;
  absl::c_copy_if(old_node->out_edges(), std::back_inserter(ctrl_edges),
                  [](const Edge* e) { return e->IsControlEdge(); });

  if (ctrl_edges.empty()) {
    return;
  }

  if (ctrl_edges.size() == 1 && ctrl_edges.front()->dst()->IsSink()) {
    // Avoid creating a Merge node if we can just add an edge to _SINK
    // instead.
    s.graph()->AddControlEdge(new_node, s.graph()->sink_node());
    return;
  }

  // We can't merge control edges directly so we instead first "convert" them to
  // normal values that can be merged, merge the values and then "convert" the
  // merged value back into control.
  //
  // NB! We need to copy out the outgoing control edges before constructing
  // old_ctrl_as_data otherwise the control edge from old_node to the constant
  // in ControlToData will be present in ctrl_edges.

  Output old_ctrl_as_data = ControlToData(s, old_node);
  Output new_ctrl_as_data = ControlToData(s, new_node);

  ops::Merge ctrl_merge_as_data(s.WithOpName("ctrl_merge"),
                                {old_ctrl_as_data, new_ctrl_as_data});
  Operation ctrl_merge = DataToControl(s, ctrl_merge_as_data.output);

  for (const Edge* e : ctrl_edges) {
    s.graph()->AddControlEdge(ctrl_merge.node(), e->dst());
    s.graph()->RemoveControlEdge(e);
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

void RemoveAllIncomingControlEdges(Graph* g, Node* n) {
  std::vector<const Edge*> incoming_ctrl_edges;
  absl::c_copy_if(n->in_edges(), std::back_inserter(incoming_ctrl_edges),
                  [](const Edge* e) { return e->IsControlEdge(); });
  for (const Edge* e : incoming_ctrl_edges) {
    g->RemoveControlEdge(e);
  }
}

// Returns true (into `result`) if `node` must be compiled.
Status NodeRequiresCompilation(Node* n, bool* result) {
  DeviceType device_type("");
  TF_RETURN_IF_ERROR(
      DeviceToDeviceType(n->assigned_device_name(), &device_type));
  const XlaOpRegistry::DeviceRegistration* registration = nullptr;
  if (!XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration)) {
    return errors::Internal("Could not find compilation device ",
                            device_type.type());
  }
  *result = registration->autoclustering_policy ==
            XlaOpRegistry::AutoclusteringPolicy::kAlways;
  return Status::OK();
}

Status ReplaceNodeWithXlaCompileAndXlaRun(
    const FunctionLibraryDefinition& flib_def, bool lazy_compilation_enabled,
    Graph* g, Node* n) {
  bool requires_compilation;
  TF_RETURN_IF_ERROR(NodeRequiresCompilation(n, &requires_compilation));
  if (!lazy_compilation_enabled) {
    requires_compilation = true;
  }

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
                               /*must_compile=*/requires_compilation,
                               cluster_info.function);
  TF_RETURN_IF_ERROR(
      CopyIncomingControlEdges(g, /*from=*/n, /*to=*/xla_compile.key.node()));

  if (requires_compilation) {
    // "Strict" compilation:  every _XlaCompile invocation must compile the
    // cluster.
    std::vector<Output> xla_run_args = cluster_info.non_constant_inputs;
    absl::c_copy(cluster_info.resource_inputs,
                 std::back_inserter(xla_run_args));
    ops::_XlaRun xla_run(root.WithOpName("xla_run"), xla_run_args,
                         xla_compile.key, n->output_types());

    MoveOutgoingEdges(g, /*old_node=*/n,
                      /*new_node=*/xla_run.operation.node());
    g->RemoveNode(n);
  } else {
    // "Lazy" compilation: an _XlaCompile invocation may decide not to compile
    // the cluster based on profitability heuristics.

    // We generate the following graph:
    //
    //   (use_tf_call, use_xla_run) =
    //       Switch(pred=xla_compile.compilation_successful,
    //              value=xla_compile.key)
    //
    //   tf_call_outputs = cluster_N(..., ^use_tf_call)
    //   xla_run_outputs = _XlaRun(..., key=use_xla_run)
    //   outputs = Merge(tf_call_outputs, xla_run_outputs).
    ops::Switch s(root.WithOpName("predicated_compilation_key"),
                  xla_compile.key, xla_compile.compilation_successful);
    Output predicated_compilation_key = s.output_true;
    Output inverse_predicated_compilation_key = s.output_false;

    std::vector<Output> xla_run_args = cluster_info.non_constant_inputs;
    absl::c_copy(cluster_info.resource_inputs,
                 std::back_inserter(xla_run_args));
    ops::_XlaRun xla_run(root.WithOpName("xla_run"), xla_run_args,
                         predicated_compilation_key, n->output_types());

    MergeOutgoingControlEdges(root, /*old_node=*/n,
                              /*new_node=*/xla_run.operation.node());

    MergeOutgoingDataEdges(root, /*old_node=*/n,
                           /*new_node=*/xla_run.operation.node());

    TF_RETURN_IF_ERROR(root.status());

    // We already have a TensorFlow function call into the cluster -- the
    // original node we set out to rewrite.  We just wire in the correct control
    // deps and we're done.
    RemoveAllIncomingControlEdges(g, n);
    g->AddControlEdge(
        DataToControl(root, inverse_predicated_compilation_key).node(), n);
    n->ClearAttr(kXlaCompiledKernelAttr);
  }

  return Status::OK();
}
}  // namespace

Status BuildXlaOpsPass::Run(const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();

  // Copy out the nodes we want to rewrite to avoid modifying the graph while we
  // iterate on graph->op_nodes().
  std::vector<Node*> xla_compiled_kernels;
  absl::c_copy_if(graph->op_nodes(), std::back_inserter(xla_compiled_kernels),
                  [](const Node* n) {
                    if (n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
                      return false;
                    }

                    // Only compile nodes that are marked for compilation by the
                    // compilation-marking pass (via 'attr_name').
                    return IsXlaCompiledKernel(*n);
                  });

  bool lazy_compilation_enabled =
      enable_lazy_compilation_
          ? *enable_lazy_compilation_
          : GetBuildXlaOpsPassFlags().tf_xla_enable_lazy_compilation;

  for (Node* n : xla_compiled_kernels) {
    TF_RETURN_IF_ERROR(ReplaceNodeWithXlaCompileAndXlaRun(
        *options.flib_def, lazy_compilation_enabled, graph, n));
  }

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("build_xla_ops", *graph, options.flib_def);
  }

  return Status::OK();
}
}  // namespace tensorflow
