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
#include "absl/strings/str_join.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/logging_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_jit_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {
struct DebuggingOpts {
  // If true, insert Print nodes to print every output from an XLA cluster.
  bool print_outputs;

  // If true, insert CheckNumerics nodes for every floating point typed input to
  // an XLA cluster.
  bool check_input_numerics;

  // If true, insert CheckNumerics nodes for every floating point typed output
  // from an XLA cluster.
  bool check_output_numerics;
};

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
  // The choice of data type here is important.
  //
  // We implement a "control merge", which is a control edge that is alive if
  // either of two nodes (denoted as A and B below) are alive, in the following
  // manner:
  //
  //   A --ctrl--> Const0 --data--> Merge --data--> Identity
  //                                 ^                 |
  //                                 |                ctrl
  //   B --ctrl--> Const1 --data-----+                 |
  //                                                   v
  //                                                  ***
  //
  // where *** denotes the merged control output.
  //
  // We want everything starting from Const{0/1} to Identity to either wholly
  // live on the host or wholly live on device so we need to pick a data type
  // that is either consistently assigned to the device (e.g. float) or
  // consistently assigned to the host (e.g. int32).  We should *not* pick a
  // data type that partly placed on the host and partly on the device
  // (e.g. bool constants are placed on the device but bool Identity is placed
  // on the host).
  Output data = ops::Const(scope.WithOpName("ctrl_as_data"),
                           Tensor(DT_INT32, TensorShape({0})));
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
void MergeOutgoingDataEdges(const Scope& s, Node* old_node, Node* new_node,
                            absl::string_view cluster_name,
                            const DebuggingOpts& debugging_opts) {
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
      Output new_output(new_node, oidx);
      if (debugging_opts.print_outputs) {
        string cpu_device = "/job:localhost/replica:0/task:0/device:CPU:0";
        ops::Print print_op(s.WithOpName("print_", oidx)
                                .WithDevice(cpu_device)
                                .WithAssignedDevice(cpu_device),
                            new_output, {new_output},
                            ops::Print::Attrs{}
                                .Message(absl::StrCat("output ", oidx, " from ",
                                                      old_node->name(), " is "))
                                .FirstN(1000)
                                .Summarize(-1));
        new_output = print_op;
      }

      if (debugging_opts.check_output_numerics &&
          DataTypeIsFloating(new_output.type())) {
        ops::CheckNumerics check_numerics_op(
            s.WithOpName("check_output_", oidx)
                .WithDevice(new_node->requested_device())
                .WithAssignedDevice(new_node->assigned_device_name()),
            new_output,
            absl::StrCat("CheckNumerics failed for output ", oidx, "(",
                         new_output.name(), ") from cluster ", cluster_name));
        new_output = check_numerics_op;
      }

      ops::_XlaMerge xla_merge_op(s.WithOpName("merge_oidx_", oidx),
                                  Output(old_node, oidx), new_output);
      merged_output = merged_outputs[oidx] = xla_merge_op.output;
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
  return OkStatus();
}

Status CopyIncomingControlEdges(Graph* g, Node* from, Node* to) {
  for (const Edge* e : from->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), to);
    }
  }

  return OkStatus();
}

void RemoveAllIncomingControlEdges(Graph* g, Node* n) {
  std::vector<const Edge*> incoming_ctrl_edges;
  absl::c_copy_if(n->in_edges(), std::back_inserter(incoming_ctrl_edges),
                  [](const Edge* e) { return e->IsControlEdge(); });
  for (const Edge* e : incoming_ctrl_edges) {
    g->RemoveControlEdge(e);
  }
}

// Returns true (into `result`) if a node placed on `device` must be compiled.
Status DeviceRequiresCompilation(const jit::DeviceInfoCache& device_info_cache,
                                 jit::DeviceId device, bool* result) {
  const XlaOpRegistry::DeviceRegistration* registration =
      device_info_cache.GetCompilationDevice(device);
  *result = registration->autoclustering_policy ==
            XlaOpRegistry::AutoclusteringPolicy::kAlways;
  return OkStatus();
}

// Replaces `n` with a `PartitionedCall` op that calls the same function.
StatusOr<Node*> ReplaceFunctionCallWithPartitionedCall(
    const GraphOptimizationPassOptions& options,
    const FunctionLibraryDefinition& flib_def, Node* n, Graph* g,
    const NameAttrList& func, const Scope& root) {
  string config_string = options.session_options->config.SerializeAsString();

  int input_count = absl::c_count_if(
      n->in_edges(), [](const Edge* e) { return !e->IsControlEdge(); });

  std::vector<Output> args(input_count);
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge()) {
      args[e->dst_input()] = Output(e->src(), e->src_output());
    }
  }

  // In theory we can use PartitionedCall if the XLA cluster does not have any
  // stateful operations.  However, for now we choose to be conservative since
  // we don't have any evidence that choosing a stateless partitioned call helps
  // for performance.
  ops::StatefulPartitionedCall call(
      root.WithOpName("stateful_partitioned_call"), args, n->output_types(),
      func, ops::StatefulPartitionedCall::Attrs{}.ConfigProto(config_string));

  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), call.operation.node());
    }
  }

  std::vector<const Edge*> edges_to_delete;

  for (const Edge* e : n->out_edges()) {
    edges_to_delete.push_back(e);
    if (e->IsControlEdge()) {
      g->AddControlEdge(call.operation.node(), e->dst());
    } else {
      g->AddEdge(call.operation.node(), e->src_output(), e->dst(),
                 e->dst_input());
    }
  }

  for (const Edge* e : edges_to_delete) {
    g->RemoveEdge(e);
  }

  g->RemoveNode(n);
  return call.operation.node();
}

StatusOr<jit::DeviceId> InferDeviceForCluster(
    jit::DeviceInfoCache* device_info_cache, Node* n,
    const string& function_name, const FunctionLibraryDefinition& flib_def) {
  const FunctionDef* func_def = flib_def.Find(function_name);
  TF_RET_CHECK(func_def) << "Could not find " << function_name;

  jit::DeviceSet device_set;

  for (const NodeDef& ndef : func_def->node_def()) {
    VLOG(3) << ndef.DebugString();
    if (!ndef.device().empty()) {
      TF_ASSIGN_OR_RETURN(jit::DeviceId device_id,
                          device_info_cache->GetIdFor(ndef.device()));
      device_set.Insert(device_id);
    }
  }

  if (!n->assigned_device_name().empty()) {
    // TODO(sanjoy): We need this because EncapsulateSubgraphsPass drops device
    // assignment when constant folding.  We should fix EncapsulateSubgraphsPass
    // instead.
    TF_ASSIGN_OR_RETURN(jit::DeviceId device_id,
                        device_info_cache->GetIdFor(n->assigned_device_name()));
    device_set.Insert(device_id);
  }

  TF_ASSIGN_OR_RETURN(jit::DeviceId result,
                      PickDeviceForXla(*device_info_cache, device_set,
                                       /*allow_mixing_unknown_and_cpu=*/true));
  VLOG(2) << "For " << function_name << " PickDeviceForXla("
          << device_info_cache->DebugString(device_set) << ") -> "
          << device_info_cache->GetNameFor(result);
  return result;
}

std::vector<Output> GetXlaRunArgs(const Scope& s,
                                  const XlaClusterInfo& cluster_info,
                                  const DebuggingOpts& debugging_opts) {
  std::vector<Output> xla_run_args;
  xla_run_args.reserve(cluster_info.non_constant_inputs.size() +
                       cluster_info.resource_inputs.size());
  int input_idx = 0;
  for (const Output& o : cluster_info.non_constant_inputs) {
    if (debugging_opts.check_input_numerics && DataTypeIsFloating(o.type())) {
      ops::CheckNumerics check_numerics_op(
          s.WithOpName("check_input_", input_idx), o,
          absl::StrCat("CheckNumerics failed for input ", input_idx, "(",
                       o.name(), ") into ", cluster_info.function.name()));
      xla_run_args.push_back(check_numerics_op);
    } else {
      xla_run_args.push_back(o);
    }
    input_idx++;
  }
  absl::c_copy(cluster_info.resource_inputs, std::back_inserter(xla_run_args));
  return xla_run_args;
}

StatusOr<MemoryTypeVector> GetOutputMemoryTypes(const Scope& root, Node* n) {
  MemoryTypeVector input_mtypes, output_mtypes;
  DeviceType device_type("");
  TF_RETURN_IF_ERROR(
      DeviceNameToDeviceType(n->assigned_device_name(), &device_type));
  TF_RETURN_IF_ERROR(MemoryTypesForNode(root.graph()->op_registry(),
                                        device_type, n->def(), &input_mtypes,
                                        &output_mtypes));
  return output_mtypes;
}

// Predicate INT32 typed inputs to `n` on the deadness of
// `predicate_as_control`.
//
// This is a performance optimization.  Since INT32 arguments to a
// PartitionedCall are placed on the host, a producer that produces them on the
// device will incur a D2H copy, even if the PartitionedCall is not executed
// (i.e. even if we choose to execute the XLA compiled computation via _XlaRun).
// To prevent this, we add control dependencies to make the int32 input edges
// into the PartitionedCall dead.  With this change the D2H copy only happens if
// the PartitionedCall is actually executed.
Status PredicateInt32Inputs(const Scope& root, Node* n,
                            Operation predicate_as_control) {
  std::vector<Output> int32_inputs;
  std::vector<int> int32_inputs_input_idxs;
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    if (e->src()->output_type(e->src_output()) == DT_INT32) {
      TF_ASSIGN_OR_RETURN(MemoryTypeVector source_output_mem_types,
                          GetOutputMemoryTypes(root, e->src()));
      if (source_output_mem_types[e->src_output()] == DEVICE_MEMORY) {
        int32_inputs.push_back(Output(e->src(), e->src_output()));
        int32_inputs_input_idxs.push_back(e->dst_input());
      }
    }
  }

  if (int32_inputs.empty()) {
    return OkStatus();
  }

  // Create a single IdentityN that is dead if and only if
  // `predicate_as_control` is dead.
  //
  // IdentityN is also special in that, unlike `Identity`, it does not place
  // int32 inputs in host memory.  Placing int32 inputs in host memory would
  // defeat the purpose of adding this indirection.
  ops::IdentityN identity_n(root.WithOpName("int32_id_n"), int32_inputs);
  root.graph()->AddControlEdge(predicate_as_control.node(),
                               identity_n.operation.node());

  for (int i = 0, end = int32_inputs.size(); i < end; i++) {
    TF_RETURN_IF_ERROR(root.graph()->UpdateEdge(identity_n[i].node(), i, n,
                                                int32_inputs_input_idxs[i]));
  }

  return OkStatus();
}

Status ReplaceNodeWithXlaCompileAndXlaRun(
    jit::DeviceInfoCache* device_info_cache,
    const GraphOptimizationPassOptions& options,
    const FunctionLibraryDefinition& flib_def, bool lazy_compilation_enabled,
    const DebuggingOpts& debugging_opts, Graph* g, Node* n) {
  XlaClusterInfo cluster_info;
  TF_RETURN_IF_ERROR(GetXlaClusterInfo(n, &cluster_info));

  TF_ASSIGN_OR_RETURN(
      jit::DeviceId device,
      InferDeviceForCluster(device_info_cache, n, cluster_info.function.name(),
                            flib_def));

  bool requires_compilation;
  TF_RETURN_IF_ERROR(DeviceRequiresCompilation(*device_info_cache, device,
                                               &requires_compilation));
  if (!lazy_compilation_enabled) {
    requires_compilation = true;
  }

  string device_name_str = string(device_info_cache->GetNameFor(device));

  Status status;
  Scope root = NewInternalScope(g, &status, /*refiner=*/nullptr)
                   .NewSubScope(n->name())
                   .WithDevice(n->requested_device())
                   .WithAssignedDevice(device_name_str);

  ops::_XlaCompile xla_compile(root.WithOpName("xla_compile"),
                               /*constants=*/cluster_info.constant_inputs,
                               /*args=*/cluster_info.non_constant_inputs,
                               /*resources=*/cluster_info.resource_inputs,
                               /*must_compile=*/requires_compilation,
                               cluster_info.function);

  bool has_ref_attr;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaHasReferenceVarsAttr, &has_ref_attr));
  xla_compile.operation.node()->AddAttr(kXlaHasReferenceVarsAttr, has_ref_attr);
  TF_RETURN_IF_ERROR(
      CopyIncomingControlEdges(g, /*from=*/n, /*to=*/xla_compile.key.node()));

  std::vector<Output> xla_run_args =
      GetXlaRunArgs(root, cluster_info, debugging_opts);

  if (requires_compilation) {
    // "Strict" compilation:  every _XlaCompile invocation must compile the
    // cluster.
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

    ops::_XlaRun xla_run(root.WithOpName("xla_run"), xla_run_args,
                         predicated_compilation_key, n->output_types());

    MergeOutgoingControlEdges(root, /*old_node=*/n,
                              /*new_node=*/xla_run.operation.node());

    MergeOutgoingDataEdges(root, /*old_node=*/n,
                           /*new_node=*/xla_run.operation.node(),
                           cluster_info.function.name(), debugging_opts);

    TF_RETURN_IF_ERROR(root.status());

    // We already have a TensorFlow function call into the cluster -- the
    // original node we set out to rewrite.  We just wire in the correct control
    // deps and we're done.
    RemoveAllIncomingControlEdges(g, n);
    Operation inverse_predicate_as_control =
        DataToControl(root, inverse_predicated_compilation_key);
    g->AddControlEdge(inverse_predicate_as_control.node(), n);
    n->ClearAttr(kXlaCompiledKernelAttr);

    TF_ASSIGN_OR_RETURN(Node* const pco, ReplaceFunctionCallWithPartitionedCall(
                                             options, flib_def, n, g,
                                             cluster_info.function, root));

    TF_RETURN_IF_ERROR(
        PredicateInt32Inputs(root, pco, inverse_predicate_as_control));
  }

  return OkStatus();
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
          : GetBuildXlaOpsPassFlags()->tf_xla_enable_lazy_compilation;

  jit::DeviceInfoCache device_info_cache;
  const BuildXlaOpsPassFlags& flags = *GetBuildXlaOpsPassFlags();

  DebuggingOpts debugging_opts;
  debugging_opts.print_outputs = flags.tf_xla_print_cluster_outputs;
  debugging_opts.check_input_numerics =
      flags.tf_xla_check_cluster_input_numerics;
  debugging_opts.check_output_numerics =
      flags.tf_xla_check_cluster_output_numerics;

  VLOG(1) << "print_outputs = " << debugging_opts.print_outputs;
  VLOG(1) << "check_input_numerics = " << debugging_opts.check_input_numerics;
  VLOG(1) << "check_output_numerics = " << debugging_opts.check_output_numerics;

  for (Node* n : xla_compiled_kernels) {
    TF_RETURN_IF_ERROR(ReplaceNodeWithXlaCompileAndXlaRun(
        &device_info_cache, options, *options.flib_def,
        lazy_compilation_enabled, debugging_opts, graph, n));
  }

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("build_xla_ops", *graph, options.flib_def);
  }

  return OkStatus();
}
}  // namespace tensorflow
