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

#include "tensorflow/compiler/jit/build_xla_launch_ops_pass.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/kernels/xla_local_launch_op.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

static Status BuildLaunchNode(
    const string& nodename, const string& function_name,
    const AttrValueMap& function_attr, const string& device_name,
    const DataTypeVector& constant_dtypes, int num_resources,
    const DataTypeVector& arg_dtypes, const DataTypeVector& result_dtypes,
    Graph* graph, Node** node) {
  NodeDef def;
  def.set_name(graph->NewName(nodename));
  def.set_op("_XlaLaunch");
  def.set_device(device_name);
  AddNodeAttr("Tconstants", constant_dtypes, &def);
  AddNodeAttr("Targs", arg_dtypes, &def);
  AddNodeAttr("Nresources", num_resources, &def);
  AddNodeAttr("Tresults", result_dtypes, &def);
  NameAttrList function;
  function.set_name(function_name);
  *function.mutable_attr() = function_attr;
  AddNodeAttr("function", function, &def);

  Status status;
  *node = graph->AddNode(def, &status);
  return status;
}

static Status ReplaceNodeWithXlaLaunch(Graph* graph, Node* node) {
  VLOG(2) << "Replacing " << node->name() << " with XlaLaunch";

  int num_constant_args, num_resource_args;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->def(), kXlaNumConstantArgsAttr, &num_constant_args));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->def(), kXlaNumResourceArgsAttr, &num_resource_args));

  if (num_constant_args < 0 || num_resource_args < 0 ||
      num_constant_args + num_resource_args > node->num_inputs()) {
    return errors::InvalidArgument(
        "Invalid number of constant/resource arguments to XLA kernel.");
  }
  const int num_nonconst_args =
      node->num_inputs() - num_constant_args - num_resource_args;

  DataTypeVector const_dtypes(node->input_types().begin(),
                              node->input_types().begin() + num_constant_args);
  DataTypeVector arg_dtypes(
      node->input_types().begin() + num_constant_args,
      node->input_types().begin() + num_constant_args + num_nonconst_args);

  // Build a _XlaLaunch operator to execute the function body.
  Node* launch_node;
  TF_RETURN_IF_ERROR(BuildLaunchNode(
      graph->NewName(node->name()), node->type_string(), node->def().attr(),
      node->def().device(), const_dtypes, num_resource_args, arg_dtypes,
      node->output_types(), graph, &launch_node));
  launch_node->set_assigned_device_name(node->assigned_device_name());

  // Copy incoming edges to the launch node.
  for (const Edge* edge : node->in_edges()) {
    if (edge->IsControlEdge()) {
      graph->AddControlEdge(edge->src(), launch_node);
    } else {
      graph->AddEdge(edge->src(), edge->src_output(), launch_node,
                     edge->dst_input());
    }
  }

  // Copy outgoing edges to the launch node.
  std::vector<const Edge*> out_edges(node->out_edges().begin(),
                                     node->out_edges().end());
  for (const Edge* edge : out_edges) {
    Node* dst = edge->dst();
    int src_output = edge->src_output();
    int dst_input = edge->dst_input();
    graph->RemoveEdge(edge);

    if (edge->IsControlEdge()) {
      graph->AddControlEdge(launch_node, dst);
    } else {
      graph->AddEdge(launch_node, src_output, dst, dst_input);
    }
  }
  graph->RemoveNode(node);

  return Status::OK();
}

Status BuildXlaLaunchOpsPass::Run(const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();

  for (Node* n : graph->nodes()) {
    // In all cases, only try to compile computational nodes.
    if (!n->IsOp() || n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
      continue;
    }

    // Only compile nodes that are marked for compilation by the
    // compilation-marking pass (via 'attr_name').
    if (IsXlaCompiledKernel(*n)) {
      TF_RETURN_IF_ERROR(ReplaceNodeWithXlaLaunch(graph, n));
    }
  }

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("build_xla_launch_ops", *graph,
                                options.flib_def);
  }
  return Status::OK();
}

namespace {

// Givens a NodeDef 'ndef' and the function library runtime 'flr', if
// 'ndef' is a call to a compilable function defined in 'flr', returns OK
// and fills in 'kernel' with a XlaLaunchOp kernel which computes the
// node. Otherwise, returns a non-OK.
//
// This routine is here so that FunctionLibraryRuntime can jit a
// specific function call as requested.
Status CreateXlaLaunchOp(FunctionLibraryRuntime* flr, const NodeDef& ndef,
                         std::unique_ptr<OpKernel>* kernel) {
  bool xla_compile = false;
  if (!flr->GetFunctionLibraryDefinition()
           ->GetAttr(ndef, kXlaCompileAttr, &xla_compile)
           .ok() ||
      !xla_compile) {
    // Not marked as _XlaCompile=true.
    return errors::InvalidArgument("No ", kXlaCompileAttr, " for ", ndef.op());
  }
  // Make sure that kernels have been registered on the JIT device.
  XlaOpRegistry::RegisterCompilationKernels();
  if (!IsCompilable(flr, ndef)) {
    // ndef is calling a function that XLA can't compile.
    return errors::InvalidArgument("Not compilable: ", ndef.ShortDebugString());
  }
  FunctionLibraryRuntime::Handle handle;
  // If ndef is not instantiable, e.g., the function does not exist,
  // simply bail out.
  TF_RETURN_IF_ERROR(flr->Instantiate(ndef.op(), ndef.attr(), &handle));
  const FunctionBody* fbody = flr->GetFunctionBody(handle);
  CHECK(fbody);  // Can't be nullptr since we just instantiated it.
  std::vector<bool> const_args(fbody->arg_types.size());
  // If we can't analyze the const args. Bail out.
  TF_RETURN_IF_ERROR(BackwardsConstAnalysis(*(fbody->graph), &const_args));

  for (int i = 0; i < const_args.size(); ++i) {
    if (const_args[i]) {
      // There is a const arg. Bail out.
      return errors::InvalidArgument("Const arg: ", i, " in ",
                                     DebugString(fbody->fdef));
    }
  }

  NodeDef launch_def;
  launch_def.set_name(ndef.name());
  launch_def.set_op("_XlaLaunch");
  launch_def.set_device(flr->device()->name());
  AddNodeAttr("Tconstants", DataTypeVector{}, &launch_def);
  AddNodeAttr("Nresources", 0, &launch_def);
  AddNodeAttr("Targs", fbody->arg_types, &launch_def);
  AddNodeAttr("Tresults", fbody->ret_types, &launch_def);
  NameAttrList func;
  func.set_name(ndef.op());
  *(func.mutable_attr()) = ndef.attr();
  AddNodeAttr("function", func, &launch_def);

  // TODO(b/32387911): Handles the host memory types across function
  // calls properly. For now, we assume all inputs and outputs are on
  // the device memory.
  MemoryTypeVector input_memory_types(fbody->arg_types.size(), DEVICE_MEMORY);
  MemoryTypeVector output_memory_types(fbody->ret_types.size(), DEVICE_MEMORY);

  Device* dev = flr->device();
  Status s;
  OpKernelConstruction construction(
      DeviceType(dev->device_type()), dev,
      dev->GetAllocator(AllocatorAttributes()), &launch_def,
      &fbody->fdef.signature(), flr, fbody->arg_types, input_memory_types,
      fbody->ret_types, output_memory_types, flr->graph_def_version(), &s);
  kernel->reset(new XlaLocalLaunchOp(&construction));
  return s;
}

bool RegisterLaunchOpCreator() {
  RegisterDefaultCustomKernelCreator(CreateXlaLaunchOp);
  return true;
}

static bool register_me = RegisterLaunchOpCreator();

}  // end namespace

}  // namespace tensorflow
