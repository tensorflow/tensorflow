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
      GetNodeAttr(node->attrs(), kXlaNumConstantArgsAttr, &num_constant_args));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), kXlaNumResourceArgsAttr, &num_resource_args));

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
      node->requested_device(), const_dtypes, num_resource_args, arg_dtypes,
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

  for (Node* n : graph->op_nodes()) {
    // In all cases, only try to compile computational nodes.
    if (n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
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
}  // namespace tensorflow
