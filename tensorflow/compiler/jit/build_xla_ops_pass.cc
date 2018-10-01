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

static Status BuildXlaCompileNode(
    const string& nodename, const string& function_name,
    const AttrValueMap& function_attr, const string& device_name,
    const DataTypeVector& constant_dtypes, int num_resources,
    const DataTypeVector& arg_dtypes, Graph* graph, Node** node) {
  NodeDef def;
  def.set_name(graph->NewName(nodename));
  def.set_op("_XlaCompile");
  def.set_device(device_name);
  AddNodeAttr("Tconstants", constant_dtypes, &def);
  AddNodeAttr("Targs", arg_dtypes, &def);
  AddNodeAttr("Nresources", num_resources, &def);
  NameAttrList function;
  function.set_name(function_name);
  *function.mutable_attr() = function_attr;
  AddNodeAttr("function", function, &def);

  Status status;
  *node = graph->AddNode(def, &status);
  return status;
}

static Status BuildXlaRunNode(const string& nodename, const string& device_name,
                              const DataTypeVector& arg_dtypes,
                              const DataTypeVector& result_dtypes, Graph* graph,
                              Node** node) {
  NodeDef def;
  def.set_name(graph->NewName(nodename));
  def.set_op("_XlaRun");
  def.set_device(device_name);
  AddNodeAttr("Targs", arg_dtypes, &def);
  AddNodeAttr("Tresults", result_dtypes, &def);

  Status status;
  *node = graph->AddNode(def, &status);
  return status;
}

static Status GetXlaAttrs(Node* node, int* num_constant_args,
                          int* num_resource_args, DataTypeVector* const_dtypes,
                          DataTypeVector* arg_dtypes) {
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), kXlaNumConstantArgsAttr, num_constant_args));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), kXlaNumResourceArgsAttr, num_resource_args));

  if (*num_constant_args < 0 || *num_resource_args < 0 ||
      *num_constant_args + *num_resource_args > node->num_inputs()) {
    return errors::InvalidArgument(
        "Invalid number of constant/resource arguments to XLA kernel.");
  }

  const int num_nonconst_args =
      node->num_inputs() - *num_constant_args - *num_resource_args;

  const DataTypeVector& input_types = node->input_types();
  std::copy(input_types.begin(), input_types.begin() + *num_constant_args,
            std::back_inserter(*const_dtypes));
  std::copy(input_types.begin() + *num_constant_args,
            input_types.begin() + *num_constant_args + num_nonconst_args,
            std::back_inserter(*arg_dtypes));
  return Status::OK();
}

static void CopyIncomingEdges(Graph* g, Node* old_node, Node* new_node,
                              int prefix_to_ignore) {
  for (const Edge* edge : old_node->in_edges()) {
    if (edge->IsControlEdge()) {
      g->AddControlEdge(edge->src(), new_node);
    } else if (edge->dst_input() >= prefix_to_ignore) {
      g->AddEdge(edge->src(), edge->src_output(), new_node,
                 edge->dst_input() - prefix_to_ignore);
    }
  }
}

static void MoveOutgoingEdges(Graph* g, Node* old_node, Node* new_node) {
  std::vector<const Edge*> out_edges(old_node->out_edges().begin(),
                                     old_node->out_edges().end());
  for (const Edge* edge : out_edges) {
    // TODO(sanjoy): This does not update NodeDef inputs.
    g->AddEdge(new_node, edge->src_output(), edge->dst(), edge->dst_input());
    g->RemoveEdge(edge);
  }
}

static Status ReplaceNodeWithXlaCompileAndRun(Graph* g, Node* n) {
  int num_constant_args, num_resource_args;
  DataTypeVector const_dtypes;
  DataTypeVector arg_dtypes;

  TF_RETURN_IF_ERROR(GetXlaAttrs(n, &num_constant_args, &num_resource_args,
                                 &const_dtypes, &arg_dtypes));

  Node *compile_node, *run_node;

  TF_RETURN_IF_ERROR(BuildXlaCompileNode(
      n->name(), n->type_string(), n->def().attr(), n->requested_device(),
      const_dtypes, num_resource_args, arg_dtypes, g, &compile_node));

  DataTypeVector arg_dtypes_with_resources = arg_dtypes;
  for (int i = 0; i < num_resource_args; i++) {
    arg_dtypes_with_resources.push_back(DT_RESOURCE);
  }

  TF_RETURN_IF_ERROR(BuildXlaRunNode(n->name(), n->requested_device(),
                                     arg_dtypes_with_resources,
                                     n->output_types(), g, &run_node));

  compile_node->set_assigned_device_name(n->assigned_device_name());
  run_node->set_assigned_device_name(n->assigned_device_name());

  CopyIncomingEdges(g, /*old_node=*/n, /*new_node=*/compile_node,
                    /*prefix_to_ignore=*/0);
  CopyIncomingEdges(g, /*old_node=*/n, /*new_node=*/run_node,
                    /*prefix_to_ignore=*/num_constant_args);

  // The compilation_key output.
  g->AddEdge(compile_node, 0, run_node, n->num_inputs() - num_constant_args);

  MoveOutgoingEdges(g, /*old_node=*/n, /*new_node=*/run_node);
  g->RemoveNode(n);

  return Status::OK();
}

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
      TF_RETURN_IF_ERROR(ReplaceNodeWithXlaCompileAndRun(graph, n));
    }
  }

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("build_xla_ops", *graph, options.flib_def);
  }
  return Status::OK();
}
}  // namespace tensorflow
