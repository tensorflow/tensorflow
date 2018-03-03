/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include <unordered_map>
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"

namespace tensorflow {
namespace grappler {

Status InlineFunction(const NodeDef& node, const FunctionDef& func,
                      const FunctionDefLibrary& library, GraphDef* graph) {
  const std::unordered_map<string, AttrValue> attr(node.attr().begin(),
                                                   node.attr().end());
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, attr, library);
  if (!item) {
    return errors::InvalidArgument("Failed to inline function ", node.op(),
                                   " instantiated by ", node.name());
  }

  std::unordered_map<string, int> input_nodes;
  for (int i = 0; i < func.signature().input_arg_size(); ++i) {
    const OpDef::ArgDef& arg = func.signature().input_arg(i);
    input_nodes[arg.name()] = i;
  }

  // Add an IdentityN op to hook the function inputs to: this ensures that
  // they're all evaluated before the evaluation of the function body starts.
  NodeDef* func_inputs = graph->add_node();
  func_inputs->set_name(strings::StrCat(node.name(), "/", "inlined_inputs"));
  func_inputs->set_op("IdentityN");
  func_inputs->set_device(node.device());
  *func_inputs->mutable_input() = node.input();
  AttrValue::ListValue* type_list =
      (*func_inputs->mutable_attr())["T"].mutable_list();
  for (const OpDef::ArgDef& arg : func.signature().input_arg()) {
    if (arg.type() != DT_INVALID) {
      type_list->add_type(arg.type());
    } else {
      auto it = attr.find(arg.type_attr());
      if (it == attr.end()) {
        return errors::InvalidArgument("Invalid input argument ", arg.name(),
                                       " for function ", node.op(),
                                       " instantiated by ", node.name());
      }
      type_list->add_type(it->second.type());
    }
  }

  for (NodeDef& func_body_node : *item->graph.mutable_node()) {
    if (input_nodes.find(func_body_node.name()) != input_nodes.end()) {
      // Turn input placeholders into identity nodes
      if (IsPlaceholder(func_body_node)) {
        func_body_node.set_op("Identity");
      }
      CHECK_EQ(0, func_body_node.input_size());
      int input_id = input_nodes[func_body_node.name()];
      func_body_node.add_input(
          strings::StrCat(func_inputs->name(), ":", input_id));
    } else {
      // Update the input names.
      for (string& input : *func_body_node.mutable_input()) {
        input = AddPrefixToNodeName(input, node.name());
      }
    }

    // Add the node name as a prefix to avoid collisions after inlining
    func_body_node.set_name(
        strings::StrCat(node.name(), "/", func_body_node.name()));

    // Make sure the node is placed
    func_body_node.set_device(node.device());

    // Move the node to the main graph
    graph->add_node()->Swap(&func_body_node);
  }

  // Add an IdentityN op to hook the function outputs to: this ensures that the
  // function body is fully evaluated before its fanout gets scheduled.
  NodeDef* func_outputs = graph->add_node();
  func_outputs->set_name(node.name());
  func_outputs->set_op("IdentityN");
  func_outputs->set_device(node.device());
  type_list = (*func_outputs->mutable_attr())["T"].mutable_list();
  for (int i = 0; i < func.signature().output_arg_size(); ++i) {
    const OpDef::ArgDef& arg = func.signature().output_arg(i);
    if (arg.type() != DT_INVALID) {
      type_list->add_type(arg.type());
    } else {
      auto it = attr.find(arg.type_attr());
      if (it == attr.end()) {
        return errors::InvalidArgument("Invalid output argument ", arg.name(),
                                       " for function ", node.op(),
                                       " instantiated by ", node.name());
      }
      type_list->add_type(it->second.type());
    }
    // Use the fetch names since they take into account the output mapping.
    func_outputs->add_input(strings::StrCat(node.name(), "/", item->fetch[i]));
  }

  return Status::OK();
}

Status FunctionOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                   GraphDef* optimized_graph) {
  std::unordered_map<string, const FunctionDef*> functions;
  for (const FunctionDef& func : item.graph.library().function()) {
    // Don't inline functions marked as noinline
    if (func.attr().count("_noinline") != 0) {
      continue;
    }
    // Can't create IdentityN nodes with no input or output: skip these
    // functions for now.
    if (func.signature().input_arg_size() == 0 ||
        func.signature().output_arg_size() == 0) {
      continue;
    }
    functions[func.signature().name()] = &func;
  }

  // Nothing to do.
  if (functions.empty()) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  // Inline functions when possible.
  for (const NodeDef& node : item.graph.node()) {
    auto it = functions.find(node.op());
    if (it == functions.end()) {
      *optimized_graph->add_node() = node;
    } else {
      TF_RETURN_IF_ERROR(InlineFunction(node, *it->second, item.graph.library(),
                                        optimized_graph));
    }
  }

  // TODO(bsteiner): specialize the implementation of functions that can't be
  // inlined based on the context in which they're instantiated.

  // TODO(bsteiner): trim the library to remove unused function definitions
  *optimized_graph->mutable_library() = item.graph.library();
  *optimized_graph->mutable_versions() = item.graph.versions();

  return Status::OK();
}

void FunctionOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                 const GraphDef& optimized_graph,
                                 double result) {
  // Nothing to do for FunctionOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
