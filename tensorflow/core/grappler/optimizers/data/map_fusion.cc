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

#include "tensorflow/core/grappler/optimizers/data/map_fusion.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

// Sets basic function parameters and copies attributes from parent and map
// node.
NodeDef MakeFusedNode(const NodeDef& parent_map_node, const NodeDef& map_node,
                      const FunctionDef& fused_function,
                      MutableGraphView* graph) {
  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName("fused_map", graph->GetGraph(),
                                      &fused_node);

  fused_node.set_op("MapDataset");
  fused_node.add_input(parent_map_node.input(0));

  auto copy_attribute = [](const string& attribute_name, const NodeDef& from,
                           NodeDef* to) {
    (*to->mutable_attr())[attribute_name] = from.attr().at(attribute_name);
  };

  auto attr = parent_map_node.attr().at("f");
  *attr.mutable_func()->mutable_name() = fused_function.signature().name();
  (*fused_node.mutable_attr())["f"] = std::move(attr);

  copy_attribute("Targuments", parent_map_node, &fused_node);

  for (auto key : {"output_shapes", "output_types"})
    copy_attribute(key, map_node, &fused_node);

  return fused_node;
}

string ParseNodeConnection(const string& name) {
  // If input/output node name has semicolon, take the prefix.  Otherwise take
  // the whole string.
  return name.substr(0, name.find(':'));
}

string ParseOutputNode(const string& name) {
  return name.substr(name.find(':'), string::npos);
}

const string& GetOutputNode(const FunctionDef& parent_function,
                            int output_idx) {
  const auto& ret_output_name =
      parent_function.signature().output_arg(output_idx).name();
  return parent_function.ret().at(ret_output_name);
}

// Nodes that will be added to the function can have the same name as the nodes
// from parent function.  We need to rename them and the connections of the
// inputs that refer to them.
void RenameFunctionNodes(FunctionDef* fused_function,
                         protobuf::RepeatedPtrField<NodeDef>* nodes_to_fuse) {
  std::unordered_map<string, string> changed_node_names;
  for (NodeDef& function_node : *nodes_to_fuse) {
    string name_before = function_node.name();
    graph_utils::SetUniqueFunctionNodeName(name_before, fused_function,
                                           &function_node);
    if (name_before != function_node.name())
      changed_node_names[name_before] = function_node.name();
  }

  auto update_name = [&changed_node_names](string* input) {
    string input_node = ParseNodeConnection(*input);
    if (changed_node_names.count(input_node) == 0) return;
    const string& new_node_name = changed_node_names.at(input_node);
    *input = new_node_name + ParseOutputNode(*input);
  };

  for (NodeDef& function_node : *nodes_to_fuse) {
    for (string& input : *function_node.mutable_input()) {
      update_name(&input);
    }
  }

  for (auto& ret : *fused_function->mutable_ret()) update_name(&ret.second);
}

// This function adds new nodes and changes their input to the output nodes
// of parent function.
void FuseFunctionNodes(const FunctionDef& parent_function,
                       const FunctionDef& function,
                       protobuf::RepeatedPtrField<NodeDef>* nodes_to_fuse) {
  const auto number_of_outputs = parent_function.signature().output_arg_size();
  CHECK(number_of_outputs == function.signature().input_arg_size())
      << "The number of input arguments of function "
      << function.signature().name()
      << " should be the same as the number of output arguments of function "
      << parent_function.signature().name() << ".";

  for (int output_idx = 0; output_idx < number_of_outputs; output_idx++) {
    const string& output = GetOutputNode(parent_function, output_idx);

    const auto& input_node_name =
        function.signature().input_arg(output_idx).name();

    for (NodeDef& function_node : *nodes_to_fuse) {
      for (auto& node_input : *function_node.mutable_input()) {
        auto parsed_name = ParseNodeConnection(node_input);
        if (parsed_name != input_node_name) continue;

        node_input = output;
      }
    }
  }
}

// This function looks for direct edges from input to return and rewrites
// them to the coresponding input of the return of parent_function.
void FuseReturns(const FunctionDef& parent_function,
                 const FunctionDef& function, FunctionDef* fused_function) {
  const auto number_of_inputs = function.signature().input_arg_size();

  for (auto& ret : *fused_function->mutable_ret()) {
    auto return_input = ParseNodeConnection(ret.second);
    for (int input_idx = 0; input_idx < number_of_inputs; input_idx++) {
      const auto& input_arg = function.signature().input_arg(input_idx);
      if (return_input != input_arg.name()) continue;

      ret.second = GetOutputNode(parent_function, input_idx);
    }
  }
}

// This function produces new function that is a result of fusion of
// `parent_function` with `function`.
FunctionDef* FuseFunctions(const FunctionDef& parent_function,
                           const FunctionDef& function,
                           FunctionDefLibrary* library) {
  FunctionDef* fused_function = library->add_function();
  graph_utils::SetUniqueGraphFunctionName("fused_function", library,
                                          fused_function);

  // Copy input signature from parent function.
  *fused_function->mutable_signature()->mutable_input_arg() =
      parent_function.signature().input_arg();

  fused_function->mutable_node_def()->CopyFrom(parent_function.node_def());
  // This code assumes functions does not have any attributes. If this is
  // not the case, we need to merge attributes and fix name conflicts.
  CHECK(parent_function.attr_size() == 0 && function.attr_size() == 0 &&
        "Functions with attributes are currently not supported");

  // Copy the returns and output signature from the second node.
  auto nodes_to_fuse = function.node_def();
  fused_function->mutable_signature()->mutable_output_arg()->CopyFrom(
      function.signature().output_arg());
  *fused_function->mutable_ret() = function.ret();

  RenameFunctionNodes(fused_function, &nodes_to_fuse);
  FuseFunctionNodes(parent_function, function, &nodes_to_fuse);
  FuseReturns(parent_function, function, fused_function);

  // Copy transformed nodes from the second function.
  fused_function->mutable_node_def()->MergeFrom(nodes_to_fuse);

  return fused_function;
}

}  // namespace

Status MapFusion::Optimize(Cluster* cluster, const GrapplerItem& item,
                           GraphDef* output) {
  GraphDef sorted_old_graph = item.graph;
  TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
  *output = sorted_old_graph;

  MutableGraphView graph(output);
  std::set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());

  auto get_map_node = [](const NodeDef& node) -> const NodeDef* {
    // TODO(prazek): we could also handle ParallelMapDataset and
    // MapAndBatchDataset.
    if (node.op() == "MapDataset") return &node;
    return nullptr;
  };

  auto get_fused_function = [&function_library, &output](
                                const NodeDef* parent_map_node,
                                const NodeDef* map_node) {
    const auto& parent_fun = parent_map_node->attr().at("f");
    const FunctionDef* parent_func =
        function_library.Find(parent_fun.func().name());
    const auto& fun = map_node->attr().at("f");
    const FunctionDef* func = function_library.Find(fun.func().name());

    return FuseFunctions(*parent_func, *func, output->mutable_library());
  };

  for (const NodeDef& node : sorted_old_graph.node()) {
    const NodeDef* map_node = get_map_node(node);
    if (!map_node) continue;

    GraphView::InputPort input_port = graph.GetInputPort(map_node->name(), 0);
    const NodeDef* parent_map_node =
        get_map_node(*graph.GetRegularFanin(input_port).node);
    if (!parent_map_node) continue;

    const auto* fused_function = get_fused_function(parent_map_node, map_node);
    const auto* fused_maps_node = graph.AddNode(
        MakeFusedNode(*parent_map_node, *map_node, *fused_function, &graph));

    graph.ReplaceInput(*map_node, *fused_maps_node);

    // TODO(prazek): we should run some optimizations on the fused map
    // functions, or make sure that optimization passes run after map
    // fusion.
    TF_RETURN_IF_ERROR(function_library.AddFunctionDef(*fused_function));

    // TODO(prazek): we could also remove map functions from library if they
    // are not used anymore.
    nodes_to_delete.insert(parent_map_node->name());
    nodes_to_delete.insert(map_node->name());
  }

  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void MapFusion::Feedback(Cluster* cluster, const GrapplerItem& item,
                         const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(MapFusion, "map_fusion");

}  // end namespace grappler
}  // end namespace tensorflow
