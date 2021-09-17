/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/time/clock.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace tfrt_stub {

StatusOr<std::unique_ptr<TfrtGraphExecutionState>>
TfrtGraphExecutionState::Create(tensorflow::GraphDef graph_def,
                                const FallbackState& fallback_state) {
  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("create_input_graph_def", graph_def);
  }

  TF_RETURN_IF_ERROR(tensorflow::GenerateResourceSharedNameIfEmpty(
      graph_def, tensorflow::OpRegistry::Global()));

  if (VLOG_IS_ON(2)) {
    DumpGraphDefToFile("after_generate_resource_shared_name_graph_def",
                       graph_def);
  }

  // `CreateExecutionState()` will preprocess the graph (e.g., apply Placer).
  TF_ASSIGN_OR_RETURN(
      auto graph_execution_state,
      fallback_state.CreateGraphExecutionState(std::move(graph_def)));

  return std::make_unique<TfrtGraphExecutionState>(
      std::move(graph_execution_state));
}

namespace {

CallableOptions PopulateCallableOptions(
    CallableOptions& callable_options,
    const tensorflow::GraphImportConfig& graph_import_config) {
  // Configure pruning with the feed/fetch/target tensor names.
  callable_options.mutable_feed()->Reserve(graph_import_config.inputs.size());
  for (const auto& feed_tensor : graph_import_config.inputs) {
    callable_options.add_feed(feed_tensor.first);
  }
  callable_options.mutable_fetch()->Reserve(graph_import_config.outputs.size());
  for (const auto& fetch_tensor_name : graph_import_config.outputs) {
    callable_options.add_fetch(fetch_tensor_name);
  }
  callable_options.mutable_target()->Reserve(
      graph_import_config.control_outputs.size());
  for (const auto& target_tensor_name : graph_import_config.control_outputs) {
    callable_options.add_target(target_tensor_name);
  }

  return callable_options;
}

tensorflow::GraphDef CreateGraphDefFromGraphAndFlibDef(
    const tensorflow::Graph& graph,
    const tensorflow::FunctionLibraryDefinition& flib_def) {
  tensorflow::GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  *graph_def.mutable_library() = flib_def.ToProto();
  return graph_def;
}

// Creates a pruned graph from `graph_def` according to `callable_options`.
StatusOr<std::unique_ptr<tensorflow::Graph>> CreatePrunedGraph(
    tensorflow::GraphDef graph_def, const CallableOptions& callable_options) {
  VLOG(1) << "Creating pruned graph: " << callable_options.DebugString();

  // Prune the graph with `callable_options`. Although
  // grappler has model_pruner stage, it may leave v1 control flows in an
  // invalid state that cannot be functionalized. So we perform additional
  // pruning before functionalization.
  TF_RETURN_IF_ERROR(PruneGraphDef(graph_def, callable_options));

  if (VLOG_IS_ON(2)) {
    DumpGraphDefToFile("before_eliminate_ref_variables_graph_def", graph_def);
  }

  TF_RETURN_IF_ERROR(EliminateRefVariablesFromV1ControlFlow(graph_def));

  auto pruned_graph =
      std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
  tensorflow::GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(options, std::move(graph_def),
                                            pruned_graph.get()));
  return pruned_graph;
}

// Creates a new identity node to replace an operand of a given `node`.
NodeDef CreateNewIdentityNode(const NodeDef& node,
                              const std::string& input_name,
                              const std::string& identity_name) {
  NodeDef identity;
  identity.set_name(identity_name);
  identity.set_op("Identity");
  identity.add_input(input_name);
  identity.set_device(node.device());
  for (const auto& name_and_attr : node.attr()) {
    if (name_and_attr.first == "T") {
      identity.mutable_attr()->insert(name_and_attr);
      break;
    }
  }
  return identity;
}

}  // namespace

StatusOr<TfrtGraphExecutionState::OptimizationResult>
TfrtGraphExecutionState::CreateOptimizedGraph(
    const tensorflow::GraphImportConfig& graph_import_config) {
  OptimizationResult result;

  tensorflow::BuildGraphOptions build_graph_options;
  PopulateCallableOptions(build_graph_options.callable_options,
                          graph_import_config);

  auto graph_def = CreateGraphDefFromGraphAndFlibDef(graph(), flib_def());

  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("before_pruning", graph_def);
  }

  TF_ASSIGN_OR_RETURN(
      result.graph,
      CreatePrunedGraph(graph_def, build_graph_options.callable_options));
  DCHECK(result.graph);

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_pruning", *result.graph);
  }

  const auto functionalization_start_time = absl::Now();

  // Perform functionalization to convert v1 control flow to v2 control flow. It
  // should be applied to the unoptimized graph, because Grappler may cause
  // unfunctionalizablity.
  TF_RETURN_IF_ERROR(tensorflow::UpgradeLegacyGraph(
      result.graph.get(),
      const_cast<tensorflow::FunctionLibraryDefinition*>(
          &result.graph->flib_def()),
      /*restrict_functionalization_to_tpu_nodes=*/false));

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_functionalization", *result.graph);
  }

  auto grappler_start_time = absl::Now();
  result.functionalization_duration =
      grappler_start_time - functionalization_start_time;

  TF_RETURN_IF_ERROR(OptimizeGraph(result.graph, build_graph_options));

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_grappler", *result.graph);
  }

  result.grappler_duration = absl::Now() - grappler_start_time;

  return result;
}

namespace {

// Given an "Exit" node, finds its corresponding "LoopCond" node.
StatusOr<const NodeDef*> FindLoopCondFromExitNode(
    const NodeDef& exit_node,
    const absl::flat_hash_map<std::string, NodeDef*>& name_to_node) {
  const NodeDef* switch_node = nullptr;
  for (const std::string& tensor_name : exit_node.input()) {
    const std::string node_name = grappler::NodeName(tensor_name);
    if (!name_to_node.contains(node_name)) {
      return errors::InvalidArgument("Graph does not contain input ", node_name,
                                     " of exit node ", exit_node.name());
    }
    const NodeDef* node = name_to_node.at(node_name);
    if (node->op() == "Switch") {
      switch_node = node;
      break;
    }
  }
  if (switch_node == nullptr) {
    return errors::InvalidArgument("Exit node ", exit_node.name(),
                                   " does not have a Switch node as its ",
                                   "predecessor.");
  }
  for (const std::string& tensor_name : switch_node->input()) {
    const std::string node_name = grappler::NodeName(tensor_name);
    if (!name_to_node.contains(node_name)) {
      return errors::InvalidArgument("Graph does not contain input ", node_name,
                                     " of switch node ", switch_node->name());
    }

    const NodeDef* node = name_to_node.at(node_name);
    if (node->op() == "LoopCond") {
      return node;
    }
  }

  return errors::InvalidArgument("Switch node ", switch_node->name(),
                                 " does not have a LoopCond node as its ",
                                 "predecessor.");
}

}  // namespace

Status PruneGraphDef(GraphDef& graph_def,
                     const CallableOptions& callable_options) {
  // Gather node names and create a map from names to NodeDefs.
  absl::flat_hash_map<std::string, NodeDef*> name_to_node;
  // All exit nodes in order to track all while loops.
  absl::flat_hash_set<const NodeDef*> exit_nodes;
  for (auto& node : *graph_def.mutable_node()) {
    name_to_node[node.name()] = &node;
    if (node.op() == "Exit") {
      exit_nodes.insert(&node);
    }

    // TODO(tfrt-devs): Add support for _Send and _Recv ops.
    if (node.op() == "_Send" || node.op() == "_Recv") {
      return errors::InvalidArgument(
          "TFRT prune graphdef cannot handle graphs contains _Send and _Recv "
          "ops.");
    }
  }

  // Find all LoopCond -> Exit nodes mapping. So when we traverse to a LoopCond
  // node, we can add corresponding Exit nodes to the traversal queue in order
  // to maintain complete structure of a while loop.
  absl::flat_hash_map<const NodeDef*, absl::flat_hash_set<const NodeDef*>>
      loop_cond_to_exit_nodes;
  for (const NodeDef* exit_node : exit_nodes) {
    TF_ASSIGN_OR_RETURN(const NodeDef* loop_cond_node,
                        FindLoopCondFromExitNode(*exit_node, name_to_node));
    loop_cond_to_exit_nodes[loop_cond_node].insert(exit_node);
  }

  // `queue` is for candidate nodes we want to visit in the graph.
  std::vector<const NodeDef*> queue;

  // Add fetch nodes to the queue.
  absl::flat_hash_set<std::string> fetch_node_names;
  for (const std::string& tensor_name : callable_options.fetch()) {
    const NodeDef* node = name_to_node[grappler::NodeName(tensor_name)];
    if (!node) {
      return errors::InvalidArgument("Graph does not contain fetch node ",
                                     tensor_name, ".");
    }
    queue.push_back(node);
    fetch_node_names.insert(node->name());
  }

  // Add control target nodes to the queue.
  for (const std::string& tensor_name : callable_options.target()) {
    const NodeDef* node = name_to_node[grappler::NodeName(tensor_name)];
    if (!node) {
      return errors::InvalidArgument("Graph does not contain target node ",
                                     tensor_name, ".");
    }
    queue.push_back(node);
    fetch_node_names.insert(node->name());
  }

  absl::flat_hash_set<NodeDef*> feed_node_defs;

  // Add feed nodes to the queue. In addition, perform necessary rewrites to
  // remove unnecessary input edges.
  for (const std::string& tensor_name : callable_options.feed()) {
    NodeDef* node = name_to_node[grappler::NodeName(tensor_name)];
    if (!node) {
      return errors::InvalidArgument("Graph does not contain feed node ",
                                     tensor_name, ".");
    }

    // If a feed node is a Const, we don't need its inputs at all.
    //
    // TODO(tfrt-devs): Consider a general solution that we could just rewrite
    // all feed nodes to Placeholder nodes.
    if (node->op() == "Const") {
      node->clear_input();
    }

    queue.push_back(node);
    feed_node_defs.insert(node);
  }

  absl::flat_hash_set<const NodeDef*> visited;
  std::vector<NodeDef> keep;

  // Perform graph traversal to find out connected nodes from fetches.
  while (!queue.empty()) {
    const NodeDef* node = queue.back();
    queue.pop_back();

    if (!visited.insert(node).second) {
      continue;
    }

    keep.push_back(*node);
    if (node->op() == "LoopCond") {
      for (const NodeDef* exit_node : loop_cond_to_exit_nodes[node]) {
        queue.push_back(exit_node);
      }
    }

    for (const std::string& tensor_name : node->input()) {
      const NodeDef* in = name_to_node[grappler::NodeName(tensor_name)];
      if (!in) {
        return errors::InvalidArgument("Graph does not contain input ",
                                       grappler::NodeName(tensor_name),
                                       " of node ", node->name(), ".");
      }
      queue.push_back(in);
    }
  }

  graph_def.clear_node();
  for (auto& node : keep) {
    if (fetch_node_names.contains(node.name())) {
      // If the fetch node is an Exit op, we insert an Identity op right after
      // it and rename it to be the new fetch node. This is to prevent
      // functionalization from removing the fetch nodes.
      if (node.op() == "Exit") {
        auto renamed_exit_node = node;
        renamed_exit_node.set_name(
            absl::StrCat(renamed_exit_node.name(), "/tfrt_renamed"));
        node.set_op("Identity");
        *node.mutable_input(0) = renamed_exit_node.name();
        *graph_def.add_node() = std::move(renamed_exit_node);
      }
    }

    *graph_def.add_node() = std::move(node);
  }

  return Status::OK();
}

Status EliminateRefVariablesFromV1ControlFlow(tensorflow::GraphDef& graph_def) {
  auto* op_factory = OpRegistry::Global();

  absl::flat_hash_set<std::string> ref_nodes;
  for (const auto& node : graph_def.node()) {
    if (node.op() == "RefEnter" || node.op() == "RefSwitch") {
      ref_nodes.insert(node.name());
    }
  }

  tensorflow::GraphDef updated_graph_def;
  absl::flat_hash_set<std::string> new_identities;
  // Insert an identity node between each "RefEnter" or "RefSwitch" node and its
  // ref input. Then modify each "RefEnter"/"RefSwitch" node in-place to an
  // "Enter"/"Switch" node.
  for (auto& node : *graph_def.mutable_node()) {
    // First find the ref input name to this RefEnter or RefSwitch.
    std::string* ref_input_name = nullptr;
    if (node.op() == "RefEnter") {
      node.set_op("Enter");
      if (node.input_size() != 1) {
        return errors::InvalidArgument("RefEnter node ", node.name(),
                                       " does not have exactly 1 input.");
      }
      ref_input_name = node.mutable_input(0);
    } else if (node.op() == "RefSwitch") {
      node.set_op("Switch");
      if (node.input_size() != 2) {
        return errors::InvalidArgument("RefSwitch node", node.name(),
                                       " does not have exactly 2 inputs.");
      }
      ref_input_name = node.mutable_input(0);
    } else {
      // For other ops, check if their inputs are the ref ops we want to
      // eliminate, and if so, these ops must not require their inputs to be
      // refs.
      std::string ref_input;
      for (const auto& tensor_name : node.input()) {
        std::string input = grappler::NodeName(tensor_name);
        if (ref_nodes.contains(input)) {
          ref_input = std::move(input);
          break;
        }
      }
      if (!ref_input.empty()) {
        const OpDef* op_def;
        TF_RETURN_IF_ERROR(op_factory->LookUpOpDef(node.op(), &op_def));
        // TODO(tfrt-devs): How to match input_args to input names in NodeDef?
        for (const auto& input_arg : op_def->input_arg()) {
          if (input_arg.is_ref()) {
            return errors::Unimplemented(
                "Cannot in-place update ref node ", ref_input,
                " to the non-ref counterpart since its user node ", node.name(),
                " requires its input to be refs.");
          }
        }
      }
    }

    if (ref_input_name != nullptr) {
      std::string identity_name =
          absl::StrCat(grappler::NodeName(*ref_input_name), "/identity");
      if (!new_identities.contains(identity_name)) {
        *updated_graph_def.add_node() =
            CreateNewIdentityNode(node, *ref_input_name, identity_name);
        new_identities.insert(identity_name);
      }
      *ref_input_name = std::move(identity_name);
    }

    *updated_graph_def.add_node() = std::move(node);
  }

  graph_def.mutable_node()->Swap(updated_graph_def.mutable_node());
  return Status::OK();
}

Status TfrtGraphExecutionState::OptimizeGraph(
    std::unique_ptr<tensorflow::Graph>& graph,
    const tensorflow::BuildGraphOptions& build_graph_options) {
  std::unique_ptr<tensorflow::Graph> optimized_graph;
  std::unique_ptr<tensorflow::FunctionLibraryDefinition> optimized_flib;

  // Invoke Grappler to optimize the graph.
  auto status = graph_execution_state_->OptimizeGraph(
      build_graph_options, *graph, &graph->flib_def(), &optimized_graph,
      &optimized_flib);

  if (!status.ok()) {
    LOG(WARNING) << "TFRT failed to optimize graph: " << status;
    return tensorflow::Status::OK();
  }

  TF_RETURN_IF_ERROR(
      optimized_graph->AddFunctionLibrary(optimized_flib->ToProto()));
  graph = std::move(optimized_graph);
  return tensorflow::Status::OK();
}

}  // namespace tfrt_stub
}  // namespace tensorflow
