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

#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"
#include "tensorflow/core/grappler/optimizers/static_schedule.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Prefix added to nodes which are recomputed.
const char* kRecomputedNodePrefix = "Recomputed";
const char* kRecomputeTriggerNodePrefix = "RecomputeTrigger";
// Attribute which may be added to nodes to manually allow them to be
// recomputed.
const char* kRecomputeHint = "_recompute_hint";

// Ops which we wouldn't mind recomputing to save memory.
// TODO(allenl): Replace this list with a cost model.
std::unordered_set<string> GetCheapToRecomputeOps() {
  std::unordered_set<string> cheap_ops = {
      "Add",  "AddN",     "BiasAdd",           "Cast",
      "Fill", "FloorDiv", "FloorMod",          "FusedBatchNorm",
      "Mul",  "Neg",      "RealDiv",           "Reciprocal",
      "Relu", "Relu6",    "Reshape",           "Rsqrt",
      "Sqrt", "Square",   "SquaredDifference", "Sub",
      "Tile", "Transpose"};
  return cheap_ops;
}

// Find recomputable ops which feed into target nodes.
std::unordered_set<const NodeDef*> FindCandidateRecomputeNodes(
    const NodeMap& node_map, const GraphDef* graph,
    const std::function<bool(const NodeDef&)>& is_candidate,
    const std::function<bool(const NodeDef&)>& is_target) {
  std::unordered_set<const NodeDef*> candidate_recompute_nodes;
  for (const auto& node : graph->node()) {
    if (!is_candidate(node)) {
      continue;
    }
    bool has_target_output = false;
    for (const NodeDef* output : node_map.GetOutputs(node.name())) {
      // It only makes sense to recompute this if it feeds into a target
      // node. We expand this to dependencies in GetOpGroupsToRecompute.
      if (is_target(*output)) {
        has_target_output = true;
        break;
      }
    }
    if (!has_target_output) {
      continue;
    }
    bool has_target_input = false;
    for (const string& input_name : node.input()) {
      // Don't recompute nodes which depend on target nodes.
      const NodeDef* input_node = node_map.GetNode(input_name);
      if (is_target(*input_node)) {
        has_target_input = true;
        break;
      }
    }
    if (has_target_input) {
      continue;
    }
    candidate_recompute_nodes.insert(&node);
  }
  return candidate_recompute_nodes;
}

void connected_subgraph(const NodeMap& node_map, bool collect_inputs,
                        bool collect_outputs,
                        const std::function<bool(const NodeDef&)>& is_candidate,
                        std::unordered_set<const NodeDef*>* expanded_nodes) {
  std::queue<const NodeDef*> to_visit;
  for (const NodeDef* starting_node : *expanded_nodes) {
    to_visit.push(starting_node);
  }
  expanded_nodes->clear();
  while (!to_visit.empty()) {
    const NodeDef* current_node = to_visit.front();
    to_visit.pop();
    if (!expanded_nodes->insert(current_node).second) {
      // We already visited this node
      continue;
    }
    if (collect_inputs) {
      // Add inputs and outputs to this subgraph if they are candidates
      for (const string& input_name_raw : current_node->input()) {
        const NodeDef* input_node = node_map.GetNode(input_name_raw);
        if (expanded_nodes->count(input_node) == 0 &&
            is_candidate(*input_node)) {
          to_visit.push(input_node);
        }
      }
    }
    if (collect_outputs) {
      for (const NodeDef* output : node_map.GetOutputs(current_node->name())) {
        if (expanded_nodes->count(output) == 0 && is_candidate(*output)) {
          to_visit.push(output);
        }
      }
    }
  }
}

struct RecomputedSubGraph {
  std::unordered_set<const NodeDef*> recomputed_source_nodes;
  std::unordered_set<NodeDef*> target_nodes;
};

// Find groups of ops to recompute together based on `should_recompute`.
std::vector<RecomputedSubGraph> GetOpGroupsToRecompute(
    const GraphDef* graph, const NodeMap& node_map,
    const std::function<bool(const NodeDef&)>& should_recompute,
    const std::function<bool(const NodeDef&)>& is_target) {
  std::unordered_set<const NodeDef*> visited_nodes;
  std::vector<RecomputedSubGraph> subgraphs_to_recompute;
  std::unordered_set<const NodeDef*> candidate_recompute_nodes =
      FindCandidateRecomputeNodes(node_map, graph, should_recompute, is_target);
  for (const NodeDef* recompute_node : candidate_recompute_nodes) {
    if (visited_nodes.count(recompute_node) > 0) {
      continue;
    }
    RecomputedSubGraph current_recomputation;
    // Build out recomputation groups by expanding to inexpensive-to-recompute
    // nodes which do not feed target nodes. The goal is to capture some
    // intermediate activations within this graph.
    std::unordered_set<const NodeDef*> unpruned_recompute_nodes;
    unpruned_recompute_nodes.insert(recompute_node);
    connected_subgraph(node_map,
                       true,  // Collect inputs
                       true,  // Collect outputs
                       should_recompute, &unpruned_recompute_nodes);
    visited_nodes.insert(unpruned_recompute_nodes.begin(),
                         unpruned_recompute_nodes.end());
    for (const NodeDef* recompute_node : unpruned_recompute_nodes) {
      bool inserted_feed = false;
      for (NodeDef* output : node_map.GetOutputs(recompute_node->name())) {
        if (is_target(*output)) {
          current_recomputation.target_nodes.insert(output);
          if (!inserted_feed) {
            // Keep track of nodes which feed directly into a target node. These
            // and nodes which feed into them will define the recomputed
            // subgraph.
            current_recomputation.recomputed_source_nodes.insert(
                recompute_node);
            inserted_feed = true;
          }
        }
      }
    }
    // Recompute only nodes which eventually feed into a target node.
    connected_subgraph(node_map,
                       true,   // Collect inputs
                       false,  // Collect outputs
                       [&unpruned_recompute_nodes](const NodeDef& node) {
                         return unpruned_recompute_nodes.count(&node) != 0;
                       },
                       &current_recomputation.recomputed_source_nodes);
    if (current_recomputation.target_nodes.empty()) {
      continue;
    }
    subgraphs_to_recompute.push_back(current_recomputation);
  }
  return subgraphs_to_recompute;
}

// Computes the maximum topological numbers of (1) target node components
// (gradient nodes being fed by the recomputation), and (2) child recompute node
// components for each recomputed node. We will not attach any control
// dependencies to a recomputation unless they have component numbers greater
// than this value (to prevent cycles).
std::unordered_map<const NodeDef*, int> GetMaxDownstreamComponents(
    const std::unordered_set<const NodeDef*>& recomputed_source_nodes,
    const std::unordered_set<NodeDef*>& target_nodes, const NodeMap& node_map,
    const std::unordered_map<const NodeDef*, int>& components) {
  std::unordered_map<const NodeDef*, int> recomputed_node_components;
  // Start by setting component numbers to the maximum among target nodes.
  for (const NodeDef* original_recompute_node : recomputed_source_nodes) {
    int max_target_component = -1;
    for (NodeDef* output :
         node_map.GetOutputs(original_recompute_node->name())) {
      if (target_nodes.count(output) != 0) {
        int current_target_component = components.find(output)->second;
        if (current_target_component > max_target_component) {
          max_target_component = current_target_component;
        }
      }
    }
    if (max_target_component > -1) {
      recomputed_node_components[original_recompute_node] =
          max_target_component;
    }
  }
  // Sort recomputed nodes topologically (based on the original graph) so we can
  // efficiently assign to each node the maximum of its recomputed child
  // components and its own targets.
  std::vector<const NodeDef*> recomputed_source_nodes_topological(
      recomputed_source_nodes.begin(), recomputed_source_nodes.end());
  std::sort(recomputed_source_nodes_topological.begin(),
            recomputed_source_nodes_topological.end(),
            [&components](const NodeDef* first, const NodeDef* second) {
              return components.find(first)->second <
                     components.find(second)->second;
            });
  for (const NodeDef* original_recompute_node :
       recomputed_source_nodes_topological) {
    int max_component;
    auto recomputed_component_iterator =
        recomputed_node_components.find(original_recompute_node);
    if (recomputed_component_iterator != recomputed_node_components.end()) {
      max_component = recomputed_component_iterator->second;
    } else {
      max_component = -1;
    }
    for (NodeDef* output :
         node_map.GetOutputs(original_recompute_node->name())) {
      if (recomputed_source_nodes.count(output) == 0) {
        continue;
      }
      auto child_component_iterator = recomputed_node_components.find(output);
      CHECK(child_component_iterator != recomputed_node_components.end());
      int child_component = child_component_iterator->second;
      if (child_component > max_component) {
        max_component = child_component;
      }
    }
    CHECK_GE(max_component, 0);
    recomputed_node_components[original_recompute_node] = max_component;
  }
  return recomputed_node_components;
}

// Modifies `graph`, adding trigger nodes and returning a mapping from
// `recomputed_source_nodes` to trigger nodes which will not create loops in the
// graph (using the component numberings in `components` and
// `recomputed_node_max_feed_components`). The copied nodes (not the nodes in
// recomputed_source_nodes, which are the originals) eventually get these
// control dependencies.
std::unordered_map<const NodeDef*, const NodeDef*>
AddRecomputeControlDependencyNodes(
    const std::unordered_set<const NodeDef*>& recomputed_source_nodes,
    const std::unordered_set<NodeDef*>& target_nodes, const NodeMap& node_map,
    const std::unordered_map<const NodeDef*, int>& components,
    const std::unordered_map<const NodeDef*, int>&
        recomputed_node_max_feed_components,
    GraphDef* graph) {
  // Sort recomputed nodes based on max downstream components.
  std::vector<const NodeDef*> recomputed_source_nodes_topological(
      recomputed_source_nodes.begin(), recomputed_source_nodes.end());
  std::sort(recomputed_source_nodes_topological.begin(),
            recomputed_source_nodes_topological.end(),
            [&recomputed_node_max_feed_components](const NodeDef* first,
                                                   const NodeDef* second) {
              int first_component =
                  recomputed_node_max_feed_components.find(first)->second;
              int second_component =
                  recomputed_node_max_feed_components.find(second)->second;
              return first_component > second_component
                     // Ensure a consistent ordering. This is necessary because
                     // we're working not with node component numbers (which are
                     // unique) but with the maximum across nodes they feed into
                     // (very much not unique).
                     || (first_component == second_component &&
                         first->name() > second->name());
            });
  // Create merged control dependency nodes by sorting target inputs
  // topologically and zipper merging with the sorted recomputed nodes.
  std::vector<const NodeDef*> target_inputs_topological;
  for (const NodeDef* target_node : target_nodes) {
    for (const string& target_input_name_raw : target_node->input()) {
      const NodeDef* target_input = node_map.GetNode(target_input_name_raw);
      // If this node has already had one of its inputs recomputed during this
      // rewriting pass, we ignore that recomputed node here (it will not be in
      // the NodeMap).
      if (target_input == nullptr ||
          recomputed_source_nodes.count(target_input) != 0 ||
          components.find(target_node)->second ==
              components.find(target_input)->second) {
        continue;
      }
      target_inputs_topological.push_back(target_input);
    }
  }
  std::sort(target_inputs_topological.begin(), target_inputs_topological.end(),
            [&components](const NodeDef* first, const NodeDef* second) {
              return components.find(first)->second >
                     components.find(second)->second;
            });
  auto target_input_iterator = target_inputs_topological.begin();
  NodeDef* current_trigger_node = nullptr;
  std::unordered_map<const NodeDef*, const NodeDef*> triggers;
  for (const NodeDef* original_recomputed_node :
       recomputed_source_nodes_topological) {
    NodeDef* new_trigger_node = graph->add_node();
    new_trigger_node->set_name(AddPrefixToNodeName(
        original_recomputed_node->name(), kRecomputeTriggerNodePrefix));
    new_trigger_node->set_op("NoOp");
    new_trigger_node->set_device(original_recomputed_node->device());
    if (current_trigger_node != nullptr) {
      *new_trigger_node->add_input() =
          strings::StrCat("^", current_trigger_node->name());
    }
    current_trigger_node = new_trigger_node;
    triggers[original_recomputed_node] = current_trigger_node;
    for (;
         target_input_iterator != target_inputs_topological.end() &&
         components.find(*target_input_iterator)->second >
             recomputed_node_max_feed_components.find(original_recomputed_node)
                 ->second;
         ++target_input_iterator) {
      *current_trigger_node->add_input() =
          strings::StrCat("^", (*target_input_iterator)->name());
      VLOG(2) << "  Recomputation trigger " << current_trigger_node->name()
              << " depends on " << (*target_input_iterator)->name();
    }
  }
  return triggers;
}

string RecomputedOrOriginalNodeName(
    const std::unordered_set<string>& recomputed_node_names,
    const string& original_node_name) {
  if (recomputed_node_names.find(original_node_name) ==
      recomputed_node_names.end()) {
    return original_node_name;
  } else {
    return AddPrefixToNodeName(original_node_name, kRecomputedNodePrefix);
  }
}

// Helper function to recompute a sub-graph (recomputed_source_nodes). Edges
// from recomputed_source_nodes to target_nodes are changed to start from the
// recomputed nodes.
void RecomputeSubgraph(
    const std::unordered_set<const NodeDef*>& recomputed_source_nodes,
    const std::unordered_set<NodeDef*>& target_nodes, const NodeMap& node_map,
    const std::unordered_map<const NodeDef*, int>& components,
    GraphDef* graph) {
  std::unordered_set<string> recomputed_node_names;
  VLOG(1) << "Recomputing a " << recomputed_source_nodes.size()
          << " node subgraph";
  std::unordered_map<const NodeDef*, int> recomputed_node_components =
      GetMaxDownstreamComponents(recomputed_source_nodes, target_nodes,
                                 node_map, components);
  for (const NodeDef* original_node : recomputed_source_nodes) {
    VLOG(2) << "  " << original_node->name();
    recomputed_node_names.insert(original_node->name());
  }
  std::unordered_map<const NodeDef*, const NodeDef*> triggers =
      AddRecomputeControlDependencyNodes(recomputed_source_nodes, target_nodes,
                                         node_map, components,
                                         recomputed_node_components, graph);
  // Create the recomputed sub-graph
  for (const NodeDef* original_node : recomputed_source_nodes) {
    NodeDef* copied_node = graph->add_node();
    copied_node->set_name(
        AddPrefixToNodeName(original_node->name(), kRecomputedNodePrefix));
    copied_node->set_op(original_node->op());
    *copied_node->mutable_attr() = original_node->attr();
    copied_node->set_device(original_node->device());
    for (const string& original_input_name : original_node->input()) {
      // Set inputs which are internal to the copied subgraph to their copied
      // versions.
      *copied_node->add_input() = RecomputedOrOriginalNodeName(
          recomputed_node_names, original_input_name);
    }
    // Each recomputed node gets a control dependency to prevent it from being
    // recomputed immediately.
    *copied_node->add_input() =
        strings::StrCat("^", triggers[original_node]->name());
  }
  // Set the inputs of nodes in the target subgraph to the recomputed nodes
  // where applicable.
  for (NodeDef* target_node : target_nodes) {
    for (string& target_input_name : *target_node->mutable_input()) {
      target_input_name = RecomputedOrOriginalNodeName(recomputed_node_names,
                                                       target_input_name);
    }
  }
}

void RecomputationRewritingPass(RewriterConfig::MemOptType optimization_level,
                                const string& recomputation_targets_name_prefix,
                                GraphDef* graph, const GrapplerItem& item) {
  // The topological numberings and NodeMap will be stale as soon as we start
  // modifying the graph in RecomputeSubgraph. However, RecomputeSubgraph only
  // looks up nodes which were in the original graph, and preserves the graph
  // topology it's interested in.
  // We don't use the results of this topological sort until later, but this
  // call invalidates all NodeDef pointers, so it needs to be done before we
  // start collecting those.
  TopologicalSort(graph);
  NodeMap node_map(graph);
  std::vector<RecomputedSubGraph> recomputed_subgraphs;
  // Do not recompute nodes which are fed, since the recomputed node would not
  // take on the fed value (i.e. gradients would be incorrect).
  std::unordered_set<string> feeds;
  for (const auto& feed : item.feed) {
    feeds.insert(NodeName(feed.first));
  }
  std::function<bool(const NodeDef&)> is_target =
      [&recomputation_targets_name_prefix](const NodeDef& node) {
        // Nodes whose inputs we may want to recompute. Typically targets will
        // be gradients (recomputation_targets_name_prefix="gradients/"),
        // although the prefix is configurable since gradients may be created in
        // a name scope.
        // TODO(allenl): Use a static schedule
        // (grappler::EstimateEarliestExecutionTimes) to recompute only nodes
        // whose outputs will sit around for a while.
        return node.name().find(recomputation_targets_name_prefix) == 0;
      };
  if (optimization_level == RewriterConfig::HEURISTICS) {
    // TODO(allenl): Handle ResNet-like architectures better. Right now all of
    // the cheap forward ops get grouped into a single subgraph which must
    // execute before gradients start executing (unless layers are manually
    // separated by identity ops).
    std::unordered_set<string> cheap_to_recompute_ops =
        GetCheapToRecomputeOps();
    recomputed_subgraphs = GetOpGroupsToRecompute(
        graph, node_map,
        [&cheap_to_recompute_ops, &feeds, &is_target](const NodeDef& node) {
          return !is_target(node) && feeds.count(node.name()) == 0 &&
                 (cheap_to_recompute_ops.count(node.op()) > 0 ||
                  node.attr().count(kRecomputeHint) > 0);
        },
        is_target);
  } else if (optimization_level == RewriterConfig::MANUAL) {
    recomputed_subgraphs = GetOpGroupsToRecompute(
        graph, node_map,
        [&feeds, &is_target](const NodeDef& node) {
          return !is_target(node) && feeds.count(node.name()) == 0 &&
                 node.attr().count(kRecomputeHint) > 0;
        },
        is_target);
  }
  if (!recomputed_subgraphs.empty()) {
    std::unordered_map<const NodeDef*, int> topological_numbering;
    for (int node_number = 0; node_number < graph->node().size();
         ++node_number) {
      topological_numbering[graph->mutable_node(node_number)] =
          graph->node().size() - node_number - 1;
    }
    // Duplicate the indicated sub-graphs and set up control dependencies
    for (const RecomputedSubGraph& subgraph : recomputed_subgraphs) {
      RecomputeSubgraph(subgraph.recomputed_source_nodes, subgraph.target_nodes,
                        node_map, topological_numbering, graph);
    }
  }
}

std::pair<NodeDef*, NodeDef*> BuildSwapPair(NodeDef* node, int input_to_swap,
                                            GraphDef* graph) {
  string tensor_to_swap = strings::StrCat(node->name(), "_", input_to_swap);

  // Force the tensor to be copied to cpu.
  NodeDef* swap_out_node = graph->add_node();
  swap_out_node->set_name(strings::StrCat("swap_out_", tensor_to_swap));
  swap_out_node->set_op("Identity");
  swap_out_node->set_device("/CPU");

  // Force the tensor to be restored to the device.
  NodeDef* swap_in_node = graph->add_node();
  swap_in_node->set_name(strings::StrCat("swap_in_", tensor_to_swap));
  swap_in_node->set_op("Identity");
  *swap_in_node->add_input() = swap_out_node->name();

  // Colocate the swap_in_ node with the node itself.
  string coloc_group = strings::StrCat("loc@", tensor_to_swap);
  (*swap_in_node->mutable_attr())["_class"].mutable_list()->add_s(coloc_group);
  (*node->mutable_attr())["_class"].mutable_list()->add_s(coloc_group);

  const DataType input_type = node->attr().at("T").type();
  (*swap_in_node->mutable_attr())["T"].set_type(input_type);
  (*swap_out_node->mutable_attr())["T"].set_type(input_type);
  return std::make_pair(swap_out_node, swap_in_node);
}

static int64 EstimateSize(const OpInfo::TensorProperties& t) {
  DataType dtype = t.dtype();
  int64 size = DataTypeSize(dtype);
  TensorShapeProto shape = t.shape();
  if (shape.unknown_rank()) {
    // Can't infer the size if the rank is unknown. It has to be at least a
    // scalar though.
    return size;
  }
  // If one of the dimensions is unknown statically, assume it's at least one.
  for (int i = 0; i < shape.dim_size(); ++i) {
    if (shape.dim(i).size() < 0) {
      shape.mutable_dim(i)->set_size(1);
    }
  }
  int64 num_elems = TensorShape(shape).num_elements();
  return num_elems * size;
}

struct SwapInfo {
  std::vector<int> inputs_to_swap;
  Costs::NanoSeconds time_to_swap = 0;
};

static const NodeDef* FindSwapTrigger(
    const NodeDef* node, const SwapInfo& swap_info,
    const std::unordered_map<string, const NodeDef*>& name_map,
    const std::unordered_map<const NodeDef*, Costs::NanoSeconds>&
        execution_times) {
  // max_trigger_time stores the time before which the swap operation needs to
  // be started in order to load the data back onto the accelerator without
  // delaying the downstream computation.
  Costs::NanoSeconds max_trigger_time(0);
  std::set<string> possible_inputs;
  for (int i = 0; i < node->input_size(); ++i) {
    const string input_node_name = NodeName(node->input(i));
    auto it1 = name_map.find(input_node_name);
    if (it1 == name_map.end()) {
      return nullptr;
    }
    const NodeDef* input_node = it1->second;

    auto it2 = execution_times.find(input_node);
    if (it2 == execution_times.end()) {
      return nullptr;
    }
    max_trigger_time = std::max(max_trigger_time, it2->second);
    possible_inputs.insert(input_node_name);
  }

  for (const int i : swap_info.inputs_to_swap) {
    const string input_node_name = NodeName(node->input(i));
    possible_inputs.erase(input_node_name);
  }
  if (possible_inputs.empty()) {
    return nullptr;
  }

  max_trigger_time -= swap_info.time_to_swap;

  std::map<Costs::NanoSeconds, const NodeDef*> candidates;
  while (!possible_inputs.empty()) {
    const string input_node_name = *possible_inputs.begin();
    possible_inputs.erase(possible_inputs.begin());
    auto it1 = name_map.find(input_node_name);
    if (it1 == name_map.end()) {
      return nullptr;
    }
    const NodeDef* input_node = it1->second;
    // Don't jump over frames, since adding a control dependency from one frame
    // to the next isn't supported. Don't go through branches, since we don't
    // know whether they'll be executed or not.
    if (IsNextIteration(*input_node) || IsSwitch(*input_node) ||
        IsMerge(*input_node)) {
      continue;
    }
    auto it2 = execution_times.find(input_node);
    if (it2 == execution_times.end()) {
      return nullptr;
    }
    if (it2->second < max_trigger_time) {
      candidates[it2->second] = input_node;
    } else {
      for (const string& fanin : input_node->input()) {
        possible_inputs.insert(NodeName(fanin));
      }
    }
  }

  // Select the candidate that will execute last, since we want to swap the data
  // back at the last minute while still allowing enough time for data to be
  // swapped back timely to feed the downstream nodes.
  if (!candidates.empty()) {
    return candidates.rbegin()->second;
  }
  return nullptr;
}

Status MemoryOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph) {
  *optimized_graph = item.graph;

  RecomputationRewritingPass(optimization_level_,
                             recomputation_targets_name_prefix_,
                             optimized_graph, item);

  // Figure out what needs to be swapped;
  std::unordered_map<NodeDef*, SwapInfo> nodes_to_swap;
  for (auto& node : *optimized_graph->mutable_node()) {
    if (node.attr().count("_swap_to_host") != 0) {
      SwapInfo& swap_info = nodes_to_swap[&node];
      const AttrValue& val = node.attr().at("_swap_to_host");
      if (val.has_list()) {
        for (int64 input_id : val.list().i()) {
          swap_info.inputs_to_swap.push_back(input_id);
        }
      } else {
        int64 input_id = val.i();
        swap_info.inputs_to_swap.push_back(input_id);
      }
    }
  }
  if (nodes_to_swap.empty()) {
    // Nothing to do.
    return Status::OK();
  }

  {
    // Estimate the size of the data to swap for each node.
    GraphProperties properties(item);
    TF_RETURN_IF_ERROR(properties.InferStatically());
    for (auto& swap : nodes_to_swap) {
      const NodeDef* node = swap.first;
      std::vector<OpInfo::TensorProperties> props =
          properties.GetInputProperties(node->name());
      SwapInfo& swap_info = swap.second;
      int64 bytes_to_swap = 0;
      for (int64 input_id : swap_info.inputs_to_swap) {
        const OpInfo::TensorProperties& t = props[input_id];
        bytes_to_swap += EstimateSize(t);
      }
      // Let's assume we're going to swap over PCIe running at 16 GBps.
      swap_info.time_to_swap = bytes_to_swap / 16;
    }
  }

  std::unordered_map<const NodeDef*, Costs::NanoSeconds> execution_times;
  TF_RETURN_IF_ERROR(
      EstimateEarliestExecutionTimes(item, cluster, &execution_times));

  std::unordered_map<string, const NodeDef*> name_map;
  for (const auto& node : item.graph.node()) {
    name_map[node.name()] = &node;
  }

  for (auto& swap : nodes_to_swap) {
    NodeDef* node = swap.first;
    SwapInfo& swap_info = swap.second;

    // Make sure the tensor isn't swapped back in right away: look for node that
    // will execute just before we need to swap the data back, and add a control
    // dependency from that node to the swap node.
    const NodeDef* trigger =
        FindSwapTrigger(node, swap_info, name_map, execution_times);
    if (!trigger) {
      continue;
    }
    // Swap all the tensors that are marked with the 'swap_to_host' attribute.
    for (int input_id : swap_info.inputs_to_swap) {
      std::pair<NodeDef*, NodeDef*> swap_nodes =
          BuildSwapPair(node, input_id, optimized_graph);
      *swap_nodes.first->add_input() = node->input(input_id);
      *node->mutable_input(input_id) = swap_nodes.second->name();

      // Add the control dependency needed to delay the execution of the swap.
      *swap_nodes.second->add_input() = strings::StrCat("^", trigger->name());
    }
  }

  return Status::OK();
}

void MemoryOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimized_graph, double result) {
  // Nothing to do for MemoryOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
