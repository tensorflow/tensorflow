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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"
#include "tensorflow/core/grappler/optimizers/static_schedule.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/math/math_util.h"
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
      "Add",      "AddN",       "BiasAdd",        "Cast",   "Fill",
      "FloorDiv", "FloorMod",   "FusedBatchNorm", "Mul",    "Neg",
      "RealDiv",  "Reciprocal", "Relu",           "Relu6",  "Reshape",
      "Rsqrt",    "Sigmoid",    "Sqrt",           "Square", "SquaredDifference",
      "Sub",      "Tile",       "Transpose"};
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
                                const string& recomputation_targets_name_scope,
                                GraphDef* graph, const GrapplerItem& item) {
  if (optimization_level != RewriterConfig::RECOMPUTATION_HEURISTICS &&
      optimization_level != RewriterConfig::HEURISTICS &&
      optimization_level != RewriterConfig::MANUAL) {
    // Nothing to do
    return;
  }
  // The topological numberings and NodeMap will be stale as soon as we start
  // modifying the graph in RecomputeSubgraph. However, RecomputeSubgraph only
  // looks up nodes which were in the original graph, and preserves the graph
  // topology it's interested in.
  // We don't use the results of this topological sort until later, but this
  // call invalidates all NodeDef pointers, so it needs to be done before we
  // start collecting those.
  TF_CHECK_OK(TopologicalSort(graph));
  NodeMap node_map(graph);
  std::vector<RecomputedSubGraph> recomputed_subgraphs;
  // Do not recompute nodes which are fed, since the recomputed node would not
  // take on the fed value (i.e. gradients would be incorrect).
  std::unordered_set<string> feeds;
  for (const auto& feed : item.feed) {
    feeds.insert(NodeName(feed.first));
  }
  std::function<bool(const NodeDef&)> is_target =
      [&recomputation_targets_name_scope](const NodeDef& node) {
        // Nodes whose inputs we may want to recompute. This matches node names
        // that contain recomputation_targets_name_scope as a name scope,
        // meaning it either begins with or contains the name scope.
        // Defaults to "gradients/" which will match any node names that begins
        // with "gradients/" or contains "/gradients/".
        return node.name().find(recomputation_targets_name_scope) == 0 ||
               node.name().find("/" + recomputation_targets_name_scope) != -1;
      };

  if (optimization_level == RewriterConfig::RECOMPUTATION_HEURISTICS ||
      optimization_level == RewriterConfig::HEURISTICS) {
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

bool SchedulingPass(Cluster* cluster, GrapplerItem* item) {
  // Look for AddN nodes (and equivalent) and record input names.
  GraphView view(&item->graph);

  std::unordered_map<string, std::unordered_set<NodeDef*>> addn_list;
  for (NodeDef& node : *item->graph.mutable_node()) {
    if (!IsAddN(node) && node.op() != "AccumulateNV2") {
      continue;
    }
    // There is nothing to gain by optimizing nodes with 2 or fewer inputs.
    if (view.NumFanins(node, false) <= 2) {
      continue;
    }
    for (const auto& input : view.GetFanins(node, false)) {
      if (input.node->device() == node.device()) {
        string tensor_name =
            strings::StrCat(input.node->name(), ":", input.port_id);
        addn_list[tensor_name].insert(&node);
      }
    }
  }

  if (addn_list.empty()) {
    return false;
  }

  GraphMemory memory(*item);
  const std::unordered_map<string, DeviceProperties>& devices =
      cluster->GetDevices();
  Status s = memory.InferStatically(devices);
  if (!s.ok()) {
    VLOG(1) << "Failed to infer memory usage: " << s.error_message();
    return false;
  }

  std::unordered_set<NodeDef*> addn_to_rewrite;
  for (const auto& device : devices) {
    const string& name = device.first;
    const DeviceProperties& prop = device.second;
    if (prop.memory_size() <= 0) {
      VLOG(1) << "Available memory unknown for device " << name;
      continue;
    }
    const GraphMemory::MemoryUsage& mem_usage = memory.GetPeakMemoryUsage(name);

    if (mem_usage.used_memory <= prop.memory_size() * 0.8) {
      continue;
    }

    for (const auto& live : mem_usage.live_tensors) {
      string tensor_name = strings::StrCat(live.node, ":", live.output_id);
      auto it = addn_list.find(tensor_name);
      if (it != addn_list.end()) {
        addn_to_rewrite.insert(it->second.begin(), it->second.end());
      }
    }
  }

  if (addn_to_rewrite.empty()) {
    return false;
  }
  GraphProperties properties(*item);
  s = properties.InferStatically(false);
  if (!s.ok()) {
    VLOG(1) << "Failed to infer shapes: " << s.error_message();
    return false;
  }

  bool updated_graph = false;
  // Rewrite the AddN.
  for (NodeDef* node : addn_to_rewrite) {
    if (!properties.HasOutputProperties(node->name())) {
      VLOG(1) << "Missing properties for " << node->name();
      continue;
    }
    const TensorShapeProto& shape =
        properties.GetOutputProperties(node->name())[0].shape();
    PartialTensorShape shp(shape);
    if (!shp.IsFullyDefined()) {
      VLOG(1) << "Shape not fully known for " << node->name();
      continue;
    }

    // Compute a topological ordering for the node fanin.
    std::unordered_map<NodeDef*, int> topo_order;
    ReverseDfs(view, {node}, nullptr,
               [&topo_order](NodeDef* n) {
                 int topo_index = topo_order.size();
                 topo_order[n] = topo_index;
               },
               nullptr);

    std::vector<int> input_topo_index;

    for (int i = 0; i < node->input_size(); ++i) {
      const string& input = node->input(i);
      const string node_name = NodeName(input);
      NodeDef* node = view.GetNode(node_name);
      input_topo_index.push_back(topo_order.at(node));
    }
    int min_input_topo_index = INT_MAX;
    int min_input_id = -1;
    for (int i = 0; i < node->input_size(); ++i) {
      if (IsControlInput(node->input(i))) {
        // control inputs are always last.
        break;
      }
      const int current = input_topo_index[i];
      if (current < min_input_topo_index) {
        min_input_topo_index = current;
        min_input_id = i;
      }
    }
    CHECK_LE(0, min_input_id);
    std::vector<string> pre_ctrl_deps;
    std::vector<string> post_ctrl_deps;
    for (int i = node->input_size() - 1; i >= 0; --i) {
      if (!IsControlInput(node->input(i))) {
        // control inputs are always last.
        break;
      }
      if (input_topo_index[i] < min_input_topo_index) {
        // These control dependencies can be executed before the node.
        pre_ctrl_deps.push_back(node->input(i));
      } else {
        // These control dependencies should be executed after the node.
        post_ctrl_deps.push_back(node->input(i));
      }
    }

    DataType dtype = node->attr().at("T").type();
    const string& device = node->device();

    // Create the temporary variable that will hold intermediate results
    NodeDef* tmp_var = item->graph.add_node();
    tmp_var->set_name(strings::StrCat(node->name(), "/tmp_var"));
    tmp_var->set_op("TemporaryVariable");
    tmp_var->set_device(device);
    (*tmp_var->mutable_attr())["dtype"].set_type(dtype);
    *(*tmp_var->mutable_attr())["shape"].mutable_shape() = shape;
    (*tmp_var->mutable_attr())["var_name"].set_s(tmp_var->name());

    for (const string& ctrl_dep : pre_ctrl_deps) {
      *tmp_var->add_input() = ctrl_dep;
    }
    *tmp_var->add_input() =
        AsControlDependency(NodeName(node->input(min_input_id)));

    // Initialize it to zero
    NodeDef* zeros = item->graph.add_node();
    zeros->set_name(strings::StrCat(node->name(), "/tmp_var_zeros"));
    zeros->set_op("ZerosLike");
    zeros->set_device(device);
    (*zeros->mutable_attr())["T"].set_type(dtype);
    *zeros->add_input() = node->input(min_input_id);

    NodeDef* initialize = item->graph.add_node();
    initialize->set_name(strings::StrCat(node->name(), "/tmp_var_initializer"));
    initialize->set_op("Assign");
    initialize->set_device(device);
    (*initialize->mutable_attr())["T"].set_type(dtype);
    (*initialize->mutable_attr())["use_locking"].set_b(false);
    (*initialize->mutable_attr())["validate_shape"].set_b(false);
    *initialize->add_input() = tmp_var->name();
    *initialize->add_input() = zeros->name();

    // Add the assignadd nodes
    std::vector<NodeDef*> accumulates;
    for (int i = 0; i < node->input_size(); ++i) {
      const string& input = node->input(i);
      if (!IsControlInput(input)) {
        NodeDef* accumulate = item->graph.add_node();
        accumulate->set_name(
            strings::StrCat(node->name(), "/tmp_var_accum_", i));
        accumulate->set_op("AssignAdd");
        accumulate->set_device(device);
        (*accumulate->mutable_attr())["T"].set_type(dtype);
        (*accumulate->mutable_attr())["use_locking"].set_b(true);
        *accumulate->add_input() = initialize->name();
        *accumulate->add_input() = input;
        accumulates.push_back(accumulate);
      }
    }

    // Rewrite the AddN node as a DestroyTemporaryVariable ops
    node->set_op("DestroyTemporaryVariable");
    node->clear_input();
    node->clear_attr();
    (*node->mutable_attr())["T"].set_type(dtype);
    (*node->mutable_attr())["var_name"].set_s(tmp_var->name());
    *node->add_input() = initialize->name();
    for (const NodeDef* accum : accumulates) {
      *node->add_input() = AsControlDependency(accum->name());
    }
    for (const string& ctrl_dep : post_ctrl_deps) {
      *node->add_input() = ctrl_dep;
    }

    updated_graph = true;
  }

  return updated_graph;
}

Status BuildSwapPair(NodeDef* node, int input_to_swap,
                     const std::unordered_map<string, const NodeDef*>& name_map,
                     GraphDef* graph,
                     std::pair<NodeDef*, NodeDef*>* swap_pair) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node->op(), &op_def));
  DataType input_type;
  TF_RETURN_IF_ERROR(
      InputTypeForNode(*node, *op_def, input_to_swap, &input_type));
  if (IsRefType(input_type)) {
    return errors::InvalidArgument("Can't swap input ", input_to_swap,
                                   " of node ", node->name(),
                                   " since it expects a reference");
  }

  string tensor_to_swap = strings::StrCat(node->name(), "_", input_to_swap);
  string swap_out_name = strings::StrCat("swap_out_", tensor_to_swap);
  string swap_in_name = strings::StrCat("swap_in_", tensor_to_swap);
  if (name_map.find(swap_out_name) != name_map.end() ||
      name_map.find(swap_in_name) != name_map.end()) {
    return errors::InvalidArgument("Input ", input_to_swap, " of node ",
                                   node->name(), " is already swapped");
  }

  // Force the tensor to be copied to cpu.
  NodeDef* swap_out_node = graph->add_node();
  swap_out_node->set_name(swap_out_name);
  swap_out_node->set_op("_CopyFromGpuToHost");

  // Force the tensor to be restored to the device.
  NodeDef* swap_in_node = graph->add_node();
  swap_in_node->set_name(swap_in_name);
  swap_in_node->set_op("_CopyFromHostToGpu");
  *swap_in_node->add_input() = swap_out_node->name();

  // Colocate the swap_out_ and swap_in_ nodes with the node itself.
  swap_out_node->set_device(node->device());
  swap_in_node->set_device(node->device());
  string coloc_group = strings::StrCat("loc@", tensor_to_swap);
  (*swap_out_node->mutable_attr())["_class"].mutable_list()->add_s(coloc_group);
  (*swap_in_node->mutable_attr())["_class"].mutable_list()->add_s(coloc_group);
  (*node->mutable_attr())["_class"].mutable_list()->add_s(coloc_group);

  (*swap_in_node->mutable_attr())["T"].set_type(input_type);
  (*swap_out_node->mutable_attr())["T"].set_type(input_type);
  *swap_pair = std::make_pair(swap_out_node, swap_in_node);

  return Status::OK();
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

static const NodeDef* FindSwapInTrigger(
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
  std::set<string> already_processed;

  while (!possible_inputs.empty()) {
    const string input_node_name = *possible_inputs.begin();
    possible_inputs.erase(possible_inputs.begin());
    already_processed.insert(input_node_name);
    auto it1 = name_map.find(input_node_name);
    if (it1 == name_map.end()) {
      return nullptr;
    }
    const NodeDef* input_node = it1->second;
    // Don't jump over frames, since adding a control dependency from one frame
    // to the next isn't supported. Don't go through branches, since we don't
    // know whether they'll be executed or not.
    if (ModifiesFrameInfo(*input_node) || IsSwitch(*input_node) ||
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
        string name = NodeName(fanin);
        if (already_processed.find(name) == already_processed.end()) {
          possible_inputs.insert(name);
        }
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

static bool IsSwappable(const GraphView& graph, GraphView::OutputPort output) {
  const NodeDef& node = *output.node;
  // There is no point in swapping out persistent tensors, since the tensor will
  // continue to use memory.
  if (IsPersistent(node)) {
    return false;
  }

  const OpDef* op_def;
  if (!OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok()) {
    return false;
  }
  DataType dtype;
  if (!OutputTypeForNode(node, *op_def, output.port_id, &dtype).ok()) {
    return false;
  }
  // References can only refer to persistent memory: therefore the node isn't
  // swappable.
  if (IsRefType(dtype)) {
    return false;
  }

  if (output.node->op() == "Identity" || output.node->op() == "Reshape") {
    // If placed on the same device, these nodes are just forwarding references
    // to their input. Therefore they are swappable iff their fanin is swappable
    // or it resides on a different device.
    GraphView::InputPort input;
    input.node = output.node;
    input.port_id = 0;
    GraphView::OutputPort fanin = graph.GetRegularFanin(input);
    if (fanin.node->device() == node.device()) {
      return IsSwappable(graph, fanin);
    }
  }
  return true;
}

static NodeDef* FindSwapOutTrigger(
    const NodeDef* node, int input_id, const GraphView& view,
    const std::unordered_map<const NodeDef*, Costs::NanoSeconds>&
        execution_times) {
  // Find the output port that generated the tensor to swap.
  GraphView::InputPort swap;
  swap.node = const_cast<NodeDef*>(node);
  swap.port_id = input_id;
  GraphView::OutputPort generator = view.GetRegularFanin(swap);
  if (!generator.node) {
    return nullptr;
  }

  const std::unordered_set<GraphView::InputPort, GraphView::HashPort>& fanout =
      view.GetFanout(generator);
  NodeDef* trigger = nullptr;
  Costs::NanoSeconds earliest_fanout(Costs::NanoSeconds::infinity());

  for (const auto& port : fanout) {
    if (port.node == node) {
      continue;
    }
    auto it = execution_times.find(port.node);
    if (it != execution_times.end() && it->second < earliest_fanout) {
      earliest_fanout = it->second;
      trigger = port.node;
    }
  }

  return trigger;
}

static bool IsSwappable(GraphView::InputPort input) {
  const NodeDef& node = *input.node;

  const OpDef* op_def;
  if (!OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok()) {
    return false;
  }

  DataType dtype;
  if (!InputTypeForNode(node, *op_def, input.port_id, &dtype).ok()) {
    return false;
  }

  return !IsRefType(dtype);
}

struct MemInfo {
  GraphView::OutputPort port;
  int64 memory_used;
  std::vector<GraphView::InputPort> uses_left;
  double fitness;

  bool operator<(const MemInfo& other) const { return fitness < other.fitness; }
};

static bool IdentifySwappingCandidates(
    Cluster* cluster, GrapplerItem* item, std::unordered_set<string>* skip_list,
    std::unordered_map<NodeDef*, SwapInfo>* nodes_to_swap) {
  GraphMemory memory(*item);
  const std::unordered_map<string, DeviceProperties>& devices =
      cluster->GetDevices();
  Status s = memory.InferStatically(devices);
  if (!s.ok()) {
    VLOG(1) << "Failed to infer memory usage: " << s.error_message();
    return false;
  }

  bool updated_graph = false;
  for (const auto& device : devices) {
    const string& name = device.first;
    const DeviceProperties& prop = device.second;
    if (prop.type() != "GPU") {
      continue;
    }
    if (prop.memory_size() <= 0) {
      VLOG(1) << "Peak memory usage unknown for device " << name;
      continue;
    }
    const GraphMemory::MemoryUsage& mem_usage = memory.GetPeakMemoryUsage(name);

    if (mem_usage.used_memory <= prop.memory_size()) {
      continue;
    }
    int64 required_savings = mem_usage.used_memory - prop.memory_size();

    std::unordered_map<string, Costs::NanoSeconds> op_completion_times;
    {
      VirtualCluster vcluster(cluster->GetDevices());
      if (!vcluster.Provision().ok()) {
        return false;
      }
      if (!vcluster.Initialize(*item).ok()) {
        return false;
      }
      RunMetadata metadata;
      Status s = vcluster.Run(item->graph, item->feed, item->fetch, &metadata);
      if (!s.ok() && s.code() != error::RESOURCE_EXHAUSTED) {
        return false;
      }

      for (const auto& dev_stats : metadata.step_stats().dev_stats()) {
        for (const auto& node_stats : dev_stats.node_stats()) {
          Costs::NanoSeconds exec_time =
              Costs::NanoSeconds(1) +
              Costs::MicroSeconds(node_stats.all_start_micros() +
                                  node_stats.op_end_rel_micros());
          op_completion_times.emplace(node_stats.node_name(), exec_time);
        }
      }
    }

    Costs::Duration peak_time = -1;
    for (const auto& live_tensor : mem_usage.live_tensors) {
      if (live_tensor.allocation_time > peak_time) {
        peak_time = live_tensor.allocation_time;
      }
    }

    std::vector<MemInfo> mem_state;

    GraphView graph(&item->graph);
    for (const auto& live_tensor : mem_usage.live_tensors) {
      if (live_tensor.memory_used <= 1024) {
        // Don't bother with small tensors.
        continue;
      }
      if (live_tensor.deallocation_time - live_tensor.allocation_time <=
          Costs::Duration(1e6)) {
        // Not enough time to swap.
        VLOG(1) << "Not enough time to swap: skipping " << live_tensor.node;
        continue;
      }

      if (skip_list->find(live_tensor.node) != skip_list->end()) {
        continue;
      }
      GraphView::OutputPort port =
          graph.GetOutputPort(live_tensor.node, live_tensor.output_id);
      if (!IsSwappable(graph, port)) {
        continue;
      }
      MemInfo mem_info;
      mem_info.port = port;
      mem_info.memory_used = live_tensor.memory_used;
      Costs::Duration allocation_time = live_tensor.allocation_time;
      Costs::Duration earliest_use(Costs::Duration::infinity());
      bool valid = true;
      for (GraphView::InputPort input : graph.GetFanout(port)) {
        // Get execution time.
        auto it = op_completion_times.find(input.node->name());
        if (it == op_completion_times.end()) {
          valid = false;
          break;
        }
        if (it->second <= peak_time) {
          continue;
        }

        if (skip_list->find(input.node->name()) != skip_list->end()) {
          valid = false;
          break;
        }
        string input_name =
            strings::StrCat(input.node->name(), ":", input.port_id);
        if (skip_list->find(input_name) != skip_list->end()) {
          valid = false;
          break;
        }
        if (!IsSwappable(input)) {
          valid = false;
          break;
        }

        // Set earliest use time that's after peak.
        mem_info.uses_left.emplace_back(input);
        earliest_use = std::min(earliest_use, it->second);
      }
      if (valid && !mem_info.uses_left.empty()) {
        // Compute the fitness: we need the tensor to be generated way away of
        // the time of peak memory usage (to ensure there is enough time to swap
        // it out). We also need to ensure it's used way after the peak time, to
        // ensure that swapping the tensor back in won't recreate the memory
        // bottleneck. Last but not least, we want the tensor to have as few
        // remaining uses as possible.
        //
        // Note that we must perform the arithmetic inexactly as "double", since
        // the values do not fit into any integral type.
        mem_info.fitness =
            MathUtil::IPow<double>((earliest_use - peak_time).count(), 2) /
            MathUtil::IPow<double>(mem_info.uses_left.size(), 2) +
            MathUtil::IPow<double>((allocation_time - peak_time).count(), 2);
        mem_info.fitness = -mem_info.fitness;
        mem_state.push_back(mem_info);
      }
    }

    // Sort by fitness
    std::sort(mem_state.begin(), mem_state.end());

    for (const MemInfo& mem_info : mem_state) {
      for (const GraphView::InputPort fanout_to_swap : mem_info.uses_left) {
        VLOG(1) << "Will swap fanout " << fanout_to_swap.node->name() << ":"
                << fanout_to_swap.port_id << " of tensor "
                << mem_info.port.node->name() << ":" << mem_info.port.port_id
                << " of size " << mem_info.memory_used;

        (*nodes_to_swap)[fanout_to_swap.node].inputs_to_swap.push_back(
            fanout_to_swap.port_id);
      }
      required_savings -= mem_info.memory_used;
      updated_graph = true;
      if (required_savings < 0) {
        break;
      }
    }
  }
  return updated_graph;
}

bool SwappingPass(RewriterConfig::MemOptType optimization_level,
                  Cluster* cluster, GrapplerItem* item,
                  std::unordered_set<string>* skip_list) {
  std::unordered_map<NodeDef*, SwapInfo> nodes_to_swap;
  if (optimization_level == RewriterConfig::DEFAULT_MEM_OPT ||
      optimization_level == RewriterConfig::SWAPPING_HEURISTICS ||
      optimization_level == RewriterConfig::HEURISTICS) {
    // Use heuristics to figure out what needs to be swapped;
    IdentifySwappingCandidates(cluster, item, skip_list, &nodes_to_swap);
  }
  // Look for manual annotatations in the graph.
  for (auto& node : *item->graph.mutable_node()) {
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
    return false;
  }

  // Estimate the size of the data to swap for each node.
  GraphProperties properties(*item);
  if (!properties.InferStatically(true).ok()) {
    return false;
  }
  for (auto& swap : nodes_to_swap) {
    const NodeDef* node = swap.first;
    const std::vector<OpInfo::TensorProperties>& props =
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

  std::unordered_map<const NodeDef*, Costs::NanoSeconds> execution_times;
  if (!EstimateEarliestExecutionTimes(*item, cluster, &execution_times).ok()) {
    return false;
  }

  std::unordered_map<string, const NodeDef*> name_map;
  for (const auto& node : item->graph.node()) {
    name_map[node.name()] = &node;
  }
  GraphView view(&item->graph);

  bool updated_graph = false;

  for (auto& swap : nodes_to_swap) {
    NodeDef* node = swap.first;
    const SwapInfo& swap_info = swap.second;
    if (skip_list->find(node->name()) != skip_list->end()) {
      continue;
    }

    // Make sure the tensor isn't swapped back in right away: look for node that
    // will execute just before we need to swap the data back, and add a control
    // dependency from that node to the swap node.
    const NodeDef* in_trigger =
        FindSwapInTrigger(node, swap_info, name_map, execution_times);
    // If we failed, don't attempt to reprocess this node in a subsequent pass.
    if (!in_trigger) {
      skip_list->insert(node->name());
      continue;
    }

    // Swap all the tensors that are marked with the 'swap_to_host' attribute.
    for (int input_id : swap_info.inputs_to_swap) {
      string input_name = strings::StrCat(node->name(), ":", input_id);
      if (skip_list->find(input_name) != skip_list->end()) {
        continue;
      } else {
        // Don't attempt to reprocess this input in a subsequent pass.
        skip_list->insert(input_name);
      }

      // Make sure the tensor is swapped out quickly: look for node that
      // will execute just after the tensor is generated and add a control
      // dependency from the swap out node to that node.
      NodeDef* out_trigger =
          FindSwapOutTrigger(node, input_id, view, execution_times);
      if (!out_trigger) {
        continue;
      }

      std::pair<NodeDef*, NodeDef*> swap_nodes;
      if (!BuildSwapPair(node, input_id, name_map, &item->graph, &swap_nodes)
               .ok()) {
        continue;
      }
      *swap_nodes.first->add_input() = node->input(input_id);
      *node->mutable_input(input_id) = swap_nodes.second->name();

      // Add the control dependencies needed to delay the execution of the swap.
      out_trigger->add_input(strings::StrCat("^", swap_nodes.first->name()));
      swap_nodes.second->add_input(strings::StrCat("^", in_trigger->name()));

      // Make sure we won't try to swap the swap nodes in subsequent passes.
      skip_list->insert(swap_nodes.first->name());
      skip_list->insert(swap_nodes.second->name());
    }
  }
  return updated_graph;
}

// TODO(rmlarsen): Add distributed TF test.
Status RelaxAllocatorConstraints(GraphDef* optimized_graph) {
  std::unordered_set<string> devices;
  std::vector<int> assign_nodes;
  bool found_send = false;
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    const NodeDef& node = optimized_graph->node(i);
    devices.insert(node.device());
    if (IsAssign(node)) {
      assign_nodes.push_back(i);
    }
    if (IsSend(node)) {
      found_send = true;
      break;
    }
  }
  if (!found_send && devices.size() == 1) {
    for (int assign_idx : assign_nodes) {
      // Set an attribute telling AssignOp to ignore allocator constraints.
      NodeDef* assign_node = optimized_graph->mutable_node(assign_idx);
      (*assign_node->mutable_attr())["_grappler_relax_allocator_constraints"]
          .set_b(true);
    }
    return Status::OK();
  }

  std::unordered_set<int> optimized_nodes;
  SimpleGraphView graph_view;
  TF_RETURN_IF_ERROR(graph_view.Initialize(*optimized_graph));
  for (int i : assign_nodes) {
    if (optimized_nodes.find(i) == optimized_nodes.end()) {
      const NodeDef& node = optimized_graph->node(i);
      optimized_nodes.insert(i);
      std::vector<int> assign_nodes_in_fanout;
      assign_nodes_in_fanout.push_back(i);
      std::set<int> transitive_fanout;
      graph_view.DepthFirstSearch(std::unordered_set<string>{}, i,
                                  &transitive_fanout);
      const string& assign_device = node.device();
      bool relax_constraint = true;
      // If all nodes in the transitive fanout are on the same device as the
      // assign node, there is no need to allocate the output in pinned memory.
      for (int fanout : transitive_fanout) {
        const NodeDef& fanout_node = optimized_graph->node(fanout);
        if (relax_constraint &&
            (fanout_node.device() != assign_device || IsSend(fanout_node))) {
          relax_constraint = false;
        }
        if (optimized_nodes.find(fanout) == optimized_nodes.end() &&
            IsAssign(fanout_node)) {
          assign_nodes_in_fanout.push_back(fanout);
        }
      }

      for (int assign_idx : assign_nodes_in_fanout) {
        if (relax_constraint) {
          // If all devices match in fanout of node(i) then, by transitivity,
          // they must also match in the fanout of other assign nodes
          // node(assign_idx) in the fanout, so we can process them here,
          // and save computing their transitive fanout later.
          optimized_nodes.insert(assign_idx);

          // Set an attribute telling AssignOp to ignore allocator constraints.
          NodeDef* assign_node = optimized_graph->mutable_node(assign_idx);
          (*assign_node
                ->mutable_attr())["_grappler_relax_allocator_constraints"]
              .set_b(true);
        }
      }
    }
  }
  return Status::OK();
}

Status MemoryOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph) {
  *optimized_graph = item.graph;

  RecomputationRewritingPass(optimization_level_,
                             recomputation_targets_name_scope_, optimized_graph,
                             item);

  GrapplerItem optimized_item(item, optimized_graph);
  std::unordered_set<string> skip_list;
  // Bound the number of rewrite passes to avoid long processing times on graphs
  // that simply won't fit in memory.
  bool updated_graph = true;
  for (int i = 0; i < 25 && updated_graph; ++i) {
    updated_graph = false;
    if ((optimization_level_ == RewriterConfig::DEFAULT_MEM_OPT ||
         optimization_level_ == RewriterConfig::SCHEDULING_HEURISTICS ||
         optimization_level_ == RewriterConfig::HEURISTICS) &&
        cluster != nullptr) {
      updated_graph |= SchedulingPass(cluster, &optimized_item);
    }

    if ((optimization_level_ == RewriterConfig::DEFAULT_MEM_OPT ||
         optimization_level_ == RewriterConfig::SWAPPING_HEURISTICS ||
         optimization_level_ == RewriterConfig::HEURISTICS ||
         optimization_level_ == RewriterConfig::MANUAL) &&
        cluster != nullptr) {
      updated_graph |= SwappingPass(optimization_level_, cluster,
                                    &optimized_item, &skip_list);
    }
  }

  TF_RETURN_IF_ERROR(RelaxAllocatorConstraints(&optimized_item.graph));

  optimized_graph->Swap(&optimized_item.graph);
  return Status::OK();
}

void MemoryOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimized_graph, double result) {
  // Nothing to do for MemoryOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
