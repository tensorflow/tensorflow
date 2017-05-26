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

#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"
#include "tensorflow/core/grappler/optimizers/static_schedule.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

const char* kRecomputedNodePrefix = "Recomputed";

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

void RecomputeSubgraph(
    const std::vector<const NodeDef*>& recomputed_source_nodes,
    const string& recompute_trigger_node_name,
    const std::vector<NodeDef*>& target_nodes, GraphDef* graph) {
  std::unordered_set<string> recomputed_node_names;
  for (const NodeDef* to_recompute : recomputed_source_nodes) {
    recomputed_node_names.insert(to_recompute->name());
  }
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
    // Set control dependencies on the recomputed nodes so that they are not run
    // until the specified trigger runs.
    *copied_node->add_input() =
        strings::StrCat("^", recompute_trigger_node_name);
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
    if (input_node->op() == "NextIteration" || input_node->op() == "Switch" ||
        input_node->op() == "Merge") {
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
