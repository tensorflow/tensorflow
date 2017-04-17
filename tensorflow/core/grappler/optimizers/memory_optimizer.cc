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
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

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

Status MemoryOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph) {
  *optimized_graph = item.graph;

  for (auto& node : *optimized_graph->mutable_node()) {
    if (node.attr().count("swap_to_host") == 0) {
      continue;
    }

    // Swap all the tensors that are marked with the 'swap_to_host' attribute.
    for (int input_id : node.attr().at("swap_to_host").list().i()) {
      std::pair<NodeDef*, NodeDef*> swap_nodes =
          BuildSwapPair(&node, input_id, optimized_graph);
      *swap_nodes.first->add_input() = node.input(input_id);
      *node.mutable_input(input_id) = swap_nodes.second->name();

      // TODO(bsteiner): Make sure the tensor isn't swapped back in right away
      // by adding a control dependency to delay the execution of the swap.
      // string trigger;
      //*swap_nodes.second->add_input() = strings::StrCat("^", trigger);
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
