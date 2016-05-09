/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

CostModelManager::~CostModelManager() {
  for (auto it : cost_models_) {
    delete it.second;
  }
}

CostModel* CostModelManager::FindOrCreateCostModel(const Graph* graph) {
  mutex_lock l(mu_);
  auto it = cost_models_.find(graph);
  if (it != cost_models_.end()) {
    return it->second;
  }
  CostModel* cost_model = new CostModel(false);
  cost_model->InitFromGraph(*graph);
  cost_models_.emplace(graph, cost_model);
  return cost_model;
}

Status CostModelManager::BuildCostGraphDef(const Graph* graph,
                                           CostGraphDef* cost_graph) {
  mutex_lock l(mu_);
  // Get the cost model for the graph.
  auto it = cost_models_.find(graph);
  if (it == cost_models_.end()) {
    return errors::InvalidArgument("The cost model graph doesn't exist.");
  }
  CostModel* cost_model = it->second;

  // Construct the cost graph.
  std::vector<const Edge*> inputs;
  for (const Node* n : graph->nodes()) {
    CostGraphDef::Node* cnode = cost_graph->add_node();
    cnode->set_name(n->name());
    cnode->set_id(cost_model->Id(n));

    inputs.clear();
    inputs.resize(n->num_inputs(), nullptr);
    for (const Edge* e : n->in_edges()) {
      inputs[e->dst_input()] = e;
    }
    for (const Edge* e : inputs) {
      CostGraphDef::Node::InputInfo* input_info = cnode->add_input_info();
      input_info->set_preceding_node(cost_model->Id(e->src()));
      input_info->set_preceding_port(e->src_output());
    }

    for (int i = 0; i < n->num_outputs(); i++) {
      CostGraphDef::Node::OutputInfo* output_info = cnode->add_output_info();
      output_info->set_size(cost_model->MaxMemSize(n, i).value());
      output_info->set_alias_input_port(cost_model->Aliases(n, i));
    }

    cnode->set_temporary_memory_size(cost_model->TempMemSize(n).value());

    // For now we treat all send nodes as final.
    // TODO(yuanbyu): Send nodes for fetches shouldn't be treated as final.
    cnode->set_is_final(n->IsSend());
  }
  return Status::OK();
}

}  // namespace tensorflow
