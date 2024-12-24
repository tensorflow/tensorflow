/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

namespace {

static const string kCostModelLogTag = "COST_MODEL";

}  // namespace

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

bool CostModelManager::RemoveCostModelForGraph(const Graph* graph) {
  mutex_lock l(mu_);
  auto itr = cost_models_.find(graph);
  if (itr == cost_models_.end()) {
    return false;
  }
  delete itr->second;
  cost_models_.erase(graph);
  return true;
}

absl::Status CostModelManager::AddToCostGraphDef(const Graph* graph,
                                                 CostGraphDef* cost_graph) {
  mutex_lock l(mu_);
  // Get the cost model for the graph.
  auto it = cost_models_.find(graph);
  if (it == cost_models_.end()) {
    return errors::InvalidArgument("The cost model graph doesn't exist.");
  }
  CostModel* cost_model = it->second;
  cost_model->AddToCostGraphDef(graph, cost_graph);
  return absl::OkStatus();
}

}  // namespace tensorflow
