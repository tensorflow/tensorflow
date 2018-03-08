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

#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"

#include <limits>
#include <unordered_map>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/costs/virtual_scheduler.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

AnalyticalCostEstimator::AnalyticalCostEstimator(Cluster* cluster,
                                                 bool use_static_shapes)
    : cluster_(cluster),
      node_estimator_(new OpLevelCostEstimator()),
      node_manager_(VirtualScheduler::ReadyNodeManagerFactory("FirstReady")),
      use_static_shapes_(use_static_shapes) {}

AnalyticalCostEstimator::AnalyticalCostEstimator(
    Cluster* cluster, OpLevelCostEstimator* node_estimator,
    ReadyNodeManager* node_manager, bool use_static_shapes)
    : cluster_(cluster),
      node_estimator_(node_estimator),
      node_manager_(node_manager),
      use_static_shapes_(use_static_shapes) {}

Status AnalyticalCostEstimator::Initialize(const GrapplerItem& item) {
  item_ = item;
  return Status::OK();
}

Status AnalyticalCostEstimator::PredictCosts(const GraphDef& optimized_graph,
                                             CostGraphDef* cost_graph,
                                             Costs* costs) const {
  GrapplerItem item = item_;
  item.graph = optimized_graph;

  std::unordered_map<string, CostGraphDef::Node*> name_to_cost;
  if (cost_graph) {
    for (auto& node : *cost_graph->mutable_node()) {
      name_to_cost[node.name()] = &node;
    }
  }
  std::vector<string> inaccurate_nodes;
  int nodes_executed = 0;
  VirtualScheduler scheduler(&item, use_static_shapes_, cluster_,
                             node_manager_.get());
  auto status = scheduler.Init();
  if (!status.ok()) {
    costs->execution_time = Costs::Duration::max();
    return status;
  }

  Costs node_costs;
  do {
    ++nodes_executed;
    OpContext op_context = scheduler.GetCurrNode();
    const string& op_name = op_context.name;

    node_costs = node_estimator_->PredictCosts(op_context);
    if (node_costs.inaccurate) {
      inaccurate_nodes.push_back(op_name);
    }
    if (cost_graph) {
      auto it = name_to_cost.find(op_name);
      CostGraphDef::Node* cost_node;
      if (it != name_to_cost.end()) {
        cost_node = it->second;
      } else {
        cost_node = cost_graph->add_node();
        cost_node->set_name(op_name);
      }
      cost_node->set_device(op_context.device_name);
      cost_node->set_compute_cost(
          node_costs.execution_time.asMicroSeconds().count());
      cost_node->set_compute_time(
          node_costs.compute_time.asMicroSeconds().count());
      cost_node->set_memory_time(
          node_costs.memory_time.asMicroSeconds().count());
      for (const auto& output : op_context.op_info.outputs()) {
        auto output_info = cost_node->add_output_info();
        output_info->set_dtype(output.dtype());
        auto shape = output_info->mutable_shape();
        *shape = output.shape();
      }
    }
  } while (scheduler.MarkCurrNodeExecuted(node_costs));

  RunMetadata run_metadata;
  *costs = scheduler.Summary(&run_metadata);
  VLOG(1) << inaccurate_nodes.size() << " out of " << nodes_executed
          << " nodes have inaccurate time estimation";
  if (VLOG_IS_ON(3)) {
    for (const auto& node : inaccurate_nodes) {
      VLOG(4) << "Node with inaccurate time estimation: " << node;
    }
  }

  if (VLOG_IS_ON(1)) {
    bool verbosity = VLOG_IS_ON(2);
    VLOG(1) << GetStatsStringFromRunMetadata(run_metadata, verbosity);
  }
  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
