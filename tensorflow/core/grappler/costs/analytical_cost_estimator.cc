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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/costs/virtual_scheduler.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

AnalyticalCostEstimator::AnalyticalCostEstimator(Cluster* cluster,
                                                 bool use_static_shapes)
    : cluster_(cluster), use_static_shapes_(use_static_shapes) {}

Status AnalyticalCostEstimator::Initialize(const GrapplerItem& item) {
  item_ = item;
  return Status::OK();
}

Status AnalyticalCostEstimator::PredictCosts(const GraphDef& optimized_graph,
                                             CostGraphDef* cost_graph,
                                             Costs* costs) const {
  GrapplerItem item = item_;
  item.graph = optimized_graph;
  GraphProperties properties(item);
  Status status;
  if (use_static_shapes_) {
    status = properties.InferStatically();
  } else {
    status = properties.InferDynamically(cluster_);
  }

  if (!status.ok()) {
    costs->execution_time = Costs::Duration::max();
    return status;
  }

  std::unordered_map<string, CostGraphDef::Node*> name_to_cost;
  if (cost_graph) {
    for (auto& node : *cost_graph->mutable_node()) {
      name_to_cost[node.name()] = &node;
    }
  }
  std::vector<string> inaccurate_nodes;
  VirtualScheduler scheduler(optimized_graph, item_.fetch);
  VirtualPlacer placer(cluster_);
  Costs node_costs;
  do {
    const NodeDef* node = scheduler.GetCurrNode();
    std::vector<OpInfo::TensorProperties> inputs =
        properties.GetInputProperties(node->name());

    OpInfo::DeviceProperties device = placer.get_device(*node);
    OpInfo op_info;
    op_info.set_op(node->op());
    *op_info.mutable_attr() = node->attr();
    for (auto& input : inputs) {
      op_info.add_inputs()->Swap(&input);
    }
    op_info.mutable_device()->Swap(&device);

    node_costs = node_estimator_.PredictCosts(op_info);
    if (node_costs.inaccurate) {
      inaccurate_nodes.push_back(node->name());
    }
    if (cost_graph) {
      auto it = name_to_cost.find(node->name());
      CostGraphDef::Node* cost_node;
      if (it != name_to_cost.end()) {
        cost_node = it->second;
      } else {
        cost_node = cost_graph->add_node();
        cost_node->set_name(node->name());
      }
      string device_name = properties.GetDeviceName(node->name());
      cost_node->set_device(device_name);
      cost_node->set_compute_cost(
          node_costs.execution_time.asMicroSeconds().count());
      cost_node->set_compute_time(
          node_costs.compute_time.asMicroSeconds().count());
      cost_node->set_memory_time(
          node_costs.memory_time.asMicroSeconds().count());
      std::vector<OpInfo::TensorProperties> outputs =
          properties.GetOutputProperties(node->name());
      for (const auto& output : outputs) {
        auto output_info = cost_node->add_output_info();
        output_info->set_dtype(output.dtype());
        auto shape = output_info->mutable_shape();
        *shape = output.shape();
      }
    }
  } while (scheduler.MarkCurrNodeExecuted(node_costs));

  *costs = scheduler.Summary();
  VLOG(1) << inaccurate_nodes.size() << " out of "
          << optimized_graph.node_size()
          << " nodes have inaccurate time estimation";
  for (const auto& node : inaccurate_nodes) {
    VLOG(2) << "Node with inaccurate time estimation: " << node;
  }
  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
