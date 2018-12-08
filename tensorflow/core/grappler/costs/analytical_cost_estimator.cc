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

#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/costs/virtual_scheduler.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

namespace {

// Helper function in PredictCosts() to add cost node to cost_graph.
void AddCostNode(ReadyNodeManager* node_manager, const OpContext& op_context,
                 int node_id, const Costs& node_costs,
                 gtl::FlatMap<string, CostGraphDef::Node*>* name_to_cost_node,
                 gtl::FlatMap<string, int>* name_to_id,
                 CostGraphDef* cost_graph) {
  const string& op_name = op_context.name;
  auto it = name_to_cost_node->find(op_name);
  CostGraphDef::Node* node;
  if (it != name_to_cost_node->end()) {
    node = it->second;
    node->clear_input_info();
    node->clear_output_info();
  } else {
    node = cost_graph->add_node();
    (*name_to_cost_node)[op_name] = node;
    node->set_name(op_name);
    node->set_id(node_id);
    (*name_to_id)[node->name()] = node->id();
  }
  // For nodes we have seen before (e.g. Merge nodes are executed twice by
  // VirtualScheduler), the following fields will be overwritten/updated
  node->set_device(op_context.device_name);
  node->set_compute_cost(node_costs.execution_time.asMicroSeconds().count());
  node->set_compute_time(node_costs.compute_time.asMicroSeconds().count());
  node->set_memory_time(node_costs.memory_time.asMicroSeconds().count());
  node->set_inaccurate(node_costs.inaccurate);

  for (const string& input : node_manager->GetCurrNode()->input()) {
    int input_port;
    string input_name = ParseNodeName(input, &input_port);

    // All inputs should have been seen already unless this is a Merge node
    if (name_to_id->find(input_name) == name_to_id->end()) {
      if (!IsMerge(*node_manager->GetCurrNode()))
        LOG(ERROR) << "input: " << input
                   << " not found for non-Merge node: " << op_name;

      // For Merge node, some of inputs may not be seen before
      // For example, for a typical while loop in tensorflow, Merge node
      // will be executed twice by VirtualScheduler (one for Enter, the
      // other for NextIteration), so eventually both inputs will be added
      continue;
    }

    if (IsControlInput(input)) {
      node->add_control_input(name_to_id->at(input_name));
    } else {
      auto* input_info = node->add_input_info();
      input_info->set_preceding_node(name_to_id->at(input_name));
      input_info->set_preceding_port(input_port);
    }
  }

  for (const auto& output : op_context.op_info.outputs()) {
    auto output_info = node->add_output_info();
    output_info->set_alias_input_port(-1);
    output_info->set_dtype(output.dtype());
    auto shape = output_info->mutable_shape();
    *shape = output.shape();
  }
}

}  // namespace

AnalyticalCostEstimator::AnalyticalCostEstimator(Cluster* cluster,
                                                 bool use_static_shapes)
    : AnalyticalCostEstimator(
          cluster, absl::make_unique<OpLevelCostEstimator>(),
          ReadyNodeManagerFactory("FirstReady"), use_static_shapes, nullptr) {}

AnalyticalCostEstimator::AnalyticalCostEstimator(
    Cluster* cluster, std::unique_ptr<OpLevelCostEstimator> node_estimator,
    std::unique_ptr<ReadyNodeManager> node_manager, bool use_static_shapes,
    RunMetadata* run_metadata)
    : cluster_(cluster),
      node_estimator_(std::move(node_estimator)),
      node_manager_(std::move(node_manager)),
      use_static_shapes_(use_static_shapes),
      run_metadata_(run_metadata) {
  scheduler_ = absl::make_unique<VirtualScheduler>(use_static_shapes_, cluster_,
                                                   node_manager_.get());
}

Status AnalyticalCostEstimator::Initialize(const GrapplerItem& item) {
  item_ = item;
  return Status::OK();
}

// TODO(b/67607683): unify logic with VirtualCluster logic
Status AnalyticalCostEstimator::PredictCosts(const GraphDef& optimized_graph,
                                             CostGraphDef* cost_graph,
                                             Costs* costs) const {
  GrapplerItem item = item_;
  item.graph = optimized_graph;

  auto status = scheduler_->Init(&item);
  if (!status.ok()) {
    costs->execution_time = Costs::Duration::max();
    return status;
  }

  gtl::FlatMap<string, CostGraphDef::Node*> name_to_cost_node;
  if (cost_graph) {
    // TODO(pcma): Clear nodes in cost_graph after we make sure we always pass
    // in an empty cost_graph (a non-empty but incomplete cost_graph will cause
    // problems, e.g., no node_id in cost_graph)
    for (auto& node : *cost_graph->mutable_node()) {
      name_to_cost_node[node.name()] = &node;
    }
  }
  std::vector<string> inaccurate_nodes;
  int nodes_executed = 0;
  int node_id = 0;
  gtl::FlatMap<string, int> name_to_id;

  Costs node_costs;
  do {
    ++nodes_executed;
    OpContext op_context = scheduler_->GetCurrNode();
    node_costs = node_estimator_->PredictCosts(op_context);

    if (node_costs.inaccurate) {
      inaccurate_nodes.push_back(op_context.name);
      if (node_costs.num_ops_with_unknown_shapes > 0)
        VLOG(4) << op_context.name << " has "
                << node_costs.num_ops_with_unknown_shapes << " unknown shapes";
    }

    // TODO(pcma): Add unit tests for generating CostGraphDef
    if (cost_graph) {
      AddCostNode(node_manager_.get(), op_context, node_id++, node_costs,
                  &name_to_cost_node, &name_to_id, cost_graph);
    }
  } while (scheduler_->MarkCurrNodeExecuted(node_costs));

  VLOG(1) << inaccurate_nodes.size() << " out of " << nodes_executed
          << " nodes have inaccurate time estimation";
  if (VLOG_IS_ON(3)) {
    for (const auto& node : inaccurate_nodes) {
      VLOG(4) << "Node with inaccurate time estimation: " << node;
    }
  }

  *costs = scheduler_->Summary(run_metadata_);
  // run_metadata_ gets step_stats and parition_graphs from Summary.
  // Note that cost_graph could already point to the cost_graph field of
  // run_metadata_, since both are set by the caller.
  if (run_metadata_ && cost_graph &&
      run_metadata_->mutable_cost_graph() != cost_graph)
    *run_metadata_->mutable_cost_graph() = *cost_graph;

  if (VLOG_IS_ON(1)) {
    bool verbose = VLOG_IS_ON(2);
    if (run_metadata_) {
      VLOG(1) << GetStatsStringFromRunMetadata(*run_metadata_, verbose);
    } else {
      RunMetadata run_metadata;
      scheduler_->GenerateRunMetadata(&run_metadata);
      VLOG(1) << GetStatsStringFromRunMetadata(run_metadata, verbose);
    }
  }

  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
