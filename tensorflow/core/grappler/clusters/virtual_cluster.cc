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

#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/grappler/costs/virtual_scheduler.h"

namespace tensorflow {
namespace grappler {

VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices)
    : Cluster(0), node_estimator_(new OpLevelCostEstimator()) {
  devices_ = devices;
}

VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices,
    OpLevelCostEstimator* node_estimator)
    : Cluster(0), node_estimator_(node_estimator) {
  devices_ = devices;
}
VirtualCluster::~VirtualCluster() {}

Status VirtualCluster::Provision() { return Status::OK(); }

Status VirtualCluster::Initialize(const GrapplerItem& item) {
  return Status::OK();
}

Status VirtualCluster::Run(const GraphDef& graph,
                           const std::vector<std::pair<string, Tensor>>& feed,
                           const std::vector<string>& fetch,
                           RunMetadata* metadata) {
  // Initialize a virtual scheduler to process the graph. Make sure to use
  // static shape inference to prevent the schedulrer from calling the Run
  // method on the cluster, and create an infinite loop.
  GrapplerItem item;
  item.graph = graph;
  item.feed = feed;
  item.fetch = fetch;
  VirtualScheduler scheduler(&item, true, this);
  TF_RETURN_IF_ERROR(scheduler.Init());

  if (metadata) {
    metadata->clear_step_stats();
    metadata->clear_cost_graph();
    metadata->clear_partition_graphs();
  }

  Costs node_costs;
  do {
    OpContext op_context = scheduler.GetCurrNode();
    node_costs = node_estimator_->PredictCosts(op_context);
    if (metadata) {
      CostGraphDef::Node* cost_node =
          metadata->mutable_cost_graph()->add_node();
      const string& op_name = op_context.name;
      cost_node->set_name(op_name);
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
        *output_info->mutable_shape() = output.shape();

        int64 size = DataTypeSize(output.dtype());
        for (const auto& dim : output.shape().dim()) {
          size *= std::max<int64>(1, dim.size());
        }
        output_info->set_size(size);
      }
    }
  } while (scheduler.MarkCurrNodeExecuted(node_costs));

  if (metadata) {
    scheduler.Summary(metadata);
  }
  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
