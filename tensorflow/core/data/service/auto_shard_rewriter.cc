/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/auto_shard_rewriter.h"

#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/data/auto_shard.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"
#include "tensorflow/core/kernels/data/experimental/auto_shard_dataset_op.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace data {

using ::tensorflow::data::experimental::AutoShardDatasetOp;

StatusOr<AutoShardRewriter> AutoShardRewriter::Create(
    AutoShardPolicy auto_shard_policy,
    absl::Span<const absl::string_view> worker_addresses,
    absl::string_view worker_address) {
  if (auto_shard_policy == AutoShardPolicy::OFF) {
    return AutoShardRewriter(AutoShardPolicy::OFF, /*num_workers=*/0,
                             /*worker_index=*/0);
  }

  TF_ASSIGN_OR_RETURN(const int64 worker_index,
                      GetWorkerIndex(worker_addresses, worker_address));
  return AutoShardRewriter(auto_shard_policy,
                           static_cast<int64>(worker_addresses.size()),
                           worker_index);
}

StatusOr<int> AutoShardRewriter::GetWorkerIndex(
    absl::Span<const absl::string_view> worker_addresses,
    absl::string_view worker_address) {
  const auto it = absl::c_find(worker_addresses, worker_address);
  if (it == worker_addresses.cend()) {
    return errors::NotFound(absl::Substitute(
        "Failed to apply auto-shard policy: Worker $0 is not in the auto-shard "
        "workers list. Got workers list $1.",
        worker_address, absl::StrJoin(worker_addresses, ",")));
  }
  return std::distance(worker_addresses.cbegin(), it);
}

AutoShardRewriter::AutoShardRewriter(const AutoShardPolicy auto_shard_policy,
                                     const int64 num_workers,
                                     const int64 worker_index)
    : auto_shard_policy_(auto_shard_policy),
      num_workers_(num_workers),
      worker_index_(worker_index) {}

StatusOr<GraphDef> AutoShardRewriter::ApplyAutoShardRewrite(
    const GraphDef& graph_def) {
  if (auto_shard_policy_ == AutoShardPolicy::OFF) {
    return graph_def;
  }

  VLOG(2) << "Applying auto-shard policy "
          << AutoShardPolicy_Name(auto_shard_policy_);
  grappler::AutoShard autoshard;
  tensorflow::RewriterConfig::CustomGraphOptimizer config = GetRewriteConfig();
  TF_RETURN_IF_ERROR(autoshard.Init(&config));

  GraphDef input_graph = graph_def;
  TF_ASSIGN_OR_RETURN(std::string dataset_node, GetDatasetNode(input_graph));
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      GetGrapplerItem(&input_graph, &dataset_node, /*add_fake_sinks=*/false);

  GraphDef rewritten_graph;
  std::unordered_map<std::string, tensorflow::DeviceProperties> device_map;
  tensorflow::grappler::VirtualCluster cluster(device_map);
  grappler::AutoShard::OptimizationStats stats;
  TF_RETURN_IF_ERROR(autoshard.OptimizeAndCollectStats(
      &cluster, *grappler_item, &rewritten_graph, &stats));
  return rewritten_graph;
}

tensorflow::RewriterConfig::CustomGraphOptimizer
AutoShardRewriter::GetRewriteConfig() const {
  tensorflow::RewriterConfig::CustomGraphOptimizer config;
  config.set_name("tf-data-service-auto-shard");
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kNumWorkers].set_i(
      num_workers_);
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kIndex].set_i(
      worker_index_);
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kAutoShardPolicy].set_i(
      auto_shard_policy_);
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kNumReplicas].set_i(1);
  return config;
}

}  // namespace data
}  // namespace tensorflow
