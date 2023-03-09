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

#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/url.h"
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
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::experimental::AutoShardDatasetOp;

// A dynamic port has form %port% or %port_foo% that is to be replaced with the
// actual port.
bool HasDynamicPort(absl::string_view address) {
  URL url(address);
  return url.has_port() && absl::StartsWith(url.port(), "%port") &&
         absl::EndsWith(url.port(), "%");
}

// Returns true if `config_address` has no port or a dynamic port (e.g.: %port%)
// and `worker_address` has an actual port (number of named port).
//
// For example, it returns true for the following cases:
//
//  config_address                    worker_address
//  ----------------------------------------------------------
//  /worker/task/0                    /worker/task/0:worker
//  /worker/task/0:%port%             /worker/task/0:10000
//  /worker/task/0:%port_worker%      /worker/task/0:worker
//  /worker/task/0:%port_worker%      /worker/task/0:10000
//  localhost                         localhost:10000
//  localhost:%port%                  localhost:10000
bool ShouldReplaceDynamicPort(absl::string_view config_address,
                              absl::string_view worker_address) {
  URL config_url(config_address), worker_url(worker_address);
  return (!config_url.has_port() || HasDynamicPort(config_address)) &&
         worker_url.has_port() && config_url.host() == worker_url.host();
}
}  // namespace

StatusOr<AutoShardRewriter> AutoShardRewriter::Create(const TaskDef& task_def) {
  TF_ASSIGN_OR_RETURN(
      AutoShardPolicy auto_shard_policy,
      ToAutoShardPolicy(task_def.processing_mode_def().sharding_policy()));
  return AutoShardRewriter(auto_shard_policy, task_def.num_workers(),
                           task_def.worker_index());
}

StatusOr<GraphDef> AutoShardRewriter::ApplyAutoShardRewrite(
    const GraphDef& graph_def) {
  if (auto_shard_policy_ == AutoShardPolicy::OFF) {
    return graph_def;
  }

  VLOG(2) << "Applying auto-shard policy "
          << AutoShardPolicy_Name(auto_shard_policy_)
          << ". Number of workers: " << num_workers_
          << "; worker index: " << worker_index_ << ".";
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

AutoShardRewriter::AutoShardRewriter(AutoShardPolicy auto_shard_policy,
                                     int64_t num_workers, int64_t worker_index)
    : auto_shard_policy_(auto_shard_policy),
      num_workers_(num_workers),
      worker_index_(worker_index) {}

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
  // This parameter is used internally by tf.distribute to rebatch the dataset.
  // It is not used outside the context of `experimental_distribute_dataset`.
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kNumReplicas].set_i(1);
  return config;
}

Status WorkerIndexResolver::ValidateWorker(
    absl::string_view worker_address) const {
  if (worker_addresses_.empty()) {
    return OkStatus();
  }

  for (absl::string_view config_address : worker_addresses_) {
    if (config_address == worker_address ||
        ShouldReplaceDynamicPort(config_address, worker_address)) {
      return OkStatus();
    }
  }

  return errors::FailedPrecondition(absl::Substitute(
      "Failed to assign an index for worker $0. Configured workers list: [$1]. "
      "The worker's address is not configured, or other workers are already "
      "running at the configured host. If your worker has restarted, make sure "
      "it runs at the same address and port.",
      worker_address, absl::StrJoin(worker_addresses_, ", ")));
}

void WorkerIndexResolver::AddWorker(absl::string_view worker_address) {
  for (std::string& config_address : worker_addresses_) {
    if (config_address == worker_address) {
      return;
    }
    if (ShouldReplaceDynamicPort(config_address, worker_address)) {
      config_address = std::string(worker_address);
      return;
    }
  }
}

StatusOr<int64_t> WorkerIndexResolver::GetWorkerIndex(
    absl::string_view worker_address) const {
  const auto it = absl::c_find(worker_addresses_, worker_address);
  if (it == worker_addresses_.cend()) {
    return errors::NotFound(absl::Substitute(
        "Failed to shard dataset in tf.data service: Worker $0 is not in the "
        "workers list. Got workers list $1.",
        worker_address, absl::StrJoin(worker_addresses_, ",")));
  }
  return std::distance(worker_addresses_.cbegin(), it);
}

}  // namespace data
}  // namespace tensorflow
