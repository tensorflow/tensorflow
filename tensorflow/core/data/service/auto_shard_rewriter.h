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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_AUTO_SHARD_REWRITER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_AUTO_SHARD_REWRITER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {

// Rewrites the dataset graph by applying an auto-shard policy.
class AutoShardRewriter {
 public:
  // Creates an `AutoShardRewriter` according to `task_def`. Returns an error if
  // the sharding policy is not a valid auto-shard policy.
  static StatusOr<AutoShardRewriter> Create(const TaskDef& task_def);

  // Applies auto-sharding to `graph_def`. If auto-shard policy is OFF, returns
  // the same graph as `graph_def`. Otherwise, returns the re-written graph.
  StatusOr<GraphDef> ApplyAutoShardRewrite(const GraphDef& graph_def);

 private:
  AutoShardRewriter(AutoShardPolicy auto_shard_policy, int64 num_workers,
                    int64 worker_index);

  // Creates a rewrite config based on the auto-shard policy.
  tensorflow::RewriterConfig::CustomGraphOptimizer GetRewriteConfig() const;

  const AutoShardPolicy auto_shard_policy_;
  const int64_t num_workers_;
  const int64_t worker_index_;
};

// Maps a worker to its index, given a list of workers. For example, suppose
// `worker_addresses` contains
//   /worker/task/0:worker, /worker/task/1:worker, /worker/task/2:worker,
// then
//   /worker/task/0:worker maps to index 0,
//   /worker/task/1:worker maps to index 1,
//   /worker/task/2:worker maps to index 2.
// This is useful for deterministically sharding a dataset among a fixed set of
// tf.data service workers.
class WorkerIndexResolver {
 public:
  // Constructs a `WorkerIndexResolver` to generate worker indexes according to
  // the specified worker addresses. The worker addresses can be "host" or
  // "host:port", where "port" is a number, named port, or "%port%" to be
  // replaced with the actual port.
  template <class T>
  explicit WorkerIndexResolver(const T& worker_addresses)
      : worker_addresses_(worker_addresses.cbegin(), worker_addresses.cend()) {}

  // Validates `worker_address`. Returns an error if the `worker_addresses` list
  // is non-empty and `worker_address` is not specified in the worker addresses
  // list (with optional port replacement).
  Status ValidateWorker(absl::string_view worker_address) const;

  // Processes a worker at address `worker_address`. Its index can be retrieved
  // by calling `GetWorkerIndex`.
  void AddWorker(absl::string_view worker_address);

  // Returns the worker index for the worker at `worker_address`. Returns a
  // NotFound error if the worker is not registered.
  StatusOr<int64_t> GetWorkerIndex(absl::string_view worker_address) const;

 private:
  std::vector<std::string> worker_addresses_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_AUTO_SHARD_REWRITER_H_
