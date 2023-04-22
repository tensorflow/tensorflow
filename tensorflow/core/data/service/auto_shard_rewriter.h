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
#include "absl/types/span.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {

// Rewrites the dataset graph by applying an auto-shard policy.
class AutoShardRewriter {
 public:
  // Creates an AutoShardRewriter. If `auto_shard_policy` is not OFF, it should
  // also provide a non-empty list of tf.data service worker addresses which
  // contains the calling worker's `worker_address`. Otherwise, this function
  // returns a NotFound error.
  static StatusOr<AutoShardRewriter> Create(
      AutoShardPolicy auto_shard_policy,
      absl::Span<const absl::string_view> worker_addresses,
      absl::string_view worker_address);

  // Applies auto-sharding to `graph_def`. If auto-shard policy is OFF, returns
  // the same graph as `graph_def`. Otherwise, returns the re-written graph.
  StatusOr<GraphDef> ApplyAutoShardRewrite(const GraphDef& graph_def);

 private:
  explicit AutoShardRewriter(AutoShardPolicy auto_shard_policy,
                             int64 num_workers, int64 worker_index);

  // Returns the shard index of the worker at `worker_address`, given the list
  // of workers from `worker_addresses`. If no worker exists at the address, it
  // returns a NotFound error.
  static StatusOr<int> GetWorkerIndex(
      absl::Span<const absl::string_view> worker_addresses,
      absl::string_view worker_address);

  // Creates a rewrite config based on the auto-shard policy.
  tensorflow::RewriterConfig::CustomGraphOptimizer GetRewriteConfig() const;

  const AutoShardPolicy auto_shard_policy_;
  const int64 num_workers_;
  const int64 worker_index_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_AUTO_SHARD_REWRITER_H_
