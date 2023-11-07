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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace tensorflow {

// RequestCost collects the costs and metrics for processing an rpc request.
class RequestCost {
 public:
  // Records costs. The inputs should be pairs of cost type and cost.
  // It's thread-safe, and can be called from different threads.
  void RecordCost(
      const std::vector<std::pair<absl::string_view, absl::Duration>>& costs);

  // Gets all types of costs for processing an rpc request.
  // It's thread-safe. It's expected to be called at the end of processing an
  // rpc request, when all the costs have been collected.
  absl::flat_hash_map<std::string, absl::Duration> GetCosts() const;

  // Metrics of each batch that processes this rpc request.
  struct BatchMetrics {
    // Size of the batch.
    int64_t processed_size = 0;
    // In this batch, input size from this rpc request.
    int64_t input_size = 0;
    // In this batch, the padding amount.
    int64_t padding_size = 0;
    // Costs for processing this batch.
    absl::flat_hash_map<std::string, absl::Duration> batch_costs;
  };

  // Records the metrics of a batch.
  // It's thread-safe, and can be called from different threads. It may be
  // called multiple times if a request is processed by more than one batches.
  void RecordBatchMetrics(const BatchMetrics& batch_metrics);

  // Get metrics of all the batches that process this rpc request.
  // It's thread-safe. It's expected to be called at the end of processing an
  // rpc request, when all batch processing has completed.
  std::vector<BatchMetrics> GetBatchMetrics() const;

 private:
  mutable absl::Mutex mutex_;

  // Query costs. Map from cost type to cost.
  absl::flat_hash_map<std::string, absl::Duration> cost_map_
      ABSL_GUARDED_BY(mutex_);

  // Metrics of batches that process this rpc request.
  std::vector<BatchMetrics> batch_metrics_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_
