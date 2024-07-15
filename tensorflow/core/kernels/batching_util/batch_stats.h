/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// The API for reporting and querying batch statistics such as the average batch
// costs for in-process use.
//
// All these statistics can also be retrieved from metrics reported by various
// modules (e.g., batch_resource_base), but it would be slow. This API, on the
// other hand, was designed to be queried on every request.
//
// The classes defined here are not supposed to be instantiated by the user.
// Instead, this file provides a single entry point:
//
//   BatchStats& GlobalBatchStats();
//
// For example, to register batch cost, do:
//
//   GlobalBatchStats()
//       .model(/* model_name= */ "m", /* op_name= */ "o")
//       .batch_size(4)
//       .tpu_cost
//       .Register(cost);
//
// To get the mean cost later, do:
//
//   std::optional<absl::Duration> cost =
//       .GlobalBatchStats()
//           .model(/* model_name= */ "m", /* op_name= */ "o")
//           .batch_size(4)
//           .tpu_cost
//           .mean();
//
// It is allowed and safe to store references to intermediate objects here
// because all intermediate objects are guaranteed to never be destroyed.
//
// All operations supported by this API are thread-safe.

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_STATS_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_STATS_H_

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow::serving {

// Tracks the average cost of registered samples.
//
// Thread-safe.
class CostTracker {
 public:
  // Registers a cost sample.
  void Register(absl::Duration cost) {
    DCHECK_GT(cost, absl::ZeroDuration());

    mutex_lock l(mu_);
    sample_count_++;
    sample_sum_ += cost;
  };

  // Returns the average cost of all registered samples, giving each sample
  // the same weight.
  //
  // Returns std::nullopt if no samples have been registered.
  //
  // TODO: b/325954758 - Switch this to an exponentially-decaying average. It's
  // likely enough to set the half-life to the last 100-1000 samples.
  std::optional<absl::Duration> mean() const {
    int64_t count;
    absl::Duration sum;

    {
      // We only hold the lock to read the values and release it before later
      // performing a relatively slow division operation.
      mutex_lock l(mu_);
      count = sample_count_;
      sum = sample_sum_;
    }

    if (count == 0) return std::nullopt;

    return sum / count;
  };

 private:
  mutable mutex mu_;

  int64_t sample_count_ TF_GUARDED_BY(mu_) = 0;
  absl::Duration sample_sum_ TF_GUARDED_BY(mu_);
};

// Tracks statistics for a particular model and batch size.
//
// Thread-safe.
class BatchSizeStats {
 public:
  CostTracker& tpu_cost() { return tpu_cost_; };

 private:
  CostTracker tpu_cost_;
};

// Tracks statistics for a particular model.
//
// Here, "model" means a specific version of a model (we assume that version is
// encoded in the op_name). In rare cases, when a model version has multiple
// BatchFunction operation, we also treat each such operation as a separate
// model in this context (they should also have different op_names).
//
// Thread-safe.
class ModelBatchStats {
 public:
  // Returns a reference to the BatchSizeStats instance for the given batch
  // size.
  //
  // The returned reference persist for as long as 'this' is alive.
  BatchSizeStats& batch_size(int32 batch_size) {
    mutex_lock l(mu_);
    return batch_size_stats_by_batch_size_[batch_size];
  }

  // Registers that the model server has processed a batch of size `size`
  // non-padding tasks for this model, updating the current cumulative
  // processed size.
  void RegisterProcessedSize(int64_t size) {
    cumulative_processed_size_.fetch_add(size, std::memory_order_relaxed);
  }

  // Returns the cumulative size processed by this model (the total
  // count of individual unit-sized queries processed by the model).
  int64_t cumulative_processed_size() const {
    return cumulative_processed_size_.load(std::memory_order_relaxed);
  }

  // Returns the list of batch sizes for which this model has statistics.
  //
  // The returned list is not guaranteed to be sorted.
  std::vector<int32> BatchSizes() const {
    std::vector<int32> result;
    mutex_lock l(mu_);
    result.reserve(batch_size_stats_by_batch_size_.size());
    for (const auto& [key, value] : batch_size_stats_by_batch_size_) {
      result.push_back(key);
    }
    return result;
  }

 private:
  mutable mutex mu_;

  // The storage of all BatchSizeStats instances.
  //
  // The mutex only protects adding/finding element in the map. Access to
  // elements themselves (after they were created) is not protected here. No
  // element deletion is possible because we return references to items in this
  // map and don't track their lifetime. We are using the node hash map so that
  // elements, once created, are fixed in memory.
  absl::node_hash_map<int32, BatchSizeStats> batch_size_stats_by_batch_size_
      TF_GUARDED_BY(mu_);

  // The total count of individual unit-sized queries processed by this model.
  // Can be used to generate an internal load metric per model. See
  // RegisterQuerySize for more details.
  std::atomic<int64_t> cumulative_processed_size_ = 0;
};

// Tracks batch statistics for all models.
//
// Thread-safe.
class BatchStats {
 public:
  // Returns a reference to ModelBatchStats for the provided model_name and
  // op_name.
  //
  // Upon invocation with a not-yet-seen arguments, creates an empty
  // ModelBatchStats instance.
  //
  // The returned reference persist for as long as 'this' is alive.
  ModelBatchStats& model(const std::string& model_name,
                         const std::string& op_name) {
    std::tuple key(model_name, op_name);
    mutex_lock l(mu_);
    return model_batch_stats_by_model_and_op_names_[key];
  }

  // Returns a list of all model and op names.
  //
  // This is the set of model/op names tracked by this BatchStats instance.
  // Note that the returned list is not guaranteed to be sorted.
  std::vector<std::tuple<std::string, std::string>> ModelAndOpNames() const {
    std::vector<std::tuple<std::string, std::string>> result;
    mutex_lock l(mu_);
    result.reserve(model_batch_stats_by_model_and_op_names_.size());
    for (const auto& [key, value] : model_batch_stats_by_model_and_op_names_) {
      result.push_back(key);
    }
    return result;
  }

 private:
  mutable mutex mu_;

  // The storage of all ModelBatchStats instances.
  //
  // The mutex only protects adding/finding element in the map. Access to
  // elements themselves (after they were created) is not protected here. No
  // element deletion is possible because we return references to items in this
  // map and don't track their lifetime. We are using the node hash map for
  // element pointer stability.
  absl::node_hash_map<std::tuple<std::string, std::string>, ModelBatchStats>
      model_batch_stats_by_model_and_op_names_ TF_GUARDED_BY(mu_);
};

// Returns the global instance of BatchStats, to use used for all production
// purposes (one should only instantiate individual classes from this file to
// test them).
inline BatchStats& GlobalBatchStats() {
  static BatchStats* instance = new BatchStats();
  return *instance;
}

}  // namespace tensorflow::serving

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_STATS_H_
