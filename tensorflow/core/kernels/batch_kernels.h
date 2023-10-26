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

#ifndef TENSORFLOW_CORE_KERNELS_BATCH_KERNELS_H_
#define TENSORFLOW_CORE_KERNELS_BATCH_KERNELS_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/types.h"

namespace tensorflow {

// Per-model inflight batches parameters.
ABSL_CONST_INIT extern const int64_t kMinInflightBatches;
ABSL_CONST_INIT extern const int64_t kInitialInflightBatches;
ABSL_CONST_INIT extern const int64_t kBatchesToAverageOver;
ABSL_CONST_INIT extern const int64_t kMaxInflightBatches;

namespace internal {
class BatchFunctionKernelTestAccess;
}

// Records the usage of attribute `enable_large_batch_splitting`.
void RecordBatchSplitUsage(
    std::optional<bool> maybe_enable_large_batch_splitting,
    absl::string_view model_name);

// Records the number of batch threads of a model.
void RecordBatchParamNumBatchThreads(int64_t num_batch_threads,
                                     absl::string_view model_name);

// Returns the model name from the context.
absl::string_view GetModelName(OpKernelContext* ctx);

// `BatchFunctionKernel` is the implementation of op `BatchFunction`.
//
// `BatchFunctionKernel` will batch (tensor) inputs by concatenating them
// along the 0-th dimension, schedule a user-defined computation, and then
// splits the returned tensors as batch output.
//
// In particular, an instance of `BatchFunctionKernel` creates or re-uses a
// a batch scheduler instance based on op attributes, pre-processes and enqueues
// concatenated inputs to the scheduler which invokes user-defined function,
// and then splits function output as op output.
//
// User defined function is named by attribute `f` and defined in the graph.
class BatchFunctionKernel : public AsyncOpKernel {
 public:
  explicit BatchFunctionKernel(OpKernelConstruction* c);

  bool IsExpensive() override;

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final;

 private:
  friend class internal::BatchFunctionKernelTestAccess;

  // Validates 'allowed_batch_sizes_'. The entries must increase monotonically.
  // If large batch split is not enabled, the last one must equal
  // `max_batch_size_`. otherwise the last element must be smaller than or equal
  // to `max_batch_size_`.
  Status ValidateAllowedBatchSizes() const;

  // Creates the function handle if it isn't initialized yet; and re-use it
  // afterwards.
  Status GetOrCreateFunctionHandle(OpKernelContext* c,
                                   FunctionLibraryRuntime::Handle* handle);

  // Instantiate the user-defined function and emits `handle`.
  Status InstantiateFunction(OpKernelContext* c,
                             FunctionLibraryRuntime::Handle* handle) const;

  // Initialize vars by reading from op-kernel-construction.
  // Vars
  // - enable_adaptive_batch_threads_
  //   true if value of attribute `kEnableAdaptiveSchedulerAttr` is true, or
  //   if `num_batch_threads` is not positive.
  // - adaptive_batch_scheduler_options_
  //   Read from corresponding attributes as long as they are set.
  void SetAdaptiveBatchSchedulerOptions(OpKernelConstruction* c,
                                        int32_t num_batch_threads);
  string container_;
  string shared_name_;
  string batcher_queue_;
  int32 num_batch_threads_;
  int32 max_batch_size_;
  int32 batch_timeout_micros_;
  int32 max_enqueued_batches_;
  std::vector<int32> allowed_batch_sizes_;
  int32 low_priority_max_batch_size_;
  int32 low_priority_batch_timeout_micros_;
  int32 low_priority_max_enqueued_batches_;
  std::vector<int32> low_priority_allowed_batch_sizes_;
  NameAttrList func_;
  absl::optional<FunctionLibraryRuntime::Handle> fhandle_ TF_GUARDED_BY(mu_);
  bool enable_large_batch_splitting_;
  bool has_attribute_enable_large_batch_splitting_;
  bool enable_adaptive_batch_threads_ = false;

  mutex mu_;

  // Parameters for adaptive batch scheduler only.
  // Note 'num_batch_threads_' above is shared by two implementations of batch
  // scheduler.
  struct AdaptiveBatchSchedulerOptions {
    int32 min_in_flight_batches_limit = kMinInflightBatches;
    int32 initial_in_flight_batches_limit = kInitialInflightBatches;
    int32 max_in_flight_batches_limit = kMaxInflightBatches;
    int32 batches_to_average_over = kBatchesToAverageOver;
    int64 full_batch_scheduling_boost_micros = -1;
  };
  absl::optional<AdaptiveBatchSchedulerOptions>
      adaptive_batch_scheduler_options_ = absl::nullopt;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCH_KERNELS_H_
