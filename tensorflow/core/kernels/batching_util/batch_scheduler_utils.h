/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

// Returns the next allowed batch size, which is the smallest allowed batch size
// greater than or equal to the given batch size. If allowed_batch_sizes,
// returns batch_size as is.
int GetNextAllowedBatchSize(int batch_size,
                            const std::vector<int32_t>& allowed_batch_sizes,
                            bool disable_padding);

// Returns the largest allowed batch size that is smaller than or equal to
// batch_size. Returns batch_size if no such size exists.
int GetPrevAllowedBatchSize(int batch_size,
                            const std::vector<int32_t>& allowed_batch_sizes,
                            bool disable_padding);

// Applies the batch padding policy to the candidate size and returns the target
// size depending on the policy.
int ApplyBatchPaddingPolicy(int candidate_size,
                            const std::vector<int32_t>& allowed_batch_sizes,
                            bool disable_padding,
                            absl::string_view batch_padding_policy,
                            ModelBatchStats* model_batch_stats);

// Constants containing possible values for the batch_padding_policy argument
// of MaybeBatchDown. This argument specifies the policy that a batch scheduler
// is using when deciding what to do when, say, 18 requests need to be batched,
// but only 16 and 32 batch sizes are allowed. The following options are
// available.
//
//   - PAD_UP: pad to size 32.
//   - BATCH_DOWN: schedule a batch of size 16 and leave 2 requests in the
//     batch buffer.
//   - MINIMIZE_TPU_COST_PER_REQUEST: a smarter greedy policy that chooses
//     to either PAD_UP or BATCH_DOWN so as to minimize the TPU costs per
//     real request. In this case, it would compare (batch_16_cost / 16) and
//     (batch_32_cost / 18).
//
inline constexpr absl::string_view kBatchDownPolicy = "BATCH_DOWN";
inline constexpr absl::string_view kPadUpPolicy = "PAD_UP";
inline constexpr absl::string_view kMinimizeTpuCostPerRequestPolicy =
    "MINIMIZE_TPU_COST_PER_REQUEST";

// Trims the batch to the next allowed batch size when possible and when
// configured by batch_padding_policy.
//
// When trimming, this function puts the trimmed tasks go into the
// out_trimmed_tasks vector in the same order as they were in the batch.
template <typename TaskType>
void MaybeBatchDown(Batch<TaskType>& batch,
                    const std::vector<int32_t>& allowed_batch_sizes,
                    bool disable_padding,
                    absl::string_view batch_padding_policy,
                    ModelBatchStats* model_batch_stats,
                    std::vector<std::unique_ptr<TaskType>>& out_trimmed_tasks) {
  if (batch.empty()) {
    return;
  }
  int32_t batch_size = batch.size();
  int32_t target_size =
      ApplyBatchPaddingPolicy(batch_size, allowed_batch_sizes, disable_padding,
                              batch_padding_policy, model_batch_stats);
  if (target_size < batch_size) {
    batch.TryTrimToNewSize(target_size, out_trimmed_tasks);
  }
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_
