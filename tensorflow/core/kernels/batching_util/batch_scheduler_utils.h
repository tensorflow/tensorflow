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
                            const std::vector<int32>& allowed_batch_sizes,
                            bool disable_padding);

// Returns the largest allowed batch size that is smaller than or equal to
// batch_size. Returns batch_size if no such size exists.
int GetPrevAllowedBatchSize(int batch_size,
                            const std::vector<int32>& allowed_batch_sizes,
                            bool disable_padding);

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
                    const std::vector<int32>& allowed_batch_sizes,
                    bool disable_padding,
                    absl::string_view batch_padding_policy,
                    ModelBatchStats* model_batch_stats,
                    std::vector<std::unique_ptr<TaskType>>& out_trimmed_tasks) {
  if (batch_padding_policy == kPadUpPolicy) {
    // This is the default behavior of batch resource when it is given a batch
    // size that doesn't match any of the allowed batch sizes.
    return;
  }
  bool minimize_tpu_cost_per_request;
  if (batch_padding_policy == kBatchDownPolicy) {
    minimize_tpu_cost_per_request = false;
  } else if (batch_padding_policy == kMinimizeTpuCostPerRequestPolicy) {
    if (model_batch_stats == nullptr) {
      LOG_FIRST_N(ERROR, 1)
          << kMinimizeTpuCostPerRequestPolicy
          << " batch padding policy has been chosen "
             "but no ModelBatchStats passed to the batch scheduler; will "
             "fall back on the "
          << kPadUpPolicy << " policy.";
      return;
    }
    minimize_tpu_cost_per_request = true;
  } else {
    LOG_FIRST_N(ERROR, 1) << "Unsupported batch_padding_policy: "
                          << batch_padding_policy << ", falling back on the "
                          << kPadUpPolicy << " policy.";
    return;
  }

  int32 batch_size = batch.size();

  int32 pad_up_size =
      GetNextAllowedBatchSize(batch_size, allowed_batch_sizes, disable_padding);
  if (pad_up_size == batch_size) {
    return;  // Good, no padding is necessary.
  }

  int32 batch_down_size =
      GetPrevAllowedBatchSize(batch_size, allowed_batch_sizes, disable_padding);
  if (batch_down_size == batch_size) {
    return;  // Can't batch down (e.g. no smaller batch size available).
  }

  if (minimize_tpu_cost_per_request) {
    // TODO: b/325954758 - Consider logging a warning here or elsewhere if
    // a larger batch doesn't cost meaningfully cheaper than a smaller batch.
    // TODO: b/325954758 - Consider logging a warning here or elsewhere if a
    // smaller batch costs unreasonably cheaper than a larger one (assuming
    // a batch cost model = constant_cost + batch_size * per_element_cost).
    // TODO: b/325954758 - Consider occasionally picking either batch size so
    // that we learn fresh costs of each batch size. For this code, it is not a
    // large priority though because if we are in between two allowed batch
    // sizes (say, 16 and 32), chances are that will occasionally organically
    // get batches of exact sizes 16 and 32 (and then we pick those
    // unconditionally). But if we explicitly occasionally explored other batch
    // sizes, we wouldn't have to rely on this "chances are". For other
    // applications of batch costs, we might also want to occasionally explore
    // all allowed batch sizes and not just 16 and 32 from this example.
    std::optional<absl::Duration> down_batch_cost =
        model_batch_stats->batch_size(batch_down_size).tpu_cost().mean();
    std::optional<absl::Duration> up_batch_cost =
        model_batch_stats->batch_size(pad_up_size).tpu_cost().mean();
    if (!down_batch_cost.has_value() || !up_batch_cost.has_value()) {
      // We have no data about batch costs, let's just do nothing.
      return;
    }

    auto batch_down_cost_per_request = *down_batch_cost / batch_down_size;
    auto pad_up_cost_per_request = *up_batch_cost / batch_size;

    if (pad_up_cost_per_request < batch_down_cost_per_request) {
      // Abort batching down because it's cheaper to pad up.
      return;
    }
  }

  // Batch down.
  batch.TryTrimToNewSize(batch_down_size, out_trimmed_tasks);
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_
