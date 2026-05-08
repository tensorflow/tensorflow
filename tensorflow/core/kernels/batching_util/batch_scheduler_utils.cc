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

#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

int GetNextAllowedBatchSize(int batch_size,
                            const std::vector<int32_t>& allowed_batch_sizes,
                            bool disable_padding) {
  if (disable_padding || allowed_batch_sizes.empty()) {
    return batch_size;
  }
  DCHECK(absl::c_is_sorted(allowed_batch_sizes));
  DCHECK_GT(batch_size, 0);
  for (int allowed_size : allowed_batch_sizes) {
    if (allowed_size >= batch_size) {
      return allowed_size;
    }
  }
  LOG(ERROR) << "Batch size " << batch_size
             << " is greater than largest allowed size; ignoring allowed sizes "
                "constraint.";
  return batch_size;
}

int32_t GetPrevAllowedBatchSize(int batch_size,
                                const std::vector<int32_t>& allowed_batch_sizes,
                                bool disable_padding) {
  if (disable_padding || allowed_batch_sizes.empty()) {
    return batch_size;
  }

  DCHECK(absl::c_is_sorted(allowed_batch_sizes));
  DCHECK_GT(batch_size, 0);

  // First from the end allowed_batch_size not larger than batch_size.
  auto result = std::find_if(
      allowed_batch_sizes.rbegin(), allowed_batch_sizes.rend(),
      [&](int allowed_size) { return allowed_size <= batch_size; });

  if (result == allowed_batch_sizes.rend()) {
    // No such element exists.
    return batch_size;
  }

  return *result;
}

int ApplyBatchPaddingPolicy(int candidate_size,
                            const std::vector<int32_t>& allowed_batch_sizes,
                            bool disable_padding,
                            absl::string_view batch_padding_policy,
                            ModelBatchStats* model_batch_stats) {
  if (candidate_size == 0) {
    return candidate_size;
  }
  if (batch_padding_policy == kPadUpPolicy) {
    return candidate_size;
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
      return candidate_size;
    }
    minimize_tpu_cost_per_request = true;
  } else {
    LOG_FIRST_N(ERROR, 1) << "Unsupported batch_padding_policy: "
                          << batch_padding_policy << ", falling back on the "
                          << kPadUpPolicy << " policy.";
    return candidate_size;
  }

  int32_t pad_up_size = GetNextAllowedBatchSize(
      candidate_size, allowed_batch_sizes, disable_padding);
  if (pad_up_size == candidate_size) {
    return candidate_size;  // Good, no padding is necessary.
  }

  int32_t batch_down_size = GetPrevAllowedBatchSize(
      candidate_size, allowed_batch_sizes, disable_padding);
  if (batch_down_size == candidate_size) {
    return candidate_size;  // Can't batch down (e.g. no smaller batch size
                            // available).
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
      return candidate_size;
    }

    auto batch_down_cost_per_request = *down_batch_cost / batch_down_size;
    auto pad_up_cost_per_request = *up_batch_cost / candidate_size;

    if (pad_up_cost_per_request < batch_down_cost_per_request) {
      // Abort batching down because it's cheaper to pad up.
      return candidate_size;
    }
  }

  return batch_down_size;
}

}  // namespace serving
}  // namespace tensorflow
