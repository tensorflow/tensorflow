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
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

ABSL_FLAG(tensorflow::serving::BatchPaddingPolicy,
          tensorflow_batch_padding_policy,
          tensorflow::serving::BatchPaddingPolicy::kPadUp,
          "The policy that a batch schduler is using when deciding what to do "
          "when, say, 18 requests need to be batched, but only 16 and 32 batch "
          "sizes are allowed. The following options are available. PAD_UP: pad "
          "to size 32. BATCH_DOWN: schedule a batch of size 16 and leave 2 "
          "requests in the batch buffer. MINIMIZE_TPU_COST_PER_REQUEST: a "
          "smarter greedy policy that chooses to either PAD_UP or BATCH_DOWN "
          "so as to minimize the TPU costs per real request. In this case, it "
          "would compare (batch_16_cost / 16) and (batch_32_cost / 18). "
          "WARNING: not all batch schedulers might support this option.");

namespace tensorflow {
namespace serving {

int GetNextAllowedBatchSize(int batch_size,
                            const std::vector<int32>& allowed_batch_sizes,
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

int32 GetPrevAllowedBatchSize(int batch_size,
                              const std::vector<int32>& allowed_batch_sizes,
                              bool disable_padding) {
  if (disable_padding || allowed_batch_sizes.empty()) {
    return batch_size;
  }

  DCHECK(absl::c_is_sorted(allowed_batch_sizes));
  DCHECK_GT(batch_size, 0);

  // First from the end allowed batch size not larger than batch_size.
  auto result = std::find_if(
      allowed_batch_sizes.rbegin(), allowed_batch_sizes.rend(),
      [&](int allowed_size) { return allowed_size <= batch_size; });

  if (result == allowed_batch_sizes.rend()) {
    // No such element exists.
    return batch_size;
  }

  return *result;
}

bool AbslParseFlag(absl::string_view text, BatchPaddingPolicy* out,
                   std::string* error) {
  if (text == "PAD_UP") {
    *out = BatchPaddingPolicy::kPadUp;
    return true;
  }
  if (text == "BATCH_DOWN") {
    *out = BatchPaddingPolicy::kBatchDown;
    return true;
  }
  if (text == "MINIMIZE_TPU_COST_PER_REQUEST") {
    *out = BatchPaddingPolicy::kMinimizeTpuCostPerRequest;
    return true;
  }
  *error = "unrecognized batching policy string";
  return false;
}

string AbslUnparseFlag(BatchPaddingPolicy in) {
  switch (in) {
    case BatchPaddingPolicy::kPadUp:
      return "PAD_UP";
    case BatchPaddingPolicy::kBatchDown:
      return "BATCH_DOWN";
    case BatchPaddingPolicy::kMinimizeTpuCostPerRequest:
      return "MINIMIZE_TPU_COST_PER_REQUEST";
  }
  CHECK(FATAL) << "Unrecognized BatchPaddingPolicy enum value.";  // Crash OK
}

}  // namespace serving
}  // namespace tensorflow
