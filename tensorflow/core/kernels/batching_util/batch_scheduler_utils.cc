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
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

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

}  // namespace serving
}  // namespace tensorflow
