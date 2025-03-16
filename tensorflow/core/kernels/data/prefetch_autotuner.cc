/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/prefetch_autotuner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/model.h"

namespace tensorflow {
namespace data {

PrefetchAutotuner::PrefetchAutotuner(
    int64_t initial_buffer_size, int64_t buffer_size_min,
    std::shared_ptr<model::RamBudgetManager> ram_budget_manager)
    : buffer_limit_(initial_buffer_size),
      ram_budget_manager_(ram_budget_manager) {
  if (initial_buffer_size == model::kAutotune) {
    mode_ = Mode::kUpswing;
    buffer_limit_ = std::max(int64_t{1}, buffer_size_min);
  }
}

namespace {
// Determines what strategy to use for increasing the buffer size limit. For
// limits less than the threshold, an exponential increase is used, while for
// limits greater than or equal to the threshold, a linear increase is used.
size_t kBufferLimitThreshold = 2048;
}  // namespace

void PrefetchAutotuner::SetElementSize(int64_t element_size_bytes) {
  // Once we know the element size we can allocate the right number of bytes for
  // the prefetch autotuner.
  // We tell the ram budget manager that we are going to allocate
  // `element_size_bytes` as we assume the buffer size will at least hold
  // one element
  if (ram_budget_manager_ && !ram_budget_manager_->RequestLegacyPrefetchBytes(
                                 element_size_bytes * buffer_limit_)) {
    LOG(WARNING)
        << "Prefetch autotuner tried to allocate "
        << element_size_bytes * buffer_limit_ << " bytes "
        << "after encountering the first element of size " << element_size_bytes
        << " bytes."
        << "This already causes the autotune ram budget to be exceeded. To "
        << "stay within the ram budget, either increase the ram budget or "
        << "reduce element size";
  }

  element_size_bytes_ = element_size_bytes;
}

void PrefetchAutotuner::RecordConsumption(size_t current_buffer_size) {
  switch (mode_) {
    case Mode::kDisabled:
      return;
    case Mode::kUpswing:
      if (static_cast<int64_t>(current_buffer_size) == buffer_limit_) {
        mode_ = Mode::kDownswing;
      }
      return;
    case Mode::kDownswing:
      if (current_buffer_size == 0) {
        if (!element_size_bytes_.has_value()) {
          // If `element_size_bytes_` has not been set,
          // do not optimize the `buffer_limit_` yet.
          return;
        }
        int64_t element_size_bytes = *element_size_bytes_;
        int64_t attempt_new_buffer_limit;
        if (buffer_limit_ >= static_cast<int64_t>(kBufferLimitThreshold)) {
          attempt_new_buffer_limit = buffer_limit_ + kBufferLimitThreshold;
        } else {
          attempt_new_buffer_limit = buffer_limit_ * 2;
        }
        int64_t delta_bytes =
            (attempt_new_buffer_limit - buffer_limit_) * element_size_bytes;

        // When `ram_budget_manager_` is a nullptr, update
        // the buffer size without checking available RAM
        // to match the legacy behavior before RamBudgetManager is introduced.
        // Otherwise, ask the `ram_budget_manager_` if there is enough memory to
        // allocate. If not, abort this optimization attempt
        if (!ram_budget_manager_ ||
            ram_budget_manager_->RequestLegacyPrefetchBytes(delta_bytes)) {
          // Overwrite the current limit
          buffer_limit_ = attempt_new_buffer_limit;
        }
        mode_ = Mode::kUpswing;
      }
      return;
  }
}

}  // namespace data
}  // namespace tensorflow
