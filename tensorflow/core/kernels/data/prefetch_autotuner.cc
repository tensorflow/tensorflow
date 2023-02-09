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
#include <optional>

#include "tensorflow/core/framework/model.h"

namespace tensorflow {
namespace data {

PrefetchAutotuner::PrefetchAutotuner(int64_t initial_buffer_size,
                                     int64_t buffer_size_min)
    : buffer_limit_(initial_buffer_size) {
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

void PrefetchAutotuner::RecordConsumption(
    size_t current_buffer_size, std::optional<int64_t> free_memory_bytes) {
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
        if (buffer_limit_ >= static_cast<int64_t>(kBufferLimitThreshold)) {
          VLOG(3) << "Increasing buffer limit from " << buffer_limit_ << " by "
                  << kBufferLimitThreshold << " to "
                  << buffer_limit_ + kBufferLimitThreshold;
          buffer_limit_ += kBufferLimitThreshold;
        } else {
          VLOG(3) << "Increasing buffer limit from " << buffer_limit_ << " to "
                  << buffer_limit_ * 2;
          buffer_limit_ *= 2;
        }
        // Use the element size and the free memory to compute the maximum
        // buffer size.
        if (free_memory_bytes.has_value() && element_size_bytes_.has_value() &&
            free_memory_bytes.value() > 0 && element_size_bytes_.value() > 0) {
          int64_t max_buffer_size =
              free_memory_bytes.value() / element_size_bytes_.value();
          if (buffer_limit_ > max_buffer_size) {
            VLOG(3) << "Increasing buffer limit to " << buffer_limit_
                    << " would result in memory usage that can exceed the free "
                       "memory value of "
                    << free_memory_bytes.value() << " bytes. Lowering it to "
                    << max_buffer_size << " elements.";
            buffer_limit_ = max_buffer_size;
          }
        }
        mode_ = Mode::kUpswing;
      }
      return;
  }
}

}  // namespace data
}  // namespace tensorflow
