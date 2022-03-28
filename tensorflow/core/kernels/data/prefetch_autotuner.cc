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

#include <memory>

#include "tensorflow/core/framework/model.h"

namespace tensorflow {
namespace data {

PrefetchAutotuner::PrefetchAutotuner(std::shared_ptr<model::Model> model,
                                     std::shared_ptr<model::Node> prefetch_node,
                                     int64_t initial_buffer_size,
                                     int64_t buffer_size_min)
    : model_(model),
      prefetch_node_(prefetch_node),
      buffer_limit_(initial_buffer_size) {
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
        double new_buffer_limit = buffer_limit_;
        if (buffer_limit_ >= static_cast<int64_t>(kBufferLimitThreshold)) {
          new_buffer_limit += kBufferLimitThreshold;
        } else {
          new_buffer_limit *= 2;
        }
        if (model_ != nullptr && prefetch_node_ != nullptr &&
            model_->AllocateBufferedBytes(
                prefetch_node_->MaximumBufferedBytes() *
                (new_buffer_limit - buffer_limit_) /
                static_cast<double>(buffer_limit_))) {
          buffer_limit_ = new_buffer_limit;
        }
        mode_ = Mode::kUpswing;
      }
      return;
  }
}

}  // namespace data
}  // namespace tensorflow
