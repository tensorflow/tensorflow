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

namespace tensorflow {

PrefetchAutotuner::PrefetchAutotuner(int64 initial_buffer_size)
    : buffer_limit_(initial_buffer_size) {
  if (initial_buffer_size == kAutoTune) {
    mode_ = Mode::kUpswing;
    buffer_limit_ = 1;
  }
}

void PrefetchAutotuner::RecordConsumption(size_t current_buffer_size) {
  switch (mode_) {
    case Mode::kDisabled:
      return;
    case Mode::kUpswing:
      if (current_buffer_size == buffer_limit_) {
        mode_ = Mode::kDownswing;
      }
      return;
    case Mode::kDownswing:
      if (current_buffer_size == 0) {
        buffer_limit_ *= 2;  // Increase the buffer size.
        mode_ = Mode::kUpswing;
      }
      return;
  }
}

}  // namespace tensorflow
