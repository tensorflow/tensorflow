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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_PREFETCH_AUTOTUNER_H_
#define TENSORFLOW_CORE_KERNELS_DATA_PREFETCH_AUTOTUNER_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

// PrefetchAutotuner dynamically adjusts the buffer size of a prefetch iterator.
//
// PrefetchAutotuner attempts to find the minimum buffer size such that there is
// always at least 1 element in the prefetch queue every time the downstream
// iterator calls GetNext().
//
// One common failure mode of input pipelines is being throughput bound. No
// amount of prefetching can address that performance mode. In order to guard
// against this condition, PrefetchAutotuner will only increase the buffer_limit
// if the prefetching thread is able to successfully fill the buffer at its
// current size.
//
// Note: in the current implementation, we never decrease the buffer_limit().
// This should change in the future!
//
// PrefetchAutotuner is NOT thread safe.
class PrefetchAutotuner {
 public:
  explicit PrefetchAutotuner(int64 initial_buffer_size);

  int64 buffer_limit() const { return buffer_limit_; }

  void RecordConsumption(size_t current_buffer_size);
  void RecordEmpty() { RecordConsumption(0); }

 private:
  // PrefetchAutotuner operates as a state machine.
  enum class Mode {
    // Disables the autotuning.
    kDisabled,

    // We have increased the size of the buffer, and will transition to
    // kDownswing if we successfully fill the buffer.
    kUpswing,

    // We have successfully filled a buffer of this size. If we ever block the
    // downstream iterator, we should increase the buffer size.
    kDownswing,
  };

  int64 buffer_limit_;
  Mode mode_ = Mode::kDisabled;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PREFETCH_AUTOTUNER_H_
