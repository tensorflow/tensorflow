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

#include <optional>

#include "tensorflow/core/platform/mem.h"
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
  // The autotuner is enabled if the `initial_buffer_size` is `AUTOTUNE`. It is
  // disabled otherwise.
  explicit PrefetchAutotuner(int64_t initial_buffer_size,
                             int64_t buffer_size_min);

  // Returns the upper limit of the buffer size, i.e. the number of buffer
  // elements the buffer is allowed to hold.
  int64_t buffer_limit() const { return buffer_limit_; }
  // Returns the element size in bytes.
  int64_t element_size() const {
    return element_size_bytes_.has_value() ? element_size_bytes_.value() : 0;
  }
  // Records the size of each element used to compute the max buffer size.
  void RecordElementSize(int64_t element_size_bytes) {
    element_size_bytes_ = element_size_bytes;
  }
  // Tells the autotuner the `current_buffer_size`. The autotuner uses this
  // information to monitor the ongoing values of the buffer size and increases
  // its limit if necessary up to the `free_memory_bytes`.
  void RecordConsumption(size_t current_buffer_size,
                         std::optional<int64_t> free_memory_bytes);
  void RecordConsumption(size_t current_buffer_size) {
    RecordConsumption(current_buffer_size, std::nullopt);
  }
  // Tells the autotuner that the buffer is empty. This may trigger the
  // autotuner to increase its buffer limit.
  void RecordEmpty() { RecordConsumption(0, port::GetMemoryInfo().free); }

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

  // This is the upper limit of the buffer size.
  int64_t buffer_limit_;
  // This is the size of each element in the buffer in bytes.
  std::optional<int64_t> element_size_bytes_;

  // Operating mode/state of the autotuner.
  Mode mode_ = Mode::kDisabled;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PREFETCH_AUTOTUNER_H_
