/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_INPUT_SPLIT_METADATA_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_INPUT_SPLIT_METADATA_H_

#include <algorithm>

#include "absl/container/fixed_array.h"

namespace tensorflow {
namespace serving {
namespace internal {
// InputSplitMetadata represents the task sizes of an batch-task after it's
// tailored according to queue status (`open_batch_remaining_slot` and
// `batch_size_limit`).
//
// This is an internal helper class, and the implementation is shared
// shared across different instantiations of internal::Queue<TaskType>
// in input-split mode (QueueOptions.enable_large_batch_splitting is true).
class InputSplitMetadata {
 public:
  InputSplitMetadata(int input_task_size, int open_batch_remaining_slot,
                     int batch_size_limit);

  // Returns underlying task sizes.
  const absl::FixedArray<int>& task_sizes() const;

  // Serializes task split metadata into a string for debugging.
  std::string DebugString() const;

 private:
  absl::FixedArray<int> generate_task_sizes(int input_task_size,
                                            int open_batch_remaining_slot,
                                            int batch_size_limit) const;

  const absl::FixedArray<int> task_sizes_;
};
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_INPUT_SPLIT_METADATA_H_
