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

#include "tensorflow/core/kernels/batching_util/input_split_metadata.h"

#include <algorithm>

#include "absl/container/fixed_array.h"
#include "absl/strings/str_join.h"

namespace tensorflow {
namespace serving {
namespace internal {
namespace {
int compute_task_size_from_open_batch(int input_task_size,
                                      int open_batch_remaining_slot,
                                      int batch_size_limit) {
  return (open_batch_remaining_slot > 0)
             ? (input_task_size + batch_size_limit - open_batch_remaining_slot)
             : input_task_size;
}

int compute_head_task_size(int input_task_size, int open_batch_remaining_slot,
                           int batch_size_limit) {
  if (open_batch_remaining_slot == 0) {
    return std::min(input_task_size, batch_size_limit);
  }
  return std::min(open_batch_remaining_slot, input_task_size);
}

int compute_tail_task_size(int task_size_from_open_batch, int input_task_size,
                           int open_batch_remaining_slot,
                           int batch_size_limit) {
  int tail_task_size;
  if (input_task_size <= open_batch_remaining_slot) {
    tail_task_size = input_task_size;
  } else {
    tail_task_size = task_size_from_open_batch % batch_size_limit;
    if (tail_task_size == 0) {
      tail_task_size = batch_size_limit;
    }
  }
  return tail_task_size;
}

int compute_num_batches(int task_size_from_open_batch, int batch_size_limit) {
  return (task_size_from_open_batch + batch_size_limit - 1) / batch_size_limit;
}
}  // namespace

InputSplitMetadata::InputSplitMetadata(int input_task_size,
                                       int open_batch_remaining_slot,
                                       int batch_size_limit)
    : task_sizes_(generate_task_sizes(
          input_task_size, open_batch_remaining_slot, batch_size_limit)) {}

const absl::FixedArray<int>& InputSplitMetadata::task_sizes() const {
  return task_sizes_;
}

std::string InputSplitMetadata::DebugString() const {
  return absl::StrJoin(task_sizes_, ", ");
}

absl::FixedArray<int> InputSplitMetadata::generate_task_sizes(
    int input_task_size, int open_batch_remaining_slot,
    int batch_size_limit) const {
  const int task_size_from_open_batch = compute_task_size_from_open_batch(
      input_task_size, open_batch_remaining_slot, batch_size_limit);

  const int num_batches =
      compute_num_batches(task_size_from_open_batch, batch_size_limit);

  absl::FixedArray<int> task_sizes(num_batches, batch_size_limit);

  task_sizes.front() = compute_head_task_size(
      input_task_size, open_batch_remaining_slot, batch_size_limit);

  task_sizes.back() =
      compute_tail_task_size(task_size_from_open_batch, input_task_size,
                             open_batch_remaining_slot, batch_size_limit);

  return task_sizes;
}
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
