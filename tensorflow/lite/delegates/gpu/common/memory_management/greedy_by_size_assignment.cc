/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.h"

#include <algorithm>

#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"

namespace tflite {
namespace gpu {

Status GreedyBySizeAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    OffsetsAssignment* assignment) {
  const size_t num_tensors = usage_records.size();
  assignment->offsets.resize(num_tensors);
  assignment->total_size = 0;

  // Ordered records are to be sorted by size of corrseponding tensor.
  std::vector<TensorUsageWithIndex<size_t>> ordered_records;
  for (size_t i = 0; i < num_tensors; ++i) {
    ordered_records.emplace_back(&usage_records[i], i);
  }
  std::sort(ordered_records.begin(), ordered_records.end(), CompareBySize);

  // Vector of ids of already allocated tensors, ordered by offset.
  std::vector<size_t> ordered_allocs;

  for (const auto& rec_with_idx : ordered_records) {
    const TensorUsageRecord<size_t>* rec = rec_with_idx.usage_record;
    size_t best_diff = kNotAssigned;
    size_t best_offset = kNotAssigned;
    size_t prev_offset = 0;
    for (const auto& allocated_id : ordered_allocs) {
      if (usage_records[allocated_id].last_task < rec->first_task ||
          usage_records[allocated_id].first_task > rec->last_task) {
        // Tensor allocated_id has usage interval, that doesn't intersect with
        // current tensor's usage interval, so we skip it.
        continue;
      }
      size_t cur_offset = assignment->offsets[allocated_id];
      if (cur_offset >= prev_offset) {
        size_t diff = cur_offset - prev_offset;
        // Check, if current_tensor fits into the gap, located directly to the
        // left of tensor allocated_id offset, and that this gap is the smallest
        // of previously considered suitable gaps.
        if (diff >= rec->tensor_size && diff < best_diff) {
          best_diff = diff;
          best_offset = prev_offset;
        }
      }
      prev_offset = std::max(
          prev_offset, cur_offset + usage_records[allocated_id].tensor_size);
    }
    if (assignment->total_size < prev_offset) {
      return InternalError("Total size is wrong.");
    }

    // If no suitable gap found, we should allocate current tensor after the
    // rightmost tensor, which usage interval intersects with the current one.
    if (best_offset == kNotAssigned) {
      best_offset = prev_offset;
    }

    // Assign best_offset to the current tensor and find the correct place to
    // insert information about it into ordered_allocs to save the order.
    auto it = ordered_allocs.begin();
    while (it != ordered_allocs.end() &&
           assignment->offsets[*it] <= best_offset) {
      ++it;
    }
    ordered_allocs.insert(it, rec_with_idx.idx);
    assignment->offsets[rec_with_idx.idx] = best_offset;
    assignment->total_size =
        std::max(assignment->total_size, best_offset + rec->tensor_size);
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace tflite
