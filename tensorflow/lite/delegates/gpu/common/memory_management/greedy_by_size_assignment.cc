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
#include <cstddef>
#include <iterator>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {

struct SizeDistPriorityInfo {
  // - Tensor with leftmost position in positional maximums vector has higher
  // priority;
  // - If two tensors have equal position, the one, that has usage interval with
  // smallest positive distance (best_dist) to some of already assigned tensors,
  // has higher priority;
  // - If two tensors have equal position and best_dist, the one with greater
  // tensor_size has higher priority.
  bool operator>(const SizeDistPriorityInfo& other) const {
    return position < other.position ||
           (position == other.position &&
            (best_dist < other.best_dist || (best_dist == other.best_dist &&
                                             tensor_size > other.tensor_size)));
  }

  // Recalculate best distance and best object, based on precalculated distances
  // in vector dist.
  void RecalcBestDist() {
    best_dist = kNotAssigned;
    for (size_t obj_id = 0; obj_id < dist.size(); ++obj_id) {
      if (dist[obj_id] < best_dist) {
        best_dist = dist[obj_id];
        best_object = obj_id;
      }
    }
  }

  size_t position;
  size_t tensor_size;
  std::vector<size_t> dist;
  size_t best_dist;
  size_t best_object;
  size_t tensor_usage_id;
};

}  // namespace

absl::Status GreedyBySizeAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    size_t base_addr_align_bytes, OffsetsAssignment* assignment) {
  const size_t num_tensors = usage_records.size();
  assignment->offsets.resize(num_tensors);
  assignment->total_size = 0;

  // Ordered records are to be sorted by size of corresponding tensor.
  std::vector<TensorUsageWithIndex<size_t>> ordered_records;
  for (size_t i = 0; i < num_tensors; ++i) {
    ordered_records.emplace_back(&usage_records[i], i);
  }
  std::stable_sort(ordered_records.begin(), ordered_records.end(),
                   CompareBySize);

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
          prev_offset,
          AlignByN(cur_offset + usage_records[allocated_id].tensor_size,
                   base_addr_align_bytes));
    }
    // prev_offset should be no more than the total size with additional
    // alignment boundary introduced in AlignByN. Per object alignment added is
    // no more than (base_addr_align_bytes - 1).
    if (assignment->total_size +
            ordered_allocs.size() * (base_addr_align_bytes - 1) <
        prev_offset) {
      return absl::InternalError("Total size is wrong.");
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
  return absl::OkStatus();
}

// Assigns given tensors to shared objects, using the following greedy
// algorithm:
// - We have tensor usage records of all intermideate tensors as an input. Each
// record consists of tensor size, first and last tasks, that use it. Let's call
// [first_task..last_task] a tensor usage interval;
// - Distance between two usage intervals is the absolute difference between
// closest tasks in their intervals. If two usage intervals don't intersect,
// than the distance between them is positive;
// - Calculate positional maximums vector, e.g. the vector of lower bounds on
// size of each shared object;
// - For each tensor find the rightmost positional maximum, that is greater or
// equal, than current tensor's size (call it position);
// - Iterate through all tensors in non-decreasing order of their
// SizeDistPriority (described above);
// - For every such tensor, assign it to the object, that already has tensor,
// which usage interval has the smallest existing positive distance to the
// current tensor's usage interval (this distance and object id are already
// precalculated in its SizeDistPriority record). Size of the chosen object can
// possible increase;
// - If there are several such objects, use the largest one;
// - If there are no suitable shared objects, assign current tensor to the new
// object with size equal to current tensor's size;
// - Modify SizeDistPriority records of tensors, that haven't been assigned yet,
// to reflect distance changes after that assignment.
absl::Status GreedyBySizeDistPriorityAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
  std::vector<size_t> positional_max =
      CalculatePositionalMaximums(usage_records);

  size_t num_records = usage_records.size();
  std::vector<SizeDistPriorityInfo> priority_info(num_records);
  for (size_t rec_id = 0; rec_id < usage_records.size(); ++rec_id) {
    priority_info[rec_id].tensor_usage_id = rec_id;
    priority_info[rec_id].tensor_size = usage_records[rec_id].tensor_size;

    // No objects have been created yet.
    priority_info[rec_id].best_dist = kNotAssigned;
    priority_info[rec_id].best_object = kNotAssigned;

    // Find the rightmost positional maximum, that is greater or
    size_t pos = 0;
    while (pos < positional_max.size() &&
           positional_max[pos] >= priority_info[rec_id].tensor_size) {
      ++pos;
    }
    if (pos == 0) {
      return absl::InternalError("Variable pos must be positive.");
    }
    priority_info[rec_id].position = pos - 1;
  }

  assignment->object_sizes.clear();
  assignment->object_ids.assign(num_records, kNotAssigned);
  for (size_t it = 0; it < num_records; ++it) {
    size_t best_info_id = kNotAssigned;
    for (size_t info_id = 0; info_id < num_records; ++info_id) {
      if (assignment->object_ids[priority_info[info_id].tensor_usage_id] !=
          kNotAssigned) {
        // Tensor already assigned.
        continue;
      }
      if (best_info_id == kNotAssigned ||
          priority_info[info_id] > priority_info[best_info_id]) {
        best_info_id = info_id;
      }
    }
    if (best_info_id == kNotAssigned) {
      // During each iteration we assign exactly one of the tensors, so some not
      // yet assigned tensors must exist.
      return absl::InternalError("Invalid value for variable best_info_id.");
    }

    size_t best_rec_id = priority_info[best_info_id].tensor_usage_id;
    size_t best_obj_id = priority_info[best_info_id].best_object;
    bool new_object = false;
    if (priority_info[best_info_id].best_dist == kNotAssigned) {
      // No suitable shared object, so we create a new one.
      new_object = true;
      best_obj_id = assignment->object_sizes.size();
      assignment->object_ids[best_rec_id] = best_obj_id;
      assignment->object_sizes.push_back(
          usage_records[best_rec_id].tensor_size);
    } else {
      // Assign tensor best_rec_id to the already existing object best_obj_id.
      assignment->object_ids[best_rec_id] = best_obj_id;
      assignment->object_sizes[best_obj_id] =
          std::max(assignment->object_sizes[best_obj_id],
                   usage_records[best_rec_id].tensor_size);
    }

    // Modify SizeDistPriority records of tensors, that haven't been assigned
    // yet, to reflect distance changes after that assignment.
    for (size_t info_id = 0; info_id < num_records; ++info_id) {
      // SizeDistPriority record info_id contains priority of tensor rec_id.
      size_t rec_id = priority_info[info_id].tensor_usage_id;

      if (assignment->object_ids[rec_id] != kNotAssigned) {
        // Tensor rec_id is already assigned.
        continue;
      }
      if (!new_object &&
          priority_info[info_id].dist[best_obj_id] == kNotAssigned) {
        // Tensor rec_id intersects with some of the tensors, that are assigned
        // to object best_obj_id.
        continue;
      }

      size_t dist = kNotAssigned;
      if (usage_records[rec_id].last_task <
          usage_records[best_rec_id].first_task) {
        dist = usage_records[best_rec_id].first_task -
               usage_records[rec_id].last_task;
      } else if (usage_records[best_rec_id].last_task <
                 usage_records[rec_id].first_task) {
        dist = usage_records[rec_id].first_task -
               usage_records[best_rec_id].last_task;
      }

      if (new_object) {
        // best_rec_id is the only tensor, assigned to the new object.
        priority_info[info_id].dist.push_back(dist);
      } else if (dist == kNotAssigned) {
        // Usage intervals of tensors rec_id and best_rec_id intersect. So
        // rec_id can't be assigned to best_obj_id anymore.
        priority_info[info_id].dist[best_obj_id] = kNotAssigned;
        if (priority_info[info_id].best_object == best_obj_id) {
          // best_obj_id was the best shared object for tensor rec_id, but now
          // it's not suitable anymore, so we need some recalculation.
          priority_info[info_id].RecalcBestDist();
        }
      } else {
        // Update distance, because it has probably been changed.
        priority_info[info_id].dist[best_obj_id] =
            std::min(priority_info[info_id].dist[best_obj_id], dist);
      }
      if (dist < priority_info[info_id].best_dist) {
        // Update best distance and best object for tensor rec_id.
        priority_info[info_id].best_dist = dist;
        priority_info[info_id].best_object = best_obj_id;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
