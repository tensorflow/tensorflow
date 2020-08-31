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

#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"

namespace tflite {
namespace gpu {
namespace {

// Set of usage records for all tensors assigned to the shared object, ordered
// by first_task.
using SharedObjectSchedule = std::set<TensorUsageRecord<size_t>>;

struct TaskBreadthWithId {
  size_t breadth;
  TaskId task_id;

  TaskBreadthWithId(size_t breadth, size_t task_id)
      : breadth(breadth), task_id(task_id) {}

  // Default order of TaskBreadthWithId is increasing order of their breadth.
  bool operator<(const TaskBreadthWithId& other) const {
    return breadth < other.breadth;
  }
};

}  // namespace

absl::Status GreedyByBreadthAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles(usage_records);

  // Task breadth is a sum of sizes of all tensors in its TaskProfile
  std::vector<TaskBreadthWithId> task_breadth;
  for (size_t task_id = 0; task_id < task_profiles.size(); ++task_id) {
    size_t breadth = 0;
    for (const auto& tensor_info : task_profiles[task_id]) {
      breadth += tensor_info.usage_record->tensor_size;
    }
    task_breadth.emplace_back(breadth, task_id);
  }

  assignment->object_sizes.clear();
  assignment->object_ids.assign(usage_records.size(), kNotAssigned);
  std::vector<SharedObjectSchedule> obj_schedules;

  // Iterate through all tasks in non-increasing order of their breadth.
  std::sort(task_breadth.rbegin(), task_breadth.rend());
  for (const auto& task : task_breadth) {
    // Iterate through all tensors, that must be allocated during the execution
    // of task, in non-increasing order of their tensor_size.
    for (const auto& tensor_info : task_profiles[task.task_id]) {
      if (assignment->object_ids[tensor_info.idx] != kNotAssigned) {
        continue;
      }
      const auto& rec = *tensor_info.usage_record;
      const size_t num_objects = obj_schedules.size();
      size_t best_object = num_objects;
      for (size_t obj_id = 0; obj_id < num_objects; ++obj_id) {
        // If size of current_object is worse than size of best found before, we
        // can skip it.
        if (best_object != num_objects) {
          const size_t best_size = assignment->object_sizes[best_object];
          const size_t cur_size = assignment->object_sizes[obj_id];
          if (best_size < rec.tensor_size) {
            if (cur_size <= best_size) {
              // best_size is smaller than tensor_size, but cur_size is even
              // smaller.
              continue;
            }
          } else if (cur_size < rec.tensor_size || cur_size >= best_size) {
            // best_size is larger or equal to tensor_size, and cur_size is
            // either smaller than tensor_size, or too large.
            continue;
          }
        }
        const auto& schedule = obj_schedules[obj_id];
        auto it = schedule.lower_bound(rec);
        bool update_best_object = true;
        if (it != schedule.end() && it->first_task <= rec.last_task) {
          // Some tensor, which usage interval intersects with current, already
          // assigned to this object.
          update_best_object = false;
        }
        if (update_best_object && it != schedule.begin()) {
          it--;
          if (it->last_task >= rec.first_task) {
            // Some tensor, which usage interval intersects with current,
            // already assigned to this object.
            update_best_object = false;
          }
        }
        if (update_best_object) {
          best_object = obj_id;
        }
      }
      if (best_object == num_objects) {
        // Create new shared object and assign current tensor to it.
        obj_schedules.push_back({rec});
        assignment->object_sizes.push_back(rec.tensor_size);
      } else {
        // Assign current tensor to best_object.
        obj_schedules[best_object].insert(rec);
        // Size of best_object can be increased, if it is smaller than
        // tensor_size.
        assignment->object_sizes[best_object] =
            std::max(assignment->object_sizes[best_object], rec.tensor_size);
      }
      assignment->object_ids[tensor_info.idx] = best_object;
    }
  }
  // In the end all tensors must be assigned to some objects.
  for (const auto& obj_id : assignment->object_ids) {
    if (obj_id == kNotAssigned) {
      return absl::InternalError("Error while calculating the assignment.");
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
