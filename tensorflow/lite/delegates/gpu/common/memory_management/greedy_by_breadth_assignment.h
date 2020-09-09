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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_BY_BREADTH_ASSIGNMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_BY_BREADTH_ASSIGNMENT_H_

#include <stddef.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Assigns given tensors to shared objects, using the following greedy
// algorithm:
// - We have tensor usage records of all intermideate tensors as an input. Each
// record consists of tensor size, first and last tasks, that use it. Let's call
// [first_task..last_task] a tensor usage interval;
// - For each task calculate its TaskProfile. By breadth of the task we
// understand sum of sizes of all tensors in its TaskProfile;
// - Iterate through all tasks in non-increasing order of breadth;
// - For each of these tasks iterate through all tensors in its TaskProfile in
// non-increasing order of tensor_size;
// - For every such tensor usage record find a shared object, that is not
// assigned to some tensors, which usage intervals intersect with usage interval
// of current tensor;
// - If there are no suitable shared objects, assign current tensor to the new
// object with size equal to current tensor's size;
// - If there are suitable objects with size greater than or equal to current
// tensor’s size, assign current tensor to the smallest of them;
// - If there are suitable objects only with size less than current tensor’s
// size, assign current tensor to the largest of them and increase its size.
absl::Status GreedyByBreadthAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_BY_BREADTH_ASSIGNMENT_H_
