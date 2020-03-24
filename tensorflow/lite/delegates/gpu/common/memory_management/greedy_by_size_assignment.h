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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_BY_SIZE_ASSIGNMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_BY_SIZE_ASSIGNMENT_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Assigns given tensors to offsets, using the following greedy algorithm:
// - We have tensor usage records of all intermideate tensors as an input. Each
// record consists of tensor size, first and last tasks, that use it. Let's call
// [first_task..last_task] a tensor usage interval;
// - Iterate through tensor usage records in non-increasing order of
// corresponding tensor sizes;
// - For each of these records consider already assigned tensors, which usage
// intervals intersect with usage interval of current tensor, and find the
// smallest gap in memory between them such, that current tensor fits into that
// gap;
// - If such a gap has been found, current tensor should be allocated into this
// gap. Otherwise we can allocate it after the rightmost tensor, which usage
// interval intersects with usage interval of current tensor. So we assign
// corresponding offset to current tensor and the tensor becomes assigned.
Status GreedyBySizeAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    OffsetsAssignment* assignment);

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
Status GreedyBySizeDistPriorityAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_GREEDY_BY_SIZE_ASSIGNMENT_H_
