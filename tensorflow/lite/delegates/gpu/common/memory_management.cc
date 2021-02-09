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

#include "tensorflow/lite/delegates/gpu/common/memory_management.h"

#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/memory_management/equality_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/greedy_in_order_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/naive_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {

size_t TotalSize(const ObjectsAssignment<size_t>& assignment) {
  return std::accumulate(assignment.object_sizes.begin(),
                         assignment.object_sizes.end(), static_cast<size_t>(0));
}

}  // namespace

OffsetsAssignment ObjectsToOffsets(
    const ObjectsAssignment<size_t>& obj_assignment) {
  size_t num_tensors = obj_assignment.object_ids.size();
  size_t num_objects = obj_assignment.object_sizes.size();
  OffsetsAssignment result = {/*offsets=*/std::vector<size_t>(num_tensors),
                              /*total_size=*/0};
  std::vector<size_t> ids_to_offset(num_objects);
  for (size_t i = 0; i < num_objects; ++i) {
    ids_to_offset[i] = result.total_size;
    result.total_size += obj_assignment.object_sizes[i];
  }
  for (size_t i = 0; i < num_tensors; ++i) {
    result.offsets[i] = ids_to_offset[obj_assignment.object_ids[i]];
  }
  return result;
}

absl::Status BestGreedy(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
  RETURN_IF_ERROR(
      GreedyBySizeDistPriorityAssignment(usage_records, assignment));
  ObjectsAssignment<size_t> assignment_by_breadth;
  if (GreedyByBreadthAssignment(usage_records, &assignment_by_breadth).ok() &&
      TotalSize(assignment_by_breadth) < TotalSize(*assignment)) {
    std::swap(*assignment, assignment_by_breadth);
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<size_t>* assignment,
    const UsageGraph* reallocation_graph) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignmentWithHash(usage_records, assignment);
    case MemoryStrategy::GREEDY_IN_ORDER:
      return GreedyInOrderAssignment(usage_records, assignment,
                                     reallocation_graph);
    case MemoryStrategy::GREEDY_BY_BREADTH:
      return GreedyByBreadthAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_BY_SIZE:
      return GreedyBySizeDistPriorityAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_BEST:
      return BestGreedy(usage_records, assignment);
    case MemoryStrategy::MINCOSTFLOW:
      return MinCostFlowAssignment(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<BHWC>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<BHWC>* assignment,
    const UsageGraph* reallocation_graph) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignmentWithHash(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<uint2>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<uint2>* assignment,
    const UsageGraph* reallocation_graph) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_IN_ORDER:
      return GreedyInOrderAssignmentMultidimensional(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<uint3>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<uint3>* assignment,
    const UsageGraph* reallocation_graph) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY_IN_ORDER:
      return GreedyInOrderAssignmentMultidimensional(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

absl::Status AssignOffsetsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    const MemoryStrategy& strategy, OffsetsAssignment* assignment,
    const UsageGraph* reallocation_graph) {
  if (strategy == MemoryStrategy::GREEDY_BY_SIZE) {
    return GreedyBySizeAssignment(usage_records, assignment);
  }
  ObjectsAssignment<size_t> objects_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, strategy, &objects_assignment, reallocation_graph));
  *assignment = ObjectsToOffsets(objects_assignment);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
