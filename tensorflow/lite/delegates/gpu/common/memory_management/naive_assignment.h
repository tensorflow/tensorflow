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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_NAIVE_ASSIGNMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_NAIVE_ASSIGNMENT_H_

#include <stddef.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Implements memory management with a naive algorithm.
//
// The problem of memory management is NP-complete. This implements a
// naive algorithm that assigns each tensor to a separate object in memory.
template <typename TensorSizeT>
absl::Status NaiveAssignment(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    ObjectsAssignment<TensorSizeT>* assignment) {
  assignment->object_sizes.resize(usage_records.size());
  assignment->object_ids.assign(usage_records.size(), kNotAssigned);
  for (size_t i = 0; i < usage_records.size(); i++) {
    auto& record = usage_records[i];
    assignment->object_ids[i] = i;
    assignment->object_sizes[i] = record.tensor_size;
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_NAIVE_ASSIGNMENT_H_
