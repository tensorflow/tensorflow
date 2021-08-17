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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_WORKGROUP_SELECTION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_WORKGROUP_SELECTION_H_

#include <vector>

namespace tflite {
namespace gpu {

// PRECISE assume that WorkGroupSize * k = GridSize;
// NO_ALIGNMENT no restrictions;
// We need PRECISE when we don't have check in kernel for boundaries
// If we have the check, we can use PRECISE or NO_ALIGNMENT as well.
enum class WorkGroupSizeAlignment { PRECISE, NO_ALIGNMENT };

std::vector<int> GetPossibleSizes(int number,
                                  WorkGroupSizeAlignment z_alignment);

// Specializations exist for int3 and uint3 in the .cc file

template <typename T>
std::vector<T> GenerateWorkGroupSizes(
    const T& grid, int min_work_group_total_size, int max_work_group_total_size,
    const T& max_work_group_sizes, WorkGroupSizeAlignment x_alignment,
    WorkGroupSizeAlignment y_alignment, WorkGroupSizeAlignment z_alignment);

template <typename T>
void GenerateWorkGroupSizesAlignedToGrid(const T& grid,
                                         const T& max_work_group_size,
                                         const int max_work_group_total_size,
                                         std::vector<T>* work_groups);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_WORKGROUP_SELECTION_H_
