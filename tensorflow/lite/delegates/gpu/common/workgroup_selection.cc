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

#include "tensorflow/lite/delegates/gpu/common/workgroup_selection.h"

#include <cmath>
#include <set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

namespace {

template <typename T>
void AddCornerCases(const T& grid, int max_work_group_total_size,
                    const T& max_work_group_sizes,
                    WorkGroupSizeAlignment x_alignment,
                    WorkGroupSizeAlignment y_alignment,
                    WorkGroupSizeAlignment z_alignment,
                    std::vector<T>* work_groups) {
  for (int x = 1; x <= 4; ++x) {
    for (int y = 1; y <= 4; ++y) {
      for (int z = 1; z <= 4; ++z) {
        int wg_x = DivideRoundUp(grid.x, x);
        int wg_y = DivideRoundUp(grid.y, y);
        int wg_z = DivideRoundUp(grid.z, z);
        if (wg_x > max_work_group_sizes.x || wg_y > max_work_group_sizes.y ||
            wg_z > max_work_group_sizes.z ||
            wg_x * wg_y * wg_z > max_work_group_total_size) {
          continue;
        }
        if (x_alignment == WorkGroupSizeAlignment::PRECISE &&
            grid.x % wg_x != 0) {
          continue;
        }
        if (y_alignment == WorkGroupSizeAlignment::PRECISE &&
            grid.y % wg_y != 0) {
          continue;
        }
        if (z_alignment == WorkGroupSizeAlignment::PRECISE &&
            grid.z % wg_z != 0) {
          continue;
        }
        work_groups->push_back({wg_x, wg_y, wg_z});
      }
    }
  }

  // this will add at least {1, 1, 1} always.
  for (int x = 1; x <= 4; ++x) {
    for (int y = 1; y <= 4; ++y) {
      for (int z = 1; z <= 4; ++z) {
        if (x > max_work_group_sizes.x || y > max_work_group_sizes.y ||
            z > max_work_group_sizes.z ||
            x * y * z > max_work_group_total_size) {
          continue;
        }
        if (x_alignment == WorkGroupSizeAlignment::PRECISE && grid.x % x != 0) {
          continue;
        }
        if (y_alignment == WorkGroupSizeAlignment::PRECISE && grid.y % y != 0) {
          continue;
        }
        if (z_alignment == WorkGroupSizeAlignment::PRECISE && grid.z % z != 0) {
          continue;
        }
        work_groups->push_back({x, y, z});
      }
    }
  }
}

std::vector<int> GetDivisors(int number) {
  const int max_divisor = static_cast<int>(std::sqrt(number));
  std::vector<int> divisors;
  // we don't know the number of dividers, so it is just heuristic.
  divisors.reserve(max_divisor / 3 + 1);
  for (int i = 1; i <= max_divisor; ++i) {
    const int d = number / i;
    if (i * d == number) {
      divisors.push_back(i);
      if (d != i) {
        divisors.push_back(d);
      }
    }
  }
  return divisors;
}

std::vector<int> GetDivisorsForRange(int number, int range) {
  const int last_number = number + range;
  const int max_divisor = static_cast<int>(std::sqrt(last_number));
  std::set<int> divisors;
  for (int i = 1; i <= max_divisor; ++i) {
    const int reminder = number % i;
    // iterate through numbers that divisible by i in our range;
    const int first_number = number + (i - reminder) % i;
    if (first_number <= last_number) {
      divisors.insert(i);
    }
    for (int j = first_number; j <= last_number; j += i) {
      const int d = j / i;
      if (d != i) {
        divisors.insert(d);
      }
    }
  }
  return std::vector<int>(divisors.begin(), divisors.end());
}

}  // namespace

std::vector<int> GetPossibleSizes(int number,
                                  WorkGroupSizeAlignment z_alignment) {
  if (z_alignment == WorkGroupSizeAlignment::PRECISE) {
    // we will use for potential sizes, sizes that cover grid precisely
    // work group size * k (k is integer) == grid_size
    return GetDivisors(number);
  } else {
    // when we chose work group size we can use work group size that
    //   work group size * k (k is integer) != grid_size (slightly bigger)
    // so in this heuristic we trying to find potential size, that satisfies
    //   to this : work group size * k (k is integer) <= grid_size + 5
    //   and this : work group size * k (k is integer) >= grid_size
    return GetDivisorsForRange(number, 5);
  }
}

template <typename T>
std::vector<T> GenerateWorkGroupSizes(
    const T& grid, int min_work_group_total_size, int max_work_group_total_size,
    const T& max_work_group_sizes, WorkGroupSizeAlignment x_alignment,
    WorkGroupSizeAlignment y_alignment, WorkGroupSizeAlignment z_alignment) {
  std::vector<T> work_groups;
  work_groups.reserve(64);

  std::vector<int> sizes_x = GetPossibleSizes(grid.x, x_alignment);
  std::vector<int> sizes_y = GetPossibleSizes(grid.y, y_alignment);
  std::vector<int> sizes_z = GetPossibleSizes(grid.z, z_alignment);

  for (auto x : sizes_x) {
    if (x > max_work_group_sizes.x) continue;
    for (auto y : sizes_y) {
      if (y > max_work_group_sizes.y) continue;
      for (auto z : sizes_z) {
        if (z > max_work_group_sizes.z) continue;
        const int work_group_size = x * y * z;
        if (work_group_size < min_work_group_total_size ||
            work_group_size > max_work_group_total_size)
          continue;
        work_groups.push_back({x, y, z});
      }
    }
  }

  return work_groups;
}

// Specializations of GenerateWorkGroupSizes for int3 and uint3

template std::vector<int3> GenerateWorkGroupSizes(
    const int3& grid, int min_work_group_total_size,
    int max_work_group_total_size, const int3& max_work_group_sizes,
    WorkGroupSizeAlignment x_alignment, WorkGroupSizeAlignment y_alignment,
    WorkGroupSizeAlignment z_alignment);

template std::vector<uint3> GenerateWorkGroupSizes(
    const uint3& grid, int min_work_group_total_size,
    int max_work_group_total_size, const uint3& max_work_group_sizes,
    WorkGroupSizeAlignment x_alignment, WorkGroupSizeAlignment y_alignment,
    WorkGroupSizeAlignment z_alignment);

template <typename T>
void GenerateWorkGroupSizesAlignedToGrid(const T& grid,
                                         const T& max_work_group_size,
                                         const int max_work_group_total_size,
                                         std::vector<T>* work_groups) {
  auto alignment = WorkGroupSizeAlignment::PRECISE;
  *work_groups = GenerateWorkGroupSizes<T>(
      grid, /*min_work_group_total_size = */ 32, max_work_group_total_size,
      max_work_group_size, alignment, alignment, alignment);
  // If the grid parameter too small, method below cannot generate workgroups.
  if (work_groups->empty()) {
    AddCornerCases(grid, max_work_group_total_size, max_work_group_size,
                   alignment, alignment, alignment, work_groups);
  }
}

// Specializations of GenerateWorkGroupSizesAlignedToGrid for int3 and uint3

template void GenerateWorkGroupSizesAlignedToGrid(
    const int3& grid, const int3& max_work_group_size,
    const int max_work_group_total_size, std::vector<int3>* work_groups);

template void GenerateWorkGroupSizesAlignedToGrid(
    const uint3& grid, const uint3& max_work_group_size,
    const int max_work_group_total_size, std::vector<uint3>* work_groups);

}  // namespace gpu
}  // namespace tflite
