/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

#include <algorithm>
#include <limits>
#include <set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

namespace {

std::vector<int2> Get2DWorkgroupsEqualTo128() {
  return {{128, 1}, {64, 2}, {32, 4}, {16, 8},
          {8, 16},  {4, 32}, {2, 64}, {1, 128}};
}

std::vector<int3> GenerateWorkGroupSizesXYMultipleOf(
    int multiplier, int3 grid, const KernelInfo& kernel_info,
    const GpuInfo& gpu_info, WorkGroupSizeAlignment z_alignment) {
  std::vector<int3> work_groups;
  work_groups.reserve(32);

  std::vector<int> possible_z_sizes = GetPossibleSizes(grid.z, z_alignment);

  for (int x = 1; x <= kernel_info.max_work_group_size; x *= 2) {
    for (int y = 1; y <= kernel_info.max_work_group_size; y *= 2) {
      int work_group_size_xy = x * y;
      if (work_group_size_xy % multiplier != 0 ||
          work_group_size_xy > kernel_info.max_work_group_size) {
        continue;
      }
      for (auto z : possible_z_sizes) {
        if (work_group_size_xy * z > kernel_info.max_work_group_size) {
          continue;
        }
        if (x <= gpu_info.GetMaxWorkGroupSizeForX() &&
            y <= gpu_info.GetMaxWorkGroupSizeForY() &&
            z <= gpu_info.GetMaxWorkGroupSizeForZ()) {
          work_groups.push_back({x, y, z});
        }
      }
    }
  }
  return work_groups;
}

std::vector<int3> GenerateWorkGroupSizesXMultipleOf(
    int multiplier, int3 grid, const KernelInfo& kernel_info,
    const GpuInfo& gpu_info, WorkGroupSizeAlignment z_alignment) {
  std::vector<int3> work_groups;
  work_groups.reserve(32);

  std::vector<int> possible_z_sizes = GetPossibleSizes(grid.z, z_alignment);
  std::vector<int> possible_y_sizes =
      GetPossibleSizes(grid.y, WorkGroupSizeAlignment::PRECISE);

  for (int x = multiplier;
       x <= kernel_info.max_work_group_size && x < grid.x + multiplier;
       x += multiplier) {
    for (auto y : possible_y_sizes) {
      for (auto z : possible_z_sizes) {
        if (x <= gpu_info.GetMaxWorkGroupSizeForX() &&
            y <= gpu_info.GetMaxWorkGroupSizeForY() &&
            z <= gpu_info.GetMaxWorkGroupSizeForZ() &&
            x * y * z <= kernel_info.max_work_group_size) {
          work_groups.push_back({x, y, z});
        }
      }
    }
  }
  return work_groups;
}

void GetWorkGroupsAlignedToGrid(const GpuInfo& gpu_info,
                                const KernelInfo& kernel_info, const int3& grid,
                                std::vector<int3>* work_groups) {
  int3 max_wg_size;
  max_wg_size.x = gpu_info.GetMaxWorkGroupSizeForX();
  max_wg_size.y = gpu_info.GetMaxWorkGroupSizeForY();
  max_wg_size.z = gpu_info.GetMaxWorkGroupSizeForZ();
  GenerateWorkGroupSizesAlignedToGrid(
      grid, max_wg_size, kernel_info.max_work_group_size, work_groups);
}

int GetPenalty(int grid_size, int group_size) {
  const int reminder = grid_size % group_size;
  return reminder == 0 ? 0 : group_size - reminder;
}

int GetPenalty(int2 grid_size, int2 group_size) {
  const int p_x = GetPenalty(grid_size.x, group_size.x);
  const int p_y = GetPenalty(grid_size.y, group_size.y);
  return p_x * grid_size.y + p_y * grid_size.x + p_x * p_y;
}

int GetMaxSizeWithMinPenalty(int size, int max_size) {
  int best_size = 128;
  int min_penalty = GetPenalty(size, best_size);
  for (int i = 2; i * 128 <= max_size; ++i) {
    if (GetPenalty(size, i * 128) == min_penalty) {
      best_size = i * 128;
    }
  }
  return best_size;
}

int2 GetMaxSizeWithMinPenalty(int2 size, int max_size) {
  std::vector<int2> base_groups = Get2DWorkgroupsEqualTo128();
  int min_penalty = std::numeric_limits<int>::max();
  for (const auto& group : base_groups) {
    min_penalty = std::min(GetPenalty(size, group), min_penalty);
  }
  for (const auto& group : base_groups) {
    for (int y = 1; y * group.y <= max_size; ++y) {
      int new_group_y = y * group.y;
      for (int x = 1; x * group.x <= max_size; ++x) {
        int new_group_x = x * group.x;
        if (new_group_x * new_group_y > max_size) {
          break;
        }
        if (GetPenalty(size, int2(new_group_x, new_group_y)) == min_penalty) {
          return int2(new_group_x, new_group_y);
        }
      }
    }
  }
  return int2(0, 0);
}

int GetBiggestDividerWithPriority(int number, int max_divider) {
  if (number % 8 == 0 && 8 <= max_divider) {
    return 8;
  }
  if (number % 4 == 0 && 4 <= max_divider) {
    return 4;
  }
  if (number % 2 == 0 && 2 <= max_divider) {
    return 2;
  }
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

int GetBiggestDivider(int number, int max_divider) {
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

int GetOptimalSizeForApple(int grid_size) {
  if (grid_size % 8 == 0 || grid_size % 8 >= 4 || grid_size >= 16) {
    return 8;
  }
  if (grid_size % 4 == 0 || grid_size % 4 >= 2 || grid_size >= 8) {
    return 4;
  }
  if (grid_size % 2 == 0 || grid_size >= 4) {
    return 2;
  }
  return 1;
}

int3 GetWorkGroupSizeForApple(const int3& grid_size) {
  int x_size = GetOptimalSizeForApple(grid_size.x);
  int y_size = GetOptimalSizeForApple(grid_size.y);
  int z_size = std::max(1, 32 / (x_size * y_size));
  z_size = std::min(z_size, static_cast<int>(grid_size.z));
  return {x_size, y_size, z_size};
}

}  // namespace

int3 GetWorkGroupXY128ConvLinear(const int3& grid) {
  int grid_z = GetBiggestDividerWithPriority(grid.z, 4);
  if (grid.x <= 128) {
    return int3(128, 1, grid_z);
  }
  int grid_x = GetMaxSizeWithMinPenalty(grid.x, 512 / grid_z);
  return {grid_x, 1, grid_z};
}

int3 GetWorkGroupXY128Conv(const int3& grid) {
  int grid_z = GetBiggestDividerWithPriority(grid.z, 4);
  if (grid.x <= 16 && grid.y <= 8) {
    return int3(16, 8, grid_z);
  }
  int2 grid_xy = GetMaxSizeWithMinPenalty(int2(grid.x, grid.y), 512 / grid_z);
  return int3(grid_xy.x, grid_xy.y, grid_z);
}

int3 GetWorkGroupXY128Simple(const int3& grid) { return int3(16, 8, 1); }

int3 GetWorkGroup(const int3& grid, int max_size) {
  int wg_z = GetBiggestDividerWithPriority(grid.z, 8);
  int wg_xy_size = max_size / wg_z;
  int wg_x = std::min(DivideRoundUp(grid.x, 2), wg_xy_size);
  int wg_y = std::min(wg_xy_size / wg_x, grid.y);
  return int3(wg_x, wg_y, wg_z);
}

int3 GetWorkGroupConv(const int3& grid, int max_size, int max_z_size) {
  int wg_z = GetBiggestDivider(grid.z, max_z_size);
  int wg_xy_size = std::min(256, max_size) / wg_z;
  int wg_x = std::min(grid.x, wg_xy_size);
  int wg_y = std::min(wg_xy_size / wg_x, grid.y);
  if (wg_y == grid.y && grid.y % 2 == 0) {
    wg_y = grid.y / 2;
  }
  return int3(wg_x, wg_y, wg_z);
}

void GetPossibleWorkGroupsXYMultipleOf(int multiplier, const GpuInfo& gpu_info,
                                       const KernelInfo& kernel_info,
                                       const int3& grid,
                                       WorkGroupSizeAlignment z_alignment,
                                       std::vector<int3>* work_groups) {
  *work_groups = GenerateWorkGroupSizesXYMultipleOf(
      multiplier, grid, kernel_info, gpu_info, z_alignment);
}

void GetPossibleWorkGroupsXMultipleOf(int multiplier, const GpuInfo& gpu_info,
                                      const KernelInfo& kernel_info,
                                      const int3& grid,
                                      WorkGroupSizeAlignment z_alignment,
                                      std::vector<int3>* work_groups) {
  *work_groups = GenerateWorkGroupSizesXMultipleOf(
      multiplier, grid, kernel_info, gpu_info, z_alignment);
}

bool XY128RequiresMoreWorkGroupsThenXY128Linear(int width, int height) {
  int planar_work_groups = DivideRoundUp(width * height, 128);
  auto base_work_groups = Get2DWorkgroupsEqualTo128();
  bool have_equal_work_groups = false;
  for (auto& work_group : base_work_groups) {
    int x_groups = DivideRoundUp(width, work_group.x);
    int y_groups = DivideRoundUp(height, work_group.y);
    int xy_groups = x_groups * y_groups;
    if (xy_groups == planar_work_groups) {
      have_equal_work_groups = true;
      break;
    }
  }
  return !have_equal_work_groups;
}

void GetPossibleWorkGroups(TuningType tuning_type, const GpuInfo& gpu_info,
                           const KernelInfo& kernel_info, const int3& grid,
                           std::vector<int3>* work_groups) {
  if (gpu_info.IsApple()) {
    work_groups->push_back(GetWorkGroupSizeForApple(grid));
    return;
  }
  switch (tuning_type) {
    case TuningType::kFast:
      work_groups->push_back(
          GetWorkGroup(grid, kernel_info.max_work_group_size));
      return;
    case TuningType::kExhaustive: {
      GetWorkGroupsAlignedToGrid(gpu_info, kernel_info, grid, work_groups);
      return;
    }
    default:
      work_groups->push_back({8, 4, 1});
      return;
  }
}

void GetPossibleWorkGroupsConv(TuningType tuning_type, const GpuInfo& gpu_info,
                               const KernelInfo& kernel_info, const int3& grid,
                               std::vector<int3>* work_groups) {
  if (gpu_info.IsApple()) {
    work_groups->push_back(GetWorkGroupSizeForApple(grid));
    return;
  }
  switch (tuning_type) {
    case TuningType::kFast: {
      int max_z_size = 16;
      if (gpu_info.IsAdreno()) {
        max_z_size = gpu_info.adreno_info.IsAdreno3xx() ? 16 : 64;
      }
      max_z_size = std::min(max_z_size, gpu_info.GetMaxWorkGroupSizeForZ());
      work_groups->push_back(
          GetWorkGroupConv(grid, kernel_info.max_work_group_size, max_z_size));
      return;
    }
    case TuningType::kExhaustive: {
      GetWorkGroupsAlignedToGrid(gpu_info, kernel_info, grid, work_groups);
      return;
    }
    default:
      work_groups->push_back({8, 4, 1});
      return;
  }
}

int3 GetFirstSuitableWorkGroup(const std::vector<int3>& wgs, int max_wg_size) {
  for (const auto& wg : wgs) {
    const int wg_size = wg.x * wg.y * wg.z;
    if (wg_size <= max_wg_size) {
      return wg;
    }
  }
  return {1, 1, 1};
}

}  // namespace gpu
}  // namespace tflite
