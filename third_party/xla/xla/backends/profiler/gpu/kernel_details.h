/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_KERNEL_DETAILS_H_
#define XLA_BACKENDS_PROFILER_GPU_KERNEL_DETAILS_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"

namespace xla::profiler {

struct KernelDetails {
  // The number of registers used in this kernel.
  uint32_t registers_per_thread;
  // The amount of shared memory space used by a thread block.
  uint32_t static_shared_memory_usage;
  // The amount of dynamic memory space used by a thread block.
  uint32_t dynamic_shared_memory_usage;
  // X-dimension of a thread block.
  uint32_t block_x;
  // Y-dimension of a thread block.
  uint32_t block_y;
  // Z-dimension of a thread block.
  uint32_t block_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;

  // ID of the hardware channel on which this operation ran.
  uint32_t channel_id = -1;
  // CUpti_ChannelType of the channel above.
  int8_t channel_type = 0;  // CUPTI_CHANNEL_TYPE_INVALID
};

inline std::string ToXStat(const KernelDetails& kernel_info,
                           double occupancy_pct) {
  return absl::StrCat(
      "regs:", kernel_info.registers_per_thread,
      " static_shared:", kernel_info.static_shared_memory_usage,
      " dynamic_shared:", kernel_info.dynamic_shared_memory_usage,
      " grid:", kernel_info.grid_x, ",", kernel_info.grid_y, ",",
      kernel_info.grid_z, " block:", kernel_info.block_x, ",",
      kernel_info.block_y, ",", kernel_info.block_z,
      " occ_pct:", occupancy_pct);
}

}  // namespace xla::profiler

#endif  // XLA_BACKENDS_PROFILER_GPU_KERNEL_DETAILS_H_
