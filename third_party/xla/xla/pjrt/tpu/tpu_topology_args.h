/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TPU_TPU_TOPOLOGY_ARGS_H_
#define XLA_PJRT_TPU_TPU_TOPOLOGY_ARGS_H_

#include <array>
#include <string>

#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/tpu/tpu_platform_type.pb.h"
#include "xla/pjrt/tpu/tpu_versions.pb.h"

namespace xla {

struct TpuTopologyArgs {
  TpuVersion version;
  std::string variant = "";
  TpuPlatformType platform_type = TpuPlatformType::TPU_PLATFORM_TYPE_HARDWARE;
  std::string chip_config_name = "default";
  xla::PjRtDeviceDimensions chips_per_host_bounds = {1, 1, 1};
  xla::PjRtDeviceDimensions host_bounds = {1, 1, 1};
  std::array<bool, 3> wrap = {false, false, false};
  bool twist = false;
  bool enhanced_barrier_enabled = true;
};

}  // namespace xla

#endif  // XLA_PJRT_TPU_TPU_TOPOLOGY_ARGS_H_
