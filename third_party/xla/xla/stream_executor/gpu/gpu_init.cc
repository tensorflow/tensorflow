/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_init.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/logging.h"

namespace stream_executor {

absl::Status ValidateGPUMachineManager() {
  return PlatformManager::PlatformWithName(GpuPlatformName()).status();
}

Platform* GPUMachineManager() {
  // Cache this result, it's on the critical path for light outside compilation
  // (and probably other things as well).
  static Platform* platform = [&] {
    absl::StatusOr<Platform*> p =
        PlatformManager::PlatformWithName(GpuPlatformName());
    if (!p.ok()) {
      LOG(FATAL) << "Could not find Platform with name " << GpuPlatformName();
    }
    return *p;
  }();

  return platform;
}

std::string GpuPlatformName() {
#if TENSORFLOW_USE_ROCM
  return "ROCM";
#else
  // This function will return "CUDA" even when building TF without GPU support
  // This is done to preserve existing functionality
  return "CUDA";
#endif
}

}  // namespace stream_executor
