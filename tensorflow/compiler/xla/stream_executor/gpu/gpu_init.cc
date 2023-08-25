/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_init.h"

#include <string>

#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace stream_executor {

tsl::Status ValidateGPUMachineManager() {
  return MultiPlatformManager::PlatformWithName(GpuPlatformName()).status();
}

Platform* GPUMachineManager() {
  auto result = MultiPlatformManager::PlatformWithName(GpuPlatformName());
  if (!result.ok()) {
    LOG(FATAL) << "Could not find Platform with name " << GpuPlatformName();
    return nullptr;
  }

  return result.value();
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
