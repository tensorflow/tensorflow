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

#include "xla/stream_executor/gpu/gpu_kernel_registry.h"

#include "xla/stream_executor/platform/platform_object_registry.h"

namespace stream_executor::gpu {

GpuKernelRegistry& GpuKernelRegistry::GetGlobalRegistry() {
  static auto registry =
      new GpuKernelRegistry(&PlatformObjectRegistry::GetGlobalRegistry());
  return *registry;
}

}  // namespace stream_executor::gpu
