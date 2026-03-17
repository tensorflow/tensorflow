/* Copyright 2022 The OpenXLA Authors.

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

#include <cstdlib>
#include <utility>

#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/tests/pjrt_client_registry.h"

namespace xla {
namespace {

// Register a GPU PjRt client for tests.
const bool kUnused =
    (RegisterPjRtClientTestFactory([]() {
       GpuAllocatorConfig gpu_config;
       gpu_config.kind = GpuAllocatorConfig::Kind::kDefault;
       gpu_config.preallocate = false;
       gpu_config.collective_memory_size = 0;
       GpuClientOptions options;
       options.allocator_config = std::move(gpu_config);
       options.use_tfrt_gpu_client = true;
       return GetXlaPjrtGpuClient(options);
     }),
     true);

}  // namespace
}  // namespace xla
