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

#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/tsl/util/env_var.h"

namespace xla {

bool UseTfrtGpuClient() {
  bool xla_pjrt_gpu_host_memory_preallocate;
  if (!tsl::ReadBoolFromEnvVar("USE_TFRT_GPU_CLIENT", false,
                               &xla_pjrt_gpu_host_memory_preallocate)
           .ok()) {
    return false;
  }
  return xla_pjrt_gpu_host_memory_preallocate;
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetXlaPjrtGpuClient(
    GpuClientOptions options) {
  // TODO(masonchang): Wrap the GPU Client inside the PJRT Sandwich
  if (UseTfrtGpuClient()) {
    return GetTfrtGpuClient(options);
  }
  return GetStreamExecutorGpuClient(options);
}

}  // namespace xla
