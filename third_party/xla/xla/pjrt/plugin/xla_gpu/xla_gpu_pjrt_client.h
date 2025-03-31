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

#ifndef XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_PJRT_CLIENT_H_
#define XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_PJRT_CLIENT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"

namespace xla {

// Whether to use the TFRT GPU Client.
bool UseTfrtGpuClient();

// Public entry point to get an XLA:GPU PjRtClient
absl::StatusOr<std::unique_ptr<PjRtClient>> GetXlaPjrtGpuClient(
    GpuClientOptions options);

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_PJRT_CLIENT_H_
