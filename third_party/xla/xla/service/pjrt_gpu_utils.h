/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_PJRT_GPU_UTILS_H_
#define XLA_SERVICE_PJRT_GPU_UTILS_H_

#include "absl/base/nullability.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla::gpu {

GpuTargetConfig GetGpuTargetConfig(PjRtClient* absl_nonnull client);

}

#endif  // XLA_SERVICE_PJRT_GPU_UTILS_H_
