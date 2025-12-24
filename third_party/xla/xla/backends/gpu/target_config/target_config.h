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

#ifndef XLA_BACKENDS_GPU_TARGET_CONFIG_TARGET_CONFIG_H_
#define XLA_BACKENDS_GPU_TARGET_CONFIG_TARGET_CONFIG_H_

#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla::gpu {

// Returns the GpuTargetConfigProto for the given GPU model.
absl::StatusOr<stream_executor::GpuTargetConfigProto> GetGpuTargetConfig(
    const std::string& gpu_model);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TARGET_CONFIG_TARGET_CONFIG_H_
