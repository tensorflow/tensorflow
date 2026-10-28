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

#include "xla/backends/gpu/runtime/kernel_thunk_utils.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {

std::string KernelThunkToString(
    absl::string_view kernel_name, absl::string_view profile_annotation,
    const LaunchDimensions& launch_dimensions,
    std::optional<stream_executor::ClusterDim> cluster_dim, int64_t shmem_bytes,
    bool use_pdl) {
  return absl::StrFormat(
      "kernel=%s, profile_annotation=%s, launch dimensions=%s, "
      "cluster_dim=%s, shmem_bytes=%d%s",
      kernel_name, profile_annotation, launch_dimensions.ToString(),
      cluster_dim.has_value() ? cluster_dim->ToString() : "nullopt",
      shmem_bytes, use_pdl ? ", use_pdl=true" : "");
}

}  // namespace xla::gpu
