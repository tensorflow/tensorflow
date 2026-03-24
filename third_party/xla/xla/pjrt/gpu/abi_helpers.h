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

#ifndef XLA_PJRT_GPU_ABI_HELPERS_H_
#define XLA_PJRT_GPU_ABI_HELPERS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"

namespace xla::gpu {

// Creates a PjRtRuntimeAbiVersion from a PjRtRuntimeAbiVersionProto using the
// GPU PjRt plugin.
absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>>
PjRtRuntimeAbiVersionFromProto(const xla::PjRtRuntimeAbiVersionProto& proto);

// Creates a PjRtExecutableAbiVersion from a PjRtExecutableAbiVersionProto using
// the GPU PjRt plugin.
absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
PjRtExecutableAbiVersionFromProto(
    const xla::PjRtExecutableAbiVersionProto& proto);

}  // namespace xla::gpu

#endif  // XLA_PJRT_GPU_ABI_HELPERS_H_
