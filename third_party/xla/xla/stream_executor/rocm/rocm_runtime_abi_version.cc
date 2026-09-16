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

#include "xla/stream_executor/rocm/rocm_runtime_abi_version.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"

namespace stream_executor::gpu {

absl::StatusOr<PlatformId> ROCmRuntimeAbiVersion::platform_id() const {
  return rocm::kROCmPlatformId;
}

absl::StatusOr<RuntimeAbiVersionProto> ROCmRuntimeAbiVersion::ToProto() const {
  return RuntimeAbiVersionProto();
}

absl::Status ROCmRuntimeAbiVersion::IsCompatibleWith(
    const ExecutableAbiVersion& executable_abi_version) const {
  const ExecutableAbiVersionProto& executable_proto =
      executable_abi_version.proto();

  if (executable_proto.platform_name() != rocm::kROCmPlatformId->ToName()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Platform name mismatch. Expected ", rocm::kROCmPlatformId->ToName(),
        ", but got ", executable_proto.platform_name()));
  }

  return absl::OkStatus();
}

}  // namespace stream_executor::gpu
