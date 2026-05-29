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

#include "xla/pjrt/gpu/se_gpu_pjrt_runtime_abi_version.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/pjrt/stream_executor_pjrt_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/abi/runtime_abi_version_resolver.h"

namespace xla {

absl::Status StreamExecutorGpuPjRtRuntimeAbiVersion::IsCompatibleWith(
    const PjRtExecutableAbiVersion& abi_version) const {
  if (abi_version.platform_id() != platform_id()) {
    return absl::InvalidArgumentError(
        "The executable ABI version platform ID does not match the runtime "
        "ABI version platform ID.");
  }

  // `abi_version` can be a PjRt wrapper instance, therefore we need to
  // serialize to proto that we can parse to a
  // `StreamExecutorPjRtExecutableAbiVersion`.
  ASSIGN_OR_RETURN(PjRtExecutableAbiVersionProto abi_version_proto,
                   abi_version.ToProto());
  ASSIGN_OR_RETURN(
      std::unique_ptr<StreamExecutorPjRtExecutableAbiVersion>
          se_executable_abi_version,
      StreamExecutorPjRtExecutableAbiVersion::FromProto(abi_version_proto));

  if (se_executable_abi_version == nullptr) {
    return absl::InvalidArgumentError(
        "The executable ABI version is not a "
        "StreamExecutorPjRtExecutableAbiVersion.");
  }

  return runtime_abi_version_->IsCompatibleWith(
      se_executable_abi_version->executable_abi_version());
}

absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>>
StreamExecutorGpuPjRtRuntimeAbiVersion::FromProto(
    const PjRtRuntimeAbiVersionProto& proto,
    const stream_executor::RuntimeAbiVersionResolver&
        runtime_abi_version_resolver) {
  stream_executor::RuntimeAbiVersionProto se_runtime_abi_version_proto;
  if (!se_runtime_abi_version_proto.ParseFromString(proto.version())) {
    return absl::InternalError(
        "Failed to parse the StreamExecutor RuntimeAbiVersionProto from the "
        "PjRtRuntimeAbiVersionProto.");
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<stream_executor::RuntimeAbiVersion> runtime_abi_version,
      runtime_abi_version_resolver.GetRuntimeAbiVersion(
          se_runtime_abi_version_proto));

  return std::make_unique<xla::StreamExecutorGpuPjRtRuntimeAbiVersion>(
      proto.platform(), std::move(runtime_abi_version));
}

absl::StatusOr<PjRtRuntimeAbiVersionProto>
StreamExecutorGpuPjRtRuntimeAbiVersion::ToProto() const {
  PjRtRuntimeAbiVersionProto proto;
  proto.set_platform(platform_id());
  ASSIGN_OR_RETURN(
      stream_executor::RuntimeAbiVersionProto se_runtime_abi_version_proto,
      runtime_abi_version_->ToProto());
  proto.set_version(se_runtime_abi_version_proto.SerializeAsString());
  return proto;
}

}  // namespace xla
