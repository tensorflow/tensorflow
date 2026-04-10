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

#include "xla/pjrt/stream_executor_pjrt_abi_version.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
PjRtPlatformId StreamExecutorPjRtExecutableAbiVersion::platform_id() const {
  return platform_id_;
}

absl::StatusOr<PjRtExecutableAbiVersionProto>
StreamExecutorPjRtExecutableAbiVersion::ToProto() const {
  PjRtExecutableAbiVersionProto proto;
  proto.set_platform(platform_id_);
  if (!executable_abi_version_.proto().SerializeToString(
          proto.mutable_version())) {
    return absl::InternalError(
        "Failed to serialize ExecutableAbiVersionProto to string.");
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<StreamExecutorPjRtExecutableAbiVersion>>
StreamExecutorPjRtExecutableAbiVersion::FromProto(
    const PjRtExecutableAbiVersionProto& proto) {
  PjRtPlatformId pjrt_platform_id = proto.platform();
  stream_executor::ExecutableAbiVersionProto executable_abi_version_proto;
  if (!executable_abi_version_proto.ParseFromString(proto.version())) {
    return absl::InternalError(
        "Failed to parse ExecutableAbiVersionProto from string.");
  }
  ASSIGN_OR_RETURN(stream_executor::ExecutableAbiVersion executable_abi_version,
                   stream_executor::ExecutableAbiVersion::FromProto(
                       executable_abi_version_proto));
  return std::make_unique<StreamExecutorPjRtExecutableAbiVersion>(
      pjrt_platform_id, std::move(executable_abi_version));
}

}  // namespace xla
