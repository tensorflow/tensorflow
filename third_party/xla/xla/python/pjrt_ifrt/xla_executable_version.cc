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

#include "xla/python/pjrt_ifrt/xla_executable_version.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/serdes_week_4_old_version_accessor.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/python/pjrt_ifrt/xla_executable_abi_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

[[maybe_unused]] char XlaExecutableVersion::ID = 0;

XlaExecutableVersion::XlaExecutableVersion(
    uint64_t platform_id, std::unique_ptr<XlaExecutableAbiVersion> abi_version)
    : platform_id(platform_id), abi_version(std::move(abi_version)) {}

absl::Status XlaExecutableVersion::IsCompatibleWith(
    const ExecutableVersion& other) const {
  if (this == &other) {
    return absl::OkStatus();
  }
  if (auto* other_xla_executable_version =
          llvm::dyn_cast<XlaExecutableVersion>(&other)) {
    if (platform_id != other_xla_executable_version->platform_id) {
      return absl::InvalidArgumentError(
          "Executable version is not compatible with current version");
    }
    // TODO(b/477624817): Add compatibility check for ABI version once we have
    // created xla runtime versions.
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      "Other ExecutableVersion is not XlaExecutableVersion");
}

absl::Status XlaExecutableVersion::ToProto(
    SerializedXlaExecutableVersion& executable_version_proto,
    SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for XlaExecutableVersion serialization"));
  }

  executable_version_proto.Clear();
  executable_version_proto.set_version_number(SerDesVersionNumber(0).value());
  executable_version_proto.set_platform_id(platform_id);
  if (abi_version) {
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::Serialized executable_abi_version_proto,
        xla::ifrt::Serialize(
            *abi_version,
            std::make_unique<xla::ifrt::SerializeOptions>(
                xla::ifrt::SerDesWeek4OldVersionAccessor::Get())));
    std::string executable_abi_version;
    if (!executable_abi_version_proto.SerializeToString(
            &executable_abi_version)) {
      return absl::InternalError(
          "Failed to serialize XlaExecutableAbiVersion to string.");
    }
    executable_version_proto.set_runtime_abi_version(
        std::move(executable_abi_version));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<XlaExecutableVersion>>
XlaExecutableVersion::FromProto(const SerializedXlaExecutableVersion& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version_number,
                     " for XlaExecutableVersion deserialization"));
  }
  if (proto.runtime_abi_version().empty()) {
    return std::make_unique<XlaExecutableVersion>(proto.platform_id(), nullptr);
  }
  xla::ifrt::Serialized executable_abi_version_proto;
  if (!executable_abi_version_proto.ParseFromString(
          proto.runtime_abi_version())) {
    return absl::InvalidArgumentError(
        "Failed to parse XlaExecutableAbiVersion from string.");
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<XlaExecutableAbiVersion>
                          xla_executable_runtime_abi_version,
                      xla::ifrt::Deserialize<XlaExecutableAbiVersion>(
                          executable_abi_version_proto,
                          std::make_unique<xla::ifrt::DeserializeOptions>()));

  return std::make_unique<XlaExecutableVersion>(
      proto.platform_id(), std::move(xla_executable_runtime_abi_version));
}

absl::StatusOr<std::unique_ptr<XlaExecutableVersion>> ToXlaExecutableVersion(
    std::unique_ptr<ExecutableVersion> executable_version) {
  if (!executable_version) {
    return absl::InvalidArgumentError("executable_version is null");
  }
  if (auto* xla_executable_version =
          llvm::dyn_cast<XlaExecutableVersion>(executable_version.get())) {
    executable_version.release();
    return std::unique_ptr<XlaExecutableVersion>(xla_executable_version);
  }
  return absl::InvalidArgumentError(
      "executable_version is not XlaExecutableVersion");
}

}  // namespace ifrt
}  // namespace xla
