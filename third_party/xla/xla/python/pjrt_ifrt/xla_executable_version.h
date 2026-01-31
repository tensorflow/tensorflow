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

#ifndef XLA_PYTHON_PJRT_IFRT_XLA_EXECUTABLE_VERSION_H_
#define XLA_PYTHON_PJRT_IFRT_XLA_EXECUTABLE_VERSION_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/serdes_default_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/python/pjrt_ifrt/xla_executable_abi_version.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace ifrt {

struct XlaExecutableVersion
    : llvm::RTTIExtends<XlaExecutableVersion, ExecutableVersion> {
  XlaExecutableVersion() = default;
  XlaExecutableVersion(uint64_t platform_id,
                       std::unique_ptr<XlaExecutableAbiVersion> abi_version);

  // ID that identifies the platform (CPU/GPU/TPU). This corresponds to
  // xla::PjRtPlatformId.
  uint64_t platform_id;
  std::unique_ptr<XlaExecutableAbiVersion> abi_version;

  absl::Status IsCompatibleWith(const ExecutableVersion& other) const override;

  absl::Status ToProto(SerializedXlaExecutableVersion& executable_version_proto,
                       SerDesVersion version = SerDesVersion::current()) const;

  absl::StatusOr<SerializedXlaExecutableVersion> ToProto(
      SerDesVersion version = SerDesVersion::current()) const {
    SerializedXlaExecutableVersion proto;
    TF_RETURN_IF_ERROR(ToProto(proto, version));
    return proto;
  }

  static absl::StatusOr<std::unique_ptr<XlaExecutableVersion>> FromProto(
      const SerializedXlaExecutableVersion& proto);

  static char ID;  // NOLINT
};

absl::StatusOr<std::shared_ptr<const XlaExecutableVersion>>
ToXlaExecutableVersion(
    std::shared_ptr<const ExecutableVersion> executable_version);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_XLA_EXECUTABLE_VERSION_H_
