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

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/pjrt_ifrt/tpu_xla_executable_abi_version.h"
#include "xla/python/pjrt_ifrt/xla_executable_abi_version.h"

namespace xla {

namespace tpu_xla_executable_abi_version_serdes {

absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
PjRtExecutableAbiVersionFromProto(
    const xla::PjRtExecutableAbiVersionProto& proto);

}  // namespace tpu_xla_executable_abi_version_serdes

namespace {

// IFRT SerDes implementation for XlaExecutableAbiVersion on TPU.
class TpuXlaExecutableAbiVersionSerDes
    : public llvm::RTTIExtends<TpuXlaExecutableAbiVersionSerDes,
                               xla::ifrt::SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::TpuXlaExecutableAbiVersion";
  }

  absl::StatusOr<absl::Cord> Serialize(
      const xla::ifrt::Serializable& serializable,
      std::unique_ptr<xla::ifrt::SerializeOptions> options) override {
    const auto& version =
        llvm::cast<xla::ifrt::XlaExecutableAbiVersion>(serializable);

    ASSIGN_OR_RETURN(xla::PjRtExecutableAbiVersionProto proto,
                     version.ExecutableAbiVersion().ToProto());
    absl::Cord executable_abi_version;
    if (!proto.SerializeToString(&executable_abi_version)) {
      return absl::InternalError(
          "Failed to serialize PjRtExecutableAbiVersion to string.");
    }
    return executable_abi_version;
  }

  absl::StatusOr<std::unique_ptr<xla::ifrt::Serializable>> Deserialize(
      const absl::Cord& serialized,
      std::unique_ptr<xla::ifrt::DeserializeOptions> options) override {
    xla::PjRtExecutableAbiVersionProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse PjRtExecutableAbiVersion from string.");
    }
    ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtExecutableAbiVersion> runtime_abi_version,
        tpu_xla_executable_abi_version_serdes::
            PjRtExecutableAbiVersionFromProto(proto));
    return std::make_unique<TpuXlaExecutableAbiVersion>(
        std::move(runtime_abi_version));
  }

  static char ID;  // NOLINT
};

bool register_tpu_abi_version_serdes =
    ([] {
      xla::ifrt::RegisterSerDes<TpuXlaExecutableAbiVersion>(
          std::make_unique<TpuXlaExecutableAbiVersionSerDes>());
    }(),
     true);

}  // namespace

[[maybe_unused]] char TpuXlaExecutableAbiVersion::ID = 0;
[[maybe_unused]] char TpuXlaExecutableAbiVersionSerDes::ID = 0;

}  // namespace xla
