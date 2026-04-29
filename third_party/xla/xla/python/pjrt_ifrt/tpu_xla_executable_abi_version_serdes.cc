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

#include "xla/python/pjrt_ifrt/tpu_xla_executable_abi_version_serdes.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/pjrt_ifrt/tpu_xla_executable_abi_version.h"
#include "xla/python/pjrt_ifrt/xla_executable_abi_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

TpuXlaExecutableAbiVersionSerDes::TpuXlaExecutableAbiVersionSerDes(
    FactoryFunction factory_function)
    : llvm::RTTIExtends<TpuXlaExecutableAbiVersionSerDes, xla::ifrt::SerDes>(),
      factory_function_(std::move(factory_function)) {}

absl::StatusOr<std::string> TpuXlaExecutableAbiVersionSerDes::Serialize(
    const xla::ifrt::Serializable& serializable,
    std::unique_ptr<xla::ifrt::SerializeOptions> options) {
  const auto& version =
      llvm::cast<xla::ifrt::XlaExecutableAbiVersion>(serializable);

  TF_ASSIGN_OR_RETURN(xla::PjRtExecutableAbiVersionProto proto,
                      version.ExecutableAbiVersion().ToProto());
  std::string executable_abi_version;
  if (!proto.SerializeToString(&executable_abi_version)) {
    return absl::InternalError(
        "Failed to serialize PjRtExecutableAbiVersion to string.");
  }
  return executable_abi_version;
}

absl::StatusOr<std::unique_ptr<xla::ifrt::Serializable>>
TpuXlaExecutableAbiVersionSerDes::Deserialize(
    const std::string& serialized,
    std::unique_ptr<xla::ifrt::DeserializeOptions> options) {
  xla::PjRtExecutableAbiVersionProto proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InvalidArgumentError(
        "Failed to parse PjRtExecutableAbiVersion from string.");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtExecutableAbiVersion> runtime_abi_version,
      factory_function_(proto));
  return std::make_unique<TpuXlaExecutableAbiVersion>(
      std::move(runtime_abi_version));
}

[[maybe_unused]] char TpuXlaExecutableAbiVersion::ID = 0;
[[maybe_unused]] char TpuXlaExecutableAbiVersionSerDes::ID = 0;

}  // namespace xla
