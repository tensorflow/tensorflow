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

#include "xla/python/pjrt_ifrt/gpu_xla_executable_abi_version_serdes.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/Support/Casting.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_helpers.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/pjrt_ifrt/gpu_xla_executable_abi_version.h"
#include "xla/python/pjrt_ifrt/xla_executable_abi_version.h"

namespace xla {
absl::StatusOr<std::string> GpuXlaExecutableAbiVersionSerDes::Serialize(
    const xla::ifrt::Serializable& serializable,
    std::unique_ptr<xla::ifrt::SerializeOptions> options) {
  const auto& version =
      llvm::cast<xla::ifrt::XlaExecutableAbiVersion>(serializable);

  ASSIGN_OR_RETURN(xla::PjRtExecutableAbiVersionProto proto,
                   version.ExecutableAbiVersion().ToProto());
  std::string executable_abi_version;
  if (!proto.SerializeToString(&executable_abi_version)) {
    return absl::InternalError(
        "Failed to serialize PjRtExecutableAbiVersion to string.");
  }
  return executable_abi_version;
}
absl::StatusOr<std::unique_ptr<xla::ifrt::Serializable>>
GpuXlaExecutableAbiVersionSerDes::Deserialize(
    const std::string& serialized,
    std::unique_ptr<xla::ifrt::DeserializeOptions> options) {
  xla::PjRtExecutableAbiVersionProto proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InvalidArgumentError(
        "Failed to parse PjRtExecutableAbiVersion from string.");
  }
  ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtExecutableAbiVersion> runtime_abi_version,
      factory_function_(proto));
  return std::make_unique<GpuXlaExecutableAbiVersion>(
      std::move(runtime_abi_version));
}

[[maybe_unused]] char GpuXlaExecutableAbiVersionSerDes::ID = 0;

namespace {

absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
CApiPjRtExecutableAbiVersionFromProto(
    const xla::PjRtExecutableAbiVersionProto& proto) {
  ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kGpuPjrtName));
  return pjrt::CApiExecutableAbiVersionFromProto(proto, c_api);
}

bool register_gpu_abi_version_serdes =
    ([] {
      xla::ifrt::RegisterSerDes<xla::GpuXlaExecutableAbiVersion>(
          std::make_unique<xla::GpuXlaExecutableAbiVersionSerDes>(
            CApiPjRtExecutableAbiVersionFromProto));
    }(),
     true);

}  // namespace
}  // namespace xla
