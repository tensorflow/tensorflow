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

#include "xla/python/pjrt_ifrt/xla_runtime_abi_version.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/serdes_week_4_old_version_accessor.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

[[maybe_unused]] char XlaRuntimeAbiVersion::ID = 0;

bool XlaRuntimeAbiVersion::IsCompatibleWith(
    const XlaRuntimeAbiVersion& other) const {
  return runtime_abi_version_->IsCompatible(*other.runtime_abi_version_);
}

absl::StatusOr<std::string> XlaRuntimeAbiVersion::Serialize() const {
  TF_ASSIGN_OR_RETURN(
      xla::ifrt ::Serialized serialized_runtime_abi_version,
      xla::ifrt::Serialize(
          *this, std::make_unique<xla::ifrt::SerializeOptions>(
                     xla::ifrt::SerDesWeek4OldVersionAccessor::Get())));
  std::string serialized;
  if (!serialized_runtime_abi_version.SerializeToString(&serialized)) {
    return absl::InternalError("Failed to serialize runtime ABI version.");
  }
  return serialized;
}

absl::StatusOr<std::unique_ptr<XlaRuntimeAbiVersion>>
XlaRuntimeAbiVersion::Deserialize(const absl::string_view serialized) {
  Serialized serialized_runtime_abi_version;
  if (!serialized_runtime_abi_version.ParseFromString(serialized)) {
    return absl::InvalidArgumentError(
        "Failed to parse Serialized runtime ABI version");
  }
  return xla::ifrt::Deserialize<XlaRuntimeAbiVersion>(
      serialized_runtime_abi_version,
      std::make_unique<xla::ifrt::DeserializeOptions>());
}

}  // namespace ifrt
}  // namespace xla
