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

#include "xla/python/pjrt_ifrt/xla_executable_abi_version.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

[[maybe_unused]] char XlaExecutableAbiVersion::ID = 0;

absl::StatusOr<std::string> XlaExecutableAbiVersion::Serialize(
    std::unique_ptr<SerializeOptions> options) const {
  if (options == nullptr) {
    options = std::make_unique<SerializeOptions>();
  }
  TF_ASSIGN_OR_RETURN(xla::ifrt::Serialized proto,
                      xla::ifrt::Serialize(*this, std::move(options)));
  std::string result;
  if (!proto.SerializeToString(&result)) {
    return absl::InternalError(
        "Failed to serialize XlaExecutableAbiVersion to string.");
  }
  return result;
}

absl::StatusOr<std::unique_ptr<XlaExecutableAbiVersion>>
XlaExecutableAbiVersion::Deserialize(const std::string& serialized) {
  xla::ifrt::Serialized proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InvalidArgumentError(
        "Failed to parse XlaExecutableAbiVersion from string.");
  }
  return xla::ifrt::Deserialize<XlaExecutableAbiVersion>(
      proto, std::make_unique<xla::ifrt::DeserializeOptions>());
}

}  // namespace ifrt
}  // namespace xla
