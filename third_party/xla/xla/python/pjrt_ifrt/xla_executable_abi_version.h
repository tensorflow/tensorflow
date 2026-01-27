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

#ifndef XLA_PYTHON_PJRT_IFRT_XLA_EXECUTABLE_ABI_VERSION_H_
#define XLA_PYTHON_PJRT_IFRT_XLA_EXECUTABLE_ABI_VERSION_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/python/ifrt/serdes.h"

namespace xla {
namespace ifrt {

class XlaExecutableAbiVersion
    : public llvm::RTTIExtends<XlaExecutableAbiVersion, Serializable> {
 public:
  XlaExecutableAbiVersion() = default;
  explicit XlaExecutableAbiVersion(
      std::unique_ptr<PjRtExecutableAbiVersion> executable_abi_version)
      : executable_abi_version_(std::move(executable_abi_version)) {}

  PjRtExecutableAbiVersion& ExecutableAbiVersion() const {
    return *executable_abi_version_;
  }

  // Serializes this object using ifrt::SerDes.
  absl::StatusOr<std::string> Serialize(
      std::unique_ptr<SerializeOptions> options = nullptr) const;

  // Deserializes from string using ifrt::SerDes.
  static absl::StatusOr<std::unique_ptr<XlaExecutableAbiVersion>> Deserialize(
      const std::string& serialized);

  static char ID;  // NOLINT

 private:
  std::unique_ptr<PjRtExecutableAbiVersion> executable_abi_version_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_XLA_EXECUTABLE_ABI_VERSION_H_
