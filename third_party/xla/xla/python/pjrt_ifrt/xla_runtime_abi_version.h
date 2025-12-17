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

#ifndef XLA_PYTHON_PJRT_IFRT_XLA_RUNTIME_ABI_VERSION_H_
#define XLA_PYTHON_PJRT_IFRT_XLA_RUNTIME_ABI_VERSION_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"

namespace xla {
namespace ifrt {

class XlaRuntimeAbiVersion
    : public llvm::RTTIExtends<XlaRuntimeAbiVersion, Serializable> {
 public:
  XlaRuntimeAbiVersion() = default;
  explicit XlaRuntimeAbiVersion(
      std::unique_ptr<xla::PjRtAbiVersion> runtime_abi_version)
      : runtime_abi_version_(std::move(runtime_abi_version)) {}

  bool IsCompatibleWith(const XlaRuntimeAbiVersion& other) const;

  std::unique_ptr<xla::PjRtAbiVersion> runtime_abi_version_;

  static char ID;  // NOLINT

  // Serialization and deserialization helper methods.
  absl::StatusOr<std::string> Serialize() const;
  static absl::StatusOr<std::unique_ptr<XlaRuntimeAbiVersion>> Deserialize(
      absl::string_view serialized);
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_XLA_RUNTIME_ABI_VERSION_H_
