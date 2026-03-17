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

#ifndef XLA_STREAM_EXECUTOR_ABI_EXECUTABLE_ABI_VERSION_H_
#define XLA_STREAM_EXECUTOR_ABI_EXECUTABLE_ABI_VERSION_H_

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor {

// Represents the ABI version of a compiled XLA program, i.e. the platform
// specific libraries that the program was compiled against.
class ExecutableAbiVersion {
 public:
  ExecutableAbiVersion() = default;

  static absl::StatusOr<ExecutableAbiVersion> FromProto(
      const ExecutableAbiVersionProto& proto);

  absl::string_view platform_name() const;

  // Creates an `ExecutableAbiVersion` from the given `DeviceDescription`.
  // Currently only implemented for CUDA.
  static absl::StatusOr<ExecutableAbiVersion> FromDeviceDescription(
      const DeviceDescription& device_description);

  const ExecutableAbiVersionProto& proto() const { return proto_; }

 private:
  explicit ExecutableAbiVersion(ExecutableAbiVersionProto proto)
      : proto_(std::move(proto)) {}
  ExecutableAbiVersionProto proto_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ABI_EXECUTABLE_ABI_VERSION_H_
