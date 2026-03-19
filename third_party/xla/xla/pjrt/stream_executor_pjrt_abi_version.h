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

#ifndef XLA_PJRT_STREAM_EXECUTOR_PJRT_ABI_VERSION_H_
#define XLA_PJRT_STREAM_EXECUTOR_PJRT_ABI_VERSION_H_

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/stream_executor/abi/executable_abi_version.h"

namespace xla {

// Represents the executable ABI version for Stream Executor-based platforms,
// e.g. CUDA, ROCm, and SYCL. It's basically a wrapper around
// stream_executor::ExecutableAbiVersion.
class StreamExecutorPjRtExecutableAbiVersion : public PjRtExecutableAbiVersion {
 public:
  explicit StreamExecutorPjRtExecutableAbiVersion(
      PjRtPlatformId platform_id,
      stream_executor::ExecutableAbiVersion executable_abi_version)
      : platform_id_(platform_id),
        executable_abi_version_(std::move(executable_abi_version)) {}

  PjRtPlatformId platform_id() const override;
  absl::StatusOr<PjRtExecutableAbiVersionProto> ToProto() const override;

  const stream_executor::ExecutableAbiVersion& executable_abi_version() const {
    return executable_abi_version_;
  }

  static absl::StatusOr<std::unique_ptr<StreamExecutorPjRtExecutableAbiVersion>>
  FromProto(const PjRtExecutableAbiVersionProto& proto);

 private:
  PjRtPlatformId platform_id_;
  stream_executor::ExecutableAbiVersion executable_abi_version_;
};

}  // namespace xla

#endif  // XLA_PJRT_STREAM_EXECUTOR_PJRT_ABI_VERSION_H_
