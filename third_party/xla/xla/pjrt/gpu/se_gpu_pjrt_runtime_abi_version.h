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

#ifndef XLA_PJRT_GPU_SE_GPU_PJRT_RUNTIME_ABI_VERSION_H_
#define XLA_PJRT_GPU_SE_GPU_PJRT_RUNTIME_ABI_VERSION_H_

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version_resolver.h"

namespace xla {

// A PjRtRuntimeAbiVersion that wraps a StreamExecutor RuntimeAbiVersion.
class StreamExecutorGpuPjRtRuntimeAbiVersion : public PjRtRuntimeAbiVersion {
 public:
  explicit StreamExecutorGpuPjRtRuntimeAbiVersion(
      PjRtPlatformId platform_id,
      std::unique_ptr<stream_executor::RuntimeAbiVersion> absl_nonnull
      runtime_abi_version)
      : platform_id_(platform_id),
        runtime_abi_version_(std::move(runtime_abi_version)) {}

  PjRtPlatformId platform_id() const override { return platform_id_; }

  // Not implemented for CUDA - it's unclear what it means for a runtime version
  // to be compatible with another runtime version.
  absl::Status IsCompatibleWith(
      const PjRtRuntimeAbiVersion& runtime_abi_version) const override {
    return absl::UnimplementedError(
        "Runtime ABI version compatibility check is not implemented.");
  }

  // Returns OK if the runtime ABI is compatible with the executable ABI. In
  // other words the runtime with this runtime ABI version can in principle run
  // executables with the given executable ABI version.
  absl::Status IsCompatibleWith(
      const PjRtExecutableAbiVersion& executable_abi_version) const override;

  static absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>> FromProto(
      const PjRtRuntimeAbiVersionProto& proto,
      const stream_executor::RuntimeAbiVersionResolver&
          runtime_abi_version_resolver);
  absl::StatusOr<PjRtRuntimeAbiVersionProto> ToProto() const override;

 private:
  PjRtPlatformId platform_id_;
  std::unique_ptr<stream_executor::RuntimeAbiVersion> runtime_abi_version_;
};
}  // namespace xla

#endif  // XLA_PJRT_GPU_SE_GPU_PJRT_RUNTIME_ABI_VERSION_H_
