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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_ABI_VERSION_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_ABI_VERSION_H_

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/cuda/cuda_runtime_abi_version.pb.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::cuda {

class CudaRuntimeAbiVersion : public RuntimeAbiVersion {
 public:
  explicit CudaRuntimeAbiVersion(const SemanticVersion& cuda_toolkit_version,
                                 const SemanticVersion& cudnn_version_,
                                 const SemanticVersion& cub_version);

  static absl::StatusOr<std::unique_ptr<CudaRuntimeAbiVersion> absl_nonnull>
  FromProto(const CudaRuntimeAbiVersionProto& proto);
  static absl::StatusOr<std::unique_ptr<CudaRuntimeAbiVersion>>
  FromSerializedProto(absl::string_view proto);

  absl::StatusOr<RuntimeAbiVersionProto> ToProto() const override;
  absl::StatusOr<PlatformId> platform_id() const override;

  absl::Status IsCompatibleWith(
      const ExecutableAbiVersion& executable_abi_version) const override;

  const SemanticVersion& cuda_toolkit_version() const {
    return cuda_toolkit_version_;
  }
  const SemanticVersion& cudnn_version() const { return cudnn_version_; }
  const SemanticVersion& cub_version() const { return cub_version_; }

 private:
  SemanticVersion cuda_toolkit_version_;
  SemanticVersion cudnn_version_;
  SemanticVersion cub_version_;
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_ABI_VERSION_H_
