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

#include "xla/stream_executor/cuda/cuda_runtime_abi_version.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::cuda {

CudaRuntimeAbiVersion::CudaRuntimeAbiVersion(
    const SemanticVersion& cuda_toolkit_version,
    const SemanticVersion& cudnn_version, const SemanticVersion& cub_version)
    : cuda_toolkit_version_(cuda_toolkit_version),
      cudnn_version_(cudnn_version),
      cub_version_(cub_version) {}

absl::StatusOr<std::unique_ptr<CudaRuntimeAbiVersion>>
CudaRuntimeAbiVersion::FromProto(const CudaRuntimeAbiVersionProto& proto) {
  ASSIGN_OR_RETURN(
      SemanticVersion cuda_toolkit_version,
      SemanticVersion::ParseFromString(proto.cuda_toolkit_version()));
  ASSIGN_OR_RETURN(SemanticVersion cudnn_version,
                   SemanticVersion::ParseFromString(proto.cudnn_version()));
  ASSIGN_OR_RETURN(SemanticVersion cub_version,
                   SemanticVersion::ParseFromString(proto.cub_version()));
  return std::make_unique<CudaRuntimeAbiVersion>(cuda_toolkit_version,
                                                 cudnn_version, cub_version);
}

absl::StatusOr<std::unique_ptr<CudaRuntimeAbiVersion>>
CudaRuntimeAbiVersion::FromSerializedProto(absl::string_view proto) {
  CudaRuntimeAbiVersionProto cuda_proto;
  if (!cuda_proto.ParseFromString(proto)) {
    return absl::InternalError(
        "Failed to parse CudaRuntimeAbiVersionProto from string.");
  }
  return FromProto(cuda_proto);
}

absl::StatusOr<RuntimeAbiVersionProto> CudaRuntimeAbiVersion::ToProto() const {
  CudaRuntimeAbiVersionProto cuda_proto;
  cuda_proto.set_cuda_toolkit_version(cuda_toolkit_version_.ToString());
  cuda_proto.set_cudnn_version(cudnn_version_.ToString());
  cuda_proto.set_cub_version(cub_version_.ToString());

  std::string serialized_proto;
  if (!cuda_proto.SerializeToString(&serialized_proto)) {
    return absl::InternalError(
        "Failed to serialize CudaRuntimeAbiVersionProto to string.");
  }

  RuntimeAbiVersionProto proto;
  proto.set_platform_name(kCudaPlatformId->ToName());
  proto.set_platform_specific_version(std::move(serialized_proto));
  return proto;
}

absl::StatusOr<PlatformId> CudaRuntimeAbiVersion::platform_id() const {
  return kCudaPlatformId;
}

absl::Status CudaRuntimeAbiVersion::IsCompatibleWith(
    const ExecutableAbiVersion& executable_abi_version) const {
  const ExecutableAbiVersionProto& executable_proto =
      executable_abi_version.proto();

  if (executable_proto.platform_name() != kCudaPlatformId->ToName()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Platform name mismatch. Expected ", kCudaPlatformId->ToName(),
        ", but got ", executable_proto.platform_name()));
  }

  if (!executable_proto.has_cuda_platform_version()) {
    return absl::FailedPreconditionError(
        "ExecutableAbiVersionProto does not have CUDA platform version.");
  }

  ASSIGN_OR_RETURN(
      SemanticVersion required_cuda_toolkit_version,
      SemanticVersion::ParseFromString(
          executable_proto.cuda_platform_version().cuda_toolkit_version()));
  if (cuda_toolkit_version_ < required_cuda_toolkit_version) {
    return absl::FailedPreconditionError(absl::StrCat(
        "CUDA toolkit version mismatch. Running with version ",
        cuda_toolkit_version_,
        ", but executable requires >= ", required_cuda_toolkit_version));
  }

  ASSIGN_OR_RETURN(
      SemanticVersion required_cudnn_version,
      SemanticVersion::ParseFromString(
          executable_proto.cuda_platform_version().cudnn_version()));
  if (cudnn_version_ < required_cudnn_version) {
    return absl::FailedPreconditionError(absl::StrCat(
        "cuDNN version mismatch. Running with version ", cudnn_version_,
        ", but executable requires >= ", required_cudnn_version));
  }

  ASSIGN_OR_RETURN(SemanticVersion required_cub_version,
                   SemanticVersion::ParseFromString(
                       executable_proto.cuda_platform_version().cub_version()));
  if (cub_version_ < required_cub_version) {
    return absl::FailedPreconditionError(absl::StrCat(
        "CUB version mismatch. Running with version ", cub_version_,
        ", but executable requires >= ", required_cub_version));
  }

  return absl::OkStatus();
}

}  // namespace stream_executor::cuda
