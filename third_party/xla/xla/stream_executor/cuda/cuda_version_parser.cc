/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_version_parser.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {

absl::StatusOr<SemanticVersion> ParseCudaVersion(int cuda_version) {
  if (cuda_version < 0) {
    return absl::InvalidArgumentError("Version numbers cannot be negative!");
  }
  // The version is encoded as `1000 * major + 10 * minor`.
  // References:
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
  int major = cuda_version / 1000;
  int minor = (cuda_version % 1000) / 10;
  return SemanticVersion(major, minor, 0);
}

}  // namespace stream_executor
