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

#include "xla/stream_executor/rocm/rocm_version_parser.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {

absl::StatusOr<SemanticVersion> ParseRocmVersion(int rocm_version) {
  if (rocm_version < 0) {
    return absl::InvalidArgumentError("Version numbers cannot be negative.");
  }

  // The exact structure of the version number is not defined in the ROCm/
  // or HIP documentation, but `HIP_VERSION` is defined as the following:
  // #define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR \
  // * 100000 + HIP_VERSION_PATCH)
  int major = rocm_version / 10'000'000;
  int minor = (rocm_version % 10'000'000) / 100'000;
  int patch = rocm_version % 100'000;
  return SemanticVersion(major, minor, patch);
}

}  // namespace stream_executor
