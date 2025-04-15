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

#include "xla/stream_executor/cuda/cuda_compute_capability.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace stream_executor {

absl::StatusOr<CudaComputeCapability> CudaComputeCapability::FromString(
    absl::string_view cuda_arch_name) {
  std::vector<absl::string_view> split = absl::StrSplit(cuda_arch_name, '.');
  if (split.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid CUDA architecture name: ", cuda_arch_name));
  }

  int major, minor;
  if (!absl::SimpleAtoi(split[0], &major) ||
      !absl::SimpleAtoi(split[1], &minor)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid CUDA architecture name: ", cuda_arch_name));
  }
  return CudaComputeCapability(major, minor);
}
}  // namespace stream_executor
