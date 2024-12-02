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
#ifndef XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_HELPERS_H_
#define XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_HELPERS_H_
#include <string_view>

#include "absl/status/status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {
// Checks whether ptxas log contains errors related to register allocation.
bool IsPtxRegisterAllocationError(std::string_view);

// Identifies errors in the ptxas log and creates an error status.
// `architecture` is the name of the GPU architecture, e.g. "sm_80" and is only
// used for error message generation. If `cancel_if_reg_spill` is true, then a
// register spill warning will be treated as an error, otherwise it will be
// ignored.
absl::Status CreateErrorFromPTXASLog(std::string_view log,
                                     std::string_view architecture,
                                     bool cancel_if_reg_spill);

// Warns if the ptxas version should be upgraded.
void WarnIfBadPtxasVersion(std::string_view method,
                           const CudaComputeCapability& cc,
                           SemanticVersion compiler_version);
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_HELPERS_H_
