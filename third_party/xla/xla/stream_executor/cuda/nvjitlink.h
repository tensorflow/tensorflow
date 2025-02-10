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

#ifndef XLA_STREAM_EXECUTOR_CUDA_NVJITLINK_H_
#define XLA_STREAM_EXECUTOR_CUDA_NVJITLINK_H_

#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"

namespace stream_executor {

using NvJitLinkVersion = std::tuple<unsigned, unsigned>;

// Returns the version of the loaded libnvjitlink library or an error otherwise.
// Returns 12.0 for libnvjitlink 12.0 through 12.2 since the version getter was
// added in 12.3.
absl::StatusOr<NvJitLinkVersion> GetNvJitLinkVersion();

struct NvJitLinkInput {
  enum class Type { kPtx, kCubin };
  Type type;
  absl::Span<const uint8_t> bytes;
};

// Compiles and links the given inputs using libnvjitlink.
// Compilation takes only place for inputs of type Type::kPtx.
absl::StatusOr<std::vector<uint8_t>> CompileAndLinkUsingLibNvJitLink(
    const CudaComputeCapability& cc, absl::Span<const NvJitLinkInput> inputs,
    GpuAsmOpts options, bool cancel_if_reg_spill);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_NVJITLINK_H_
