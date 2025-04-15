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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_ASM_COMPILER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_ASM_COMPILER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cubin_or_ptx_image.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"

namespace stream_executor {
// Compiles the given PTX string using a statically determined compilation
// method and returns the resulting machine code (i.e. a cubin) as a byte array.
// The generated cubin matches the compute capabilities provided by 'cc'.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
absl::StatusOr<std::vector<uint8_t>> CompileGpuAsm(
    const CudaComputeCapability& cc, const std::string& ptx_contents,
    GpuAsmOpts options, bool cancel_if_reg_spill = false);

// Temporary overload for users outside of XLA that still use the old API.
inline absl::StatusOr<std::vector<uint8_t>> CompileGpuAsm(
    int cc_major, int cc_minor, const char* ptx_contents, GpuAsmOpts options,
    bool cancel_if_reg_spill = false) {
  return CompileGpuAsm(CudaComputeCapability(cc_major, cc_minor),
                       std::string(ptx_contents), options, cancel_if_reg_spill);
}

// Same as CompileGpuAsm, but caches the result, and returns unowned view of
// the compiled binary.
//
// A copy of the string provided in ptx will be made.
absl::StatusOr<absl::Span<const uint8_t>> CompileGpuAsmOrGetCached(
    const CudaComputeCapability& cc, const std::string& ptx_contents,
    GpuAsmOpts compilation_options);

// Bundles the GPU machine code (cubins) and PTX if requested and returns the
// resulting binary (i.e. a fatbin) as a byte array.
absl::StatusOr<std::vector<uint8_t>> BundleGpuAsm(
    std::vector<CubinOrPTXImage> images, GpuAsmOpts options);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_ASM_COMPILER_H_
