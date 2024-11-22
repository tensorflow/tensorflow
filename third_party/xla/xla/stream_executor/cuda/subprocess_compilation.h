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

#ifndef XLA_STREAM_EXECUTOR_CUDA_SUBPROCESS_COMPILATION_H_
#define XLA_STREAM_EXECUTOR_CUDA_SUBPROCESS_COMPILATION_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cubin_or_ptx_image.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {
// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array. The generated cubin matches the compute
// capabilities provided by `cc`.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
absl::StatusOr<std::vector<uint8_t>> CompileGpuAsmUsingPtxAs(
    const CudaComputeCapability& cc, std::string_view ptx_contents,
    GpuAsmOpts options, bool cancel_if_reg_spill = false);

// Finds the CUDA executable with the given binary_name
// The path <preferred_cuda_dir>/bin is checked first, afterwards some other
// predefined locations are being checked.
//
// A binary is only considered if it is of at least `minimum_version` and not
// in `excluded_versions`.
absl::StatusOr<std::string> FindCudaExecutable(
    std::string_view binary_name, std::string_view preferred_cuda_dir,
    SemanticVersion minimum_version,
    absl::Span<const SemanticVersion> excluded_versions);

// Same as above, but with no version constraints.
absl::StatusOr<std::string> FindCudaExecutable(
    std::string_view binary_name, std::string_view preferred_cuda_dir);

// Runs tool --version and parses its version string. All the usual CUDA
// tools are supported.
absl::StatusOr<SemanticVersion> GetToolVersion(std::string_view tool_path);

// On NVIDIA GPUs, returns the version of the ptxas command line tool.
absl::StatusOr<SemanticVersion> GetAsmCompilerVersion(
    std::string_view preferred_cuda_dir);

// On NVIDIA GPUs, returns the version of the nvlink command line tool.
absl::StatusOr<SemanticVersion> GetNvLinkVersion(
    std::string_view preferred_cuda_dir);

// Bundles the GPU machine code (cubins) and PTX if requested and returns the
// resulting binary (i.e. a fatbin) as a byte array.
absl::StatusOr<std::vector<uint8_t>> BundleGpuAsmUsingFatbin(
    std::vector<CubinOrPTXImage> images, GpuAsmOpts options);

// Links the given CUBIN `images` using nvlink.
absl::StatusOr<std::vector<uint8_t>> LinkUsingNvlink(
    stream_executor::CudaComputeCapability cc,
    std::string_view preferred_cuda_dir, std::vector<CubinOrPTXImage> images);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_SUBPROCESS_COMPILATION_H_
