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
#include <string_view>
#include <vector>

#include "absl/base/macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/semantic_version.h"

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

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array. The generated cubin matches the compute
// capabilities provided by 'cc'.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
absl::StatusOr<std::vector<uint8_t>> CompileGpuAsmUsingPtxAs(
    const CudaComputeCapability& cc, const std::string& ptx_contents,
    GpuAsmOpts options, bool cancel_if_reg_spill = false);

// Same as CompileGpuAsm, but caches the result, and returns unowned view of
// the compiled binary.
//
// A copy of the string provided in ptx will be made.
absl::StatusOr<absl::Span<const uint8_t>> CompileGpuAsmOrGetCached(
    const CudaComputeCapability& cc, const std::string& ptx_contents,
    GpuAsmOpts compilation_options);

struct CubinOrPTXImage {
  std::string profile;
  std::vector<uint8_t> bytes;
};

// Bundles the GPU machine code (cubins) and PTX if requested and returns the
// resulting binary (i.e. a fatbin) as a byte array.
absl::StatusOr<std::vector<uint8_t>> BundleGpuAsm(
    std::vector<CubinOrPTXImage> images, GpuAsmOpts options);

// Links multiple relocatable GPU images (e.g. results of ptxas -c) into a
// single image.
absl::StatusOr<std::vector<uint8_t>> LinkGpuAsm(
    stream_executor::CudaComputeCapability cc,
    std::vector<CubinOrPTXImage> images);

absl::StatusOr<std::vector<uint8_t>> LinkUsingNvlink(
    stream_executor::CudaComputeCapability cc,
    std::string_view preferred_cuda_dir, std::vector<CubinOrPTXImage> images);

absl::StatusOr<std::string> FindCudaExecutable(
    std::string_view binary_name, std::string_view preferred_cuda_dir,
    SemanticVersion minimum_version,
    absl::Span<const SemanticVersion> excluded_versions);

absl::StatusOr<std::string> FindCudaExecutable(
    std::string_view binary_name, std::string_view preferred_cuda_dir);

// Runs tool --version and parses its version string.
absl::StatusOr<SemanticVersion> GetToolVersion(std::string_view tool_path);

// On NVIDIA GPUs, returns the version of the ptxas command line tool.
absl::StatusOr<SemanticVersion> GetAsmCompilerVersion(
    std::string_view preferred_cuda_dir);

// On NVIDIA GPUs, returns the version of the nvlink command line tool.
absl::StatusOr<SemanticVersion> GetNvLinkVersion(
    std::string_view preferred_cuda_dir);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_ASM_COMPILER_H_
