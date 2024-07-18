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

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_driver.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"

namespace stream_executor {
// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array. The generated cubin matches the compute
// capabilities of the device associated with 'device_ordinal'.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
absl::StatusOr<std::vector<uint8_t>> CompileGpuAsm(int device_ordinal,
                                                   const char* ptx_contents,
                                                   GpuAsmOpts options);

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array. The generated cubin matches the compute
// capabilities provided by 'cc_major' and 'cc_minor'.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
absl::StatusOr<std::vector<uint8_t>> CompileGpuAsm(
    int cc_major, int cc_minor, const char* ptx_contents, GpuAsmOpts options,
    bool cancel_if_reg_spill = false);

absl::StatusOr<std::vector<uint8_t>> CompileGpuAsmUsingPtxAs(
    int cc_major, int cc_minor, const char* ptx_contents, GpuAsmOpts options,
    bool cancel_if_reg_spill = false);

// Same as CompileGpuAsm, but caches the result, and returns unowned view of
// the compiled binary.
//
// A copy of the string provided in ptx will be made.
absl::StatusOr<absl::Span<const uint8_t>> CompileGpuAsmOrGetCached(
    int device_ordinal, const char* ptx, GpuAsmOpts compilation_options);

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
    gpu::GpuContext* context, std::vector<CubinOrPTXImage> images);

absl::StatusOr<std::vector<uint8_t>> LinkUsingNvlink(
    std::string_view preferred_cuda_dir, gpu::GpuContext* context,
    std::vector<CubinOrPTXImage> images);

using ToolVersion = std::array<int64_t, 3>;
absl::StatusOr<std::string> FindCudaExecutable(
    std::string_view binary_name, std::string_view preferred_cuda_dir,
    ToolVersion minimum_version,
    absl::Span<const ToolVersion> excluded_versions);

absl::StatusOr<std::string> FindCudaExecutable(
    std::string_view binary_name, std::string_view preferred_cuda_dir);

// Runs tool --version and parses its version string.
absl::StatusOr<ToolVersion> GetToolVersion(std::string_view tool_path);

// On NVIDIA GPUs, returns the version of the ptxas command line tool.
absl::StatusOr<ToolVersion> GetAsmCompilerVersion(
    std::string_view preferred_cuda_dir);

// On NVIDIA GPUs, returns the version of the nvlink command line tool.
absl::StatusOr<ToolVersion> GetNvLinkVersion(
    std::string_view preferred_cuda_dir);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_ASM_COMPILER_H_
