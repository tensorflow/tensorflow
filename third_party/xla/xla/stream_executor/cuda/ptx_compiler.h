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
#ifndef XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_H_
#define XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {

// Takes PTX as a null-terminated string and compiles it to SASS (CUBIN)
// targeting the sm_<cc_major>.<cc_minor> NVIDIA GPU architecture.
absl::StatusOr<std::vector<uint8_t>> CompileGpuAsmUsingLibNvPtxCompiler(
    const CudaComputeCapability& cc, const std::string& ptx, GpuAsmOpts options,
    bool cancel_if_reg_spill);

absl::StatusOr<SemanticVersion> GetLibNvPtxCompilerVersion();

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_H_
