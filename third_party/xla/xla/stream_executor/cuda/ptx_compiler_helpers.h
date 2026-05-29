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

#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/semantic_version.h"

// If defined, flags returned by this function will be passed to
// ptxas/libnvptxcompiler before those by `GpuAsmOpts`.
ABSL_ATTRIBUTE_WEAK extern "C" absl::Span<const absl::string_view>
XlaGpuPtxCompilerExtraFlagsToPrepend();

namespace stream_executor {

// Appends all necessary flags to pass to ptxas/libnvptxcompiler, including
// those from `XlaGpuPtxCompilerExtraFlagsToPrepend` if defined. The flags
// appended are a superset of those appended by `AppendPtxCompilerFlags`.
void AppendArchitectureSpecificPtxCompilerFlags(
    const CudaComputeCapability& cc, GpuAsmOpts options,
    bool dump_compilation_log, std::vector<std::string>& flags);

// Appends flags from `GpuAsmOptions` onto `flags`, including those from
// `XlaGpuPtxCompilerExtraFlagsToPrepend` if defined. The flags appended are a
// subset of those appended by `AppendArchitectureSpecificPtxCompilerFlags`.
void AppendPtxCompilerFlags(GpuAsmOpts options,
                            std::vector<std::string>& flags);

// Creates a status with a payload indicating a register allocation error.
absl::Status PtxRegisterAllocationError(absl::string_view message);

// Checks whether ptxas log contains errors related to register allocation.
bool IsPtxRegisterAllocationError(absl::string_view);

// Checks whether the status is a register allocation error.
bool IsPtxRegisterAllocationError(absl::Status status);

// Identifies errors in the ptxas log and creates an error status.
// `architecture` is the name of the GPU architecture, e.g. "sm_80" and is only
// used for error message generation. If `cancel_if_reg_spill` is true, then a
// register spill warning will be treated as an error, otherwise it will be
// ignored.
absl::Status CreateErrorFromPTXASLog(absl::string_view log,
                                     absl::string_view architecture,
                                     bool cancel_if_reg_spill);

// Warns if the ptxas version should be upgraded.
void WarnIfBadPtxasVersion(absl::string_view method,
                           const CudaComputeCapability& cc,
                           SemanticVersion compiler_version);

// Determines the latest supported PTX ISA from an "unsupported version" error
// log issued by ptxas.
//
// The output of ptxas in such a case is expected to look like:
//
// ptxas application ptx input, line 1; fatal   :
// Unsupported .version 99.99; current version is '8.8'
absl::StatusOr<int> GetLatestPtxIsaVersionFromUnsupportedVersionErrorLog(
    absl::string_view error_log);

// Extracts the module stats from the ptxas log.
//
// Example: "Registers are spilled to local memory in function 'rr', 1080 bytes
// spill stores, 968 bytes spill loads" will return:
// ModuleStats{ KernelStats{"rr", {1080, 968}} }
ModuleStats ExtractModuleStatsFromLog(absl::string_view log);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_HELPERS_H_
