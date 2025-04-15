/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_METRICS_H_
#define XLA_SERVICE_GPU_METRICS_H_

#include <cstdint>

#include "absl/strings/string_view.h"

namespace xla {

// HLO passes (HLO -> HLO).
void RecordHloPassesDuration(uint64_t time_usecs);

// Compiling HLO to LLVM.
void RecordHloToLlvmDuration(uint64_t time_usecs);

// The entire LLVM to PTX pipeline, including both LLVM optimization (LLVM ->
// LLVM) and compiling LLVM -> PTX.
void RecordLlvmPassesAndLlvmToPtxDuration(uint64_t time_usecs);

// LLVM passes (linking and optimization) only (LLVM -> LLVM).
void RecordLlvmPassesDuration(uint64_t time_usecs);

// Compiling LLVM to PTX only.
void RecordLlvmToPtxDuration(uint64_t time_usecs);

// Compiling PTX to cubin.
void RecordPtxToCubinDuration(uint64_t time_usecs);

// Counts compiled programs count.
void IncrementCompiledProgramsCount();

// DO NOT USE---this is exposed only for testing.
// Resets compiled programs count.
void ResetCompiledProgramsCountForTesting();

// Gets compiled programs count.
int64_t GetCompiledProgramsCount();

// Records the size of the XLA device binary in bytes.
void RecordXlaDeviceBinarySize(int64_t size);

// Records the stacktrace of the GPU compiler.
void RecordGpuCompilerStacktrace();

// Returns the number of times the GPU compiler was called with the given
// stacktrace.
int GetGpuCompilerStacktraceCount(absl::string_view stacktrace);

}  // namespace xla

#endif  // XLA_SERVICE_GPU_METRICS_H_
