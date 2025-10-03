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

#ifndef XLA_SERVICE_GPU_CUSTOM_KERNEL_EMITTER_H_
#define XLA_SERVICE_GPU_CUSTOM_KERNEL_EMITTER_H_

#include <memory>

#include "absl/status/statusor.h"

namespace xla {

// Forward declaration to avoid heavy includes.
class HloCustomCallInstruction;

namespace gpu {

class Thunk;
class IrEmitterContext;

// Emit a platform-specific custom kernel thunk for PTX custom calls.
// This function has separate implementations for CUDA and ROCm backends,
// selected at build time via conditional compilation in BUILD rules.
absl::StatusOr<std::unique_ptr<Thunk>> EmitPtxCustomKernelThunk(
    const HloCustomCallInstruction* instr, IrEmitterContext* context);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUSTOM_KERNEL_EMITTER_H_
