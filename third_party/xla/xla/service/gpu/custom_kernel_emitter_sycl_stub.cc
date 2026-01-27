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

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/custom_kernel_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::unique_ptr<Thunk>> EmitPtxCustomKernelThunk(
    const HloCustomCallInstruction* /*instr*/, IrEmitterContext* /*context*/) {
  return absl::UnimplementedError(
      "Custom kernel emitter for PTX custom call is not yet implemented in "
      "SYCL platform.");
}

}  // namespace gpu
}  // namespace xla
