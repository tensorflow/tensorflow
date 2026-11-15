/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_RNG_GET_AND_UPDATE_STATE_H_
#define XLA_BACKENDS_GPU_CODEGEN_RNG_GET_AND_UPDATE_STATE_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla::gpu {

// Builds the MLIR module implementing an `RngGetAndUpdateState` kernel.
// Visible for testing.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
CreateRngGetAndUpdateStateMlirModule(
    mlir::MLIRContext& mlir_context,
    const HloRngGetAndUpdateStateInstruction* instr,
    const BufferAssignment* buffer_assignment,
    absl::string_view entry_function_name);

// Emits the kernel thunk for an `RngGetAndUpdateState` instruction on GPU.
AsyncThunkSequence EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr,
    IrEmitterContext* ir_emitter_context);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_RNG_GET_AND_UPDATE_STATE_H_
