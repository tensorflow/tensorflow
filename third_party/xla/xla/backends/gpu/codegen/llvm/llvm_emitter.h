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

#ifndef XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_to_ir_bindings.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/ir_builder_mixin.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape_util.h"

namespace xla::gpu {

// Emit a constant with a given number of element, given byte size of the
// element, given symbol name and content.
GpuExecutable::ConstantInfo AppendGlobalConstant(llvm::Module* module,
                                                 int64_t num_elements,
                                                 int64_t bytes_per_element,
                                                 absl::string_view symbol_name,
                                                 int allocation_idx,
                                                 DenseDataIntermediate content);

absl::StatusOr<ThunkSequence> EmitBitonicSortLLVMIR(
    const HloSortInstruction* sort, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context);

absl::StatusOr<ThunkSequence> EmitPadToStaticLLVMIR(
    const HloCustomCallInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context);

absl::StatusOr<ThunkSequence> EmitSliceToDynamicLLVMIR(
    const HloCustomCallInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context);

absl::StatusOr<ThunkSequence> EmitRngGetAndUpdateStateLLVMIR(
    const HloRngGetAndUpdateStateInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_
