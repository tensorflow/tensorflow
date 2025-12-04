/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_LLVM_IR_EMITTER_NESTED_H_
#define XLA_BACKENDS_GPU_CODEGEN_LLVM_IR_EMITTER_NESTED_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla {
namespace gpu {

// Emits LLVM IR for a "nested computation" into a non-kernel device function.
//
// This is used to emit code for HloComputations that don't require a separate
// kernel call.  For example, IrEmitterNested is used to emit code for a kReduce
// HLO's elementwise reduction computation.  Notably, IrEmitterNested is *not*
// used to emit code for fusion nodes -- fusion nodes use FusedIrEmitter, which
// is a different beast altogether.
//
// IrEmitterNested generates a non-kernel function with the following
// parameters:
//
//   - N pointers to the buffers of each of the N parameters to the computation,
//   - a pointer to the output buffer of the computation, and
//   - a pointer to the top-level temp buffer.
absl::Status CallNestedComputation(llvm::IRBuilderBase* builder,
                                   IrEmitterContext& ir_emitter_context,
                                   llvm::Module* llvm_module,
                                   const HloComputation& computation,
                                   absl::Span<llvm::Value* const> operands,
                                   llvm::Value* output);

// Emit a constant with a given number of element, given byte size of the
// element, given symbol name and content.
GpuExecutable::ConstantInfo AppendGlobalConstant(llvm::Module* module,
                                                 int64_t num_elements,
                                                 int64_t bytes_per_element,
                                                 absl::string_view symbol_name,
                                                 int allocation_idx,
                                                 DenseDataIntermediate content);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_LLVM_IR_EMITTER_NESTED_H_
