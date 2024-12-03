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

#ifndef XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_
#define XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_computation.h"
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
                                   const HloComputation& computation,
                                   absl::Span<llvm::Value* const> operands,
                                   llvm::Value* output);

// Like CallNestedComputation, but parameters and results are scalars.
absl::StatusOr<std::vector<llvm::Value*>> CallNestedComputationWithScalars(
    llvm::IRBuilderBase* builder, IrEmitterContext& ir_emitter_context,
    const HloComputation& computation,
    absl::Span<llvm::Value* const> parameter_elements);

// Like CallNestedComputationWithScalars, but parameters are scalar addresses.
absl::StatusOr<std::vector<llvm::Value*>> CallNestedComputationWithScalarAddrs(
    llvm::IRBuilderBase* builder, IrEmitterContext& ir_emitter_context,
    const HloComputation& computation,
    absl::Span<llvm::Value* const> parameter_elements_addrs);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_
