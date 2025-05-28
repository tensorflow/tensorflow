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

#ifndef XLA_BACKENDS_CPU_CODEGEN_COMPUTATION_KERNEL_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_COMPUTATION_KERNEL_EMITTER_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Emits a kernel definition for a call instruction, including all nested
// computations.
//
// This is useful where the alternative of using the thunk runtime would
// introduce unreasonable overheads, e.g. for tight while loops with scalar
// operations.
//
// This class leverages the legacy IrEmitter to emit the kernel definition,
// producing a synthetic buffer_table for all arguments and results (including
// intermediate instructions), though this may change in the future to use stack
// allocations for small buffers.
class ComputationKernelEmitter final : public KernelEmitter {
 public:
  ComputationKernelEmitter(const HloInstruction* instr,
                           const BufferAssignment* buffer_assignment,
                           const TargetMachineFeatures* target_machine);

  absl::StatusOr<KernelDefinition> EmitKernelDefinition() final;

 private:
  absl::StatusOr<llvm::Function*> EmitNestedComputation(
      llvm::Function* function, llvm::BasicBlock* return_block,
      llvm::IRBuilderBase& builder, llvm::Module& module,
      absl::flat_hash_map<BufferAllocation::Slice, int64_t> buffer_table_index)
      const;

 private:
  const HloInstruction* instr_;

  const BufferAssignment* buffer_assignment_;
  const TargetMachineFeatures* target_machine_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_COMPUTATION_KERNEL_EMITTER_H_
