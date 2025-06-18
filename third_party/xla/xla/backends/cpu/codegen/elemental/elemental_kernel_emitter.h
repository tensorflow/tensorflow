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

#ifndef XLA_BACKENDS_CPU_CODEGEN_ELEMENTAL_ELEMENTAL_KERNEL_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_ELEMENTAL_ELEMENTAL_KERNEL_EMITTER_H_

#include <string>

#include "absl/status/statusor.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_kernel_definition.h"
#include "xla/codegen/llvm_kernel_emitter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/llvm_ir/loop_emitter.h"

namespace xla::cpu {

class ElementalKernelEmitter final : public LlvmKernelEmitter {
 public:
  ElementalKernelEmitter(const HloInstruction* instr,
                         const BufferAssignment* buffer_assignment,
                         const TargetMachineFeatures* target_machine);

  absl::StatusOr<LlvmKernelDefinition> EmitKernelDefinition() override;

  std::string name() const final { return "elemental_kernel_emitter"; }

 private:
  // Emits LLVM IR using elemental loop emitter and the given element generator.
  // If the instruction is parallelized, it will emit a parallel loop partition
  // and return the requested number of execution threads.
  absl::StatusOr<NumWorkGroups> EmitElementalLoops(
      llvm::IRBuilderBase& b, const HloInstruction* instr,
      const KernelApiIrBuilder::KernelPrototype& kernel_prototype,
      const llvm_ir::ElementGenerator& element_generator);

  // Create a thread local call callback, can be empty if no IrEmitter is
  // registered.
  absl::StatusOr<CpuElementalIrEmitter::ThreadLocalCallCallback>
  ThreadLocalCallbackFactory(llvm::IRBuilderBase& builder,
                             llvm::Module& module) const;

 private:
  const HloInstruction* instr_;

  const BufferAssignment* buffer_assignment_ = nullptr;
  const TargetMachineFeatures* target_machine_ = nullptr;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_ELEMENTAL_ELEMENTAL_KERNEL_EMITTER_H_
