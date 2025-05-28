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

#ifndef XLA_BACKENDS_CPU_CODEGEN_DOT_DOT_KERNEL_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_DOT_DOT_KERNEL_EMITTER_H_

#include "absl/status/statusor.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/llvm_kernel_definition.h"
#include "xla/codegen/llvm_kernel_emitter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

class DotKernelEmitter final : public LlvmKernelEmitter {
 public:
  DotKernelEmitter(const HloInstruction* instr,
                   const BufferAssignment* buffer_assignment,
                   const TargetMachineFeatures* target_machine);

  absl::StatusOr<LlvmKernelDefinition> EmitKernelDefinition() override;

 private:
  const HloInstruction* instr_;

  const BufferAssignment* buffer_assignment_ = nullptr;
  const TargetMachineFeatures* target_machine_ = nullptr;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_DOT_DOT_KERNEL_EMITTER_H_
