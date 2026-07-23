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

#include "xla/backends/cpu/codegen/elemental/concatenate_kernel_emitter.h"

#include "absl/status/statusor.h"
#include "xla/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

ConcatenateKernelEmitter::ConcatenateKernelEmitter(
    const HloInstruction* instr, const BufferAssignment* buffer_assignment,
    const TargetMachineFeatures* target_machine)
    : instr_(instr),
      buffer_assignment_(buffer_assignment),
      target_machine_(target_machine) {}

absl::StatusOr<ConcatenateKernelEmitter::KernelDefinition>
ConcatenateKernelEmitter::EmitKernelDefinition() {
  return ElementalKernelEmitter(instr_, buffer_assignment_, target_machine_)
      .EmitKernelDefinition();
}

}  // namespace xla::cpu
