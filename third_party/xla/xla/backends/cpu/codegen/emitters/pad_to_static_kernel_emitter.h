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

#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_PAD_TO_STATIC_KERNEL_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_PAD_TO_STATIC_KERNEL_EMITTER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Emits an MLIR kernel for a PadToStatic custom call on CPU.
class PadToStaticKernelEmitter final : public KernelEmitter<MlirKernelSource> {
 public:
  PadToStaticKernelEmitter(mlir::MLIRContext& mlir_context,
                           const HloInstruction* instr,
                           const BufferAssignment* buffer_assignment,
                           bool use_unique_c_name = false);

  absl::string_view name() const final {
    return "pad_to_static_kernel_emitter";
  }

  absl::StatusOr<KernelDefinition> EmitKernelDefinition() final;

 private:
  mlir::MLIRContext& mlir_context_;
  const HloInstruction* instr_;
  const BufferAssignment* buffer_assignment_;
  bool use_unique_c_name_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_PAD_TO_STATIC_KERNEL_EMITTER_H_
