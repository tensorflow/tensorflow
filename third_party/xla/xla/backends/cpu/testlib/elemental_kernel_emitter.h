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

#ifndef XLA_BACKENDS_CPU_TESTLIB_ELEMENTAL_KERNEL_EMITTER_H_
#define XLA_BACKENDS_CPU_TESTLIB_ELEMENTAL_KERNEL_EMITTER_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/elemental_ir_emitter.h"

namespace xla::cpu {

class ElementalKernelEmitter final : public KernelEmitter {
 public:
  using ElementalIrEmitterFactory =
      absl::AnyInvocable<std::unique_ptr<ElementalIrEmitter>(
          llvm::Module*, llvm::IRBuilderBase*)>;

  explicit ElementalKernelEmitter(std::unique_ptr<HloInstruction> op_hlo);

  absl::StatusOr<std::unique_ptr<KernelSpec>> EmitKernelSpec() override;

 private:
  std::unique_ptr<HloInstruction> op_hlo_;

  llvm::orc::ThreadSafeContext context_;

  KernelApiIrBuilder kernel_api_ir_builder_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TESTLIB_ELEMENTAL_KERNEL_EMITTER_H_
