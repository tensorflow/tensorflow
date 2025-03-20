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

#include "xla/backends/cpu/codegen/dot/dot_kernel_emitter.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

static bool IsDotCodegenStrategy(DotImplementationStrategy strategy) {
  switch (strategy) {
    case DotImplementationStrategy::kNaiveLlvmIr:
    case DotImplementationStrategy::kTiledLlvmIrGemv:
    case DotImplementationStrategy::kTiledLlvmIrGemm:
      return true;
    default:
      return false;
  }
}

DotKernelEmitter::DotKernelEmitter(const HloInstruction* instr,
                                   const BufferAssignment* buffer_assignment,
                                   const TargetMachineFeatures* target_machine)
    : instr_(instr),
      buffer_assignment_(buffer_assignment),
      target_machine_(target_machine) {}

absl::StatusOr<KernelDefinition> DotKernelEmitter::EmitKernelDefinition() {
  const HloModuleConfig& config = instr_->GetModule()->config();

  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      config, *instr_, *target_machine_, /*allow_runtime_calls=*/false);

  if (!IsDotCodegenStrategy(strategy)) {
    return Internal("Unsupported dot implementation strategy");
  }

  auto ctx = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return Internal("HloModule is null");
  }

  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));

  std::unique_ptr<llvm::Module> llvm_module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(instr_->name(), "_elemental_kernel_module"), *ctx);

  TF_ASSIGN_OR_RETURN(KernelApiIrBuilder::KernelPrototype kernel_prototype,
                      kernel_api_ir_builder.EmitKernelPrototype(
                          *llvm_module, instr_, buffer_assignment_, "_kernel"));

  llvm::IRBuilder<> builder(*ctx);
  builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray lhs_array = kernel_prototype.arguments[0];
  llvm_ir::IrArray rhs_array = kernel_prototype.arguments[1];
  llvm_ir::IrArray target_array = kernel_prototype.results[0];

  TF_RETURN_IF_ERROR(EmitDotOperation(
      *instr_, target_array, lhs_array, rhs_array,
      /*addend_array=*/nullptr, /*executable_run_options_value=*/nullptr,
      &builder, config, *target_machine_,
      /*allow_runtime_calls=*/false));

  auto source = std::make_unique<LlvmIrKernelSource>(std::move(ctx),
                                                     std::move(llvm_module));

  KernelSpec spec(kernel_prototype.function->getName(), se::ThreadDim(),
                  std::move(kernel_prototype.argument_buffers),
                  std::move(kernel_prototype.result_buffers),
                  std::move(kernel_prototype.invariant_arguments));

  return KernelDefinition(std::move(spec), std::move(source));
}

}  // namespace xla::cpu
