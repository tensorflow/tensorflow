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

#include "xla/backends/cpu/codegen/slice_to_dynamic_kernel_emitter.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

namespace xla::cpu {

SliceToDynamicKernelEmitter::SliceToDynamicKernelEmitter(
    const HloInstruction* instr, const BufferAssignment* buffer_assignment,
    const TargetMachineFeatures* target_machine)
    : instr_(instr),
      buffer_assignment_(buffer_assignment),
      target_machine_(target_machine) {}

absl::StatusOr<SliceToDynamicKernelEmitter::KernelDefinition>
SliceToDynamicKernelEmitter::EmitKernelDefinition() {
  TF_RET_CHECK(instr_ != nullptr);
  TF_RET_CHECK(instr_->opcode() == HloOpcode::kCustomCall &&
               instr_->custom_call_target() == "SliceToDynamic")
      << "Expected SliceToDynamic custom-call: " << instr_->ToString();
  TF_RET_CHECK(instr_->shape().IsArray())
      << "Expected array output shape: " << instr_->ToString();
  TF_RET_CHECK(instr_->operand_count() ==
               1 + instr_->shape().dimensions().size())
      << "Expected one dynamic size operand per dimension: "
      << instr_->ToString();
  TF_RET_CHECK(target_machine_ != nullptr);

  const HloModule* hlo_module = instr_->GetModule();
  TF_RET_CHECK(hlo_module != nullptr);
  const HloModuleConfig& config = hlo_module->config();

  auto ctx = std::make_unique<llvm::LLVMContext>();
  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx, KernelApiIrBuilder::Options::FromHloModuleConfig(config));
  std::unique_ptr<llvm::Module> llvm_module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(instr_->name(), "_slice_to_dynamic_kernel_module"), *ctx,
      target_machine_);

  ABSL_ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype kernel_prototype,
      kernel_api_ir_builder.EmitKernelPrototype(
          *llvm_module, instr_, buffer_assignment_, name(), "_kernel"));

  llvm::IRBuilder<> ir_builder(*ctx);
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  std::vector<llvm::Value*> dynamic_dims;
  dynamic_dims.reserve(instr_->operand_count() - 1);
  int64_t raw_data_size =
      ShapeUtil::ByteSizeOf(ShapeUtil::MakeStaticShape(instr_->shape()));
  llvm::Value* dest_buffer = output_array.GetBasePointer();
  for (int64_t i = 1; i < instr_->operand_count(); ++i) {
    const int64_t dim_index = i - 1;
    llvm::Value* source_buffer = kernel_prototype.arguments[i].GetBasePointer();
    llvm::LoadInst* dyn_dim_size = ir_builder.CreateLoad(
        llvm_ir::ShapeToIrType(instr_->operand(i)->shape(), *ctx),
        source_buffer, "dyn_dim_size");

    llvm::Value* metadata = ir_builder.CreateConstInBoundsGEP1_64(
        ir_builder.getInt8Ty(), dest_buffer,
        raw_data_size + dim_index * static_cast<int64_t>(sizeof(int32_t)));
    ir_builder.CreateStore(dyn_dim_size, metadata);
    dynamic_dims.push_back(
        ir_builder.CreateIntCast(dyn_dim_size, ir_builder.getInt64Ty(),
                                 /*isSigned=*/true, "i64_dyn_dim_size"));
  }

  auto loop_body_emitter =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* source_element =
        kernel_prototype.arguments[0].EmitReadArrayElement(array_index,
                                                           &ir_builder);
    llvm::Value* linear_index =
        array_index.Linearize(dynamic_dims, &ir_builder);
    llvm_ir::IrArray::Index dest_index(linear_index, output_array.GetShape(),
                                       &ir_builder);
    output_array.EmitWriteArrayElement(dest_index, source_element, &ir_builder);
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(llvm_ir::LoopEmitter(loop_body_emitter,
                                            output_array.GetShape(),
                                            dynamic_dims, &ir_builder)
                           .EmitLoop(llvm_ir::IrName(instr_)));

  LlvmKernelSource source(std::move(ctx), std::move(llvm_module));

  KernelSpec::Buffers arguments, results;
  for (const KernelApiIrBuilder::KernelParameter& buffer :
       kernel_prototype.argument_buffers) {
    arguments.push_back({buffer.slice, buffer.shape});
  }
  for (const KernelApiIrBuilder::KernelParameter& buffer :
       kernel_prototype.result_buffers) {
    results.push_back({buffer.slice, buffer.shape});
  }

  KernelSpec spec(kernel_prototype.function->getName(), NumWorkGroups{},
                  std::move(arguments), std::move(results),
                  std::move(kernel_prototype.invariant_arguments));
  return KernelDefinition(std::move(spec), std::move(source));
}

}  // namespace xla::cpu
