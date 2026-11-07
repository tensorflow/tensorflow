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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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

namespace {

// Dot operands and the optional addend, as kernel parameter indices.
struct DotOperands {
  const HloInstruction* dot;
  int64_t lhs;
  int64_t rhs;
  std::optional<int64_t> addend;
};

absl::StatusOr<DotOperands> GetDotOperands(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kDot) {
    return DotOperands{instr, 0, 1, std::nullopt};
  }

  if (instr->opcode() != HloOpcode::kFusion ||
      instr->fusion_kind() != HloInstruction::FusionKind::kOutput) {
    return Internal("Expected a dot or an output fusion: %s",
                    instr->ToString());
  }
  for (const HloInstruction* operand : instr->operands()) {
    if (!operand->shape().IsArray()) {
      return Internal("Dot fusion parameters must be arrays: %s",
                      instr->ToString());
    }
  }
  const HloInstruction* add = instr->fused_expression_root();
  if (add->opcode() != HloOpcode::kAdd) {
    return Internal("Dot fusion supports only `add` root instruction");
  }
  bool is_dot_operand0 = add->operand(0)->opcode() == HloOpcode::kDot;
  bool is_dot_operand1 = add->operand(1)->opcode() == HloOpcode::kDot;
  if (is_dot_operand0 == is_dot_operand1) {
    return Internal("Dot fusion root instruction must have single dot operand");
  }
  const HloInstruction* dot = add->operand(is_dot_operand0 ? 0 : 1);
  const HloInstruction* addend = add->operand(is_dot_operand0 ? 1 : 0);
  for (const HloInstruction* operand :
       {dot->operand(0), dot->operand(1), addend}) {
    if (operand->opcode() != HloOpcode::kParameter) {
      return Internal("Dot fusion operands must be fusion parameters: %s",
                      instr->ToString());
    }
  }
  return DotOperands{dot, dot->operand(0)->parameter_number(),
                     dot->operand(1)->parameter_number(),
                     addend->parameter_number()};
}

}  // namespace

absl::StatusOr<DotKernelEmitter::KernelDefinition>
DotKernelEmitter::EmitKernelDefinition() {
  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return Internal("HloModule is null");
  }
  const HloModuleConfig& config = hlo_module->config();

  ABSL_ASSIGN_OR_RETURN(DotOperands operands, GetDotOperands(instr_));
  const DotDimensionNumbers& dnums = operands.dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1) {
    return Internal("Unsupported number of contracting dimensions: %s",
                    operands.dot->ToString());
  }
  int64_t num_batch_dims = dnums.lhs_batch_dimensions_size();
  if (dnums.rhs_batch_dimensions_size() != num_batch_dims) {
    return Internal("Mismatched batch dimension counts: %s",
                    operands.dot->ToString());
  }
  for (int64_t i = 0; i < num_batch_dims; ++i) {
    if (dnums.lhs_batch_dimensions(i) != i ||
        dnums.rhs_batch_dimensions(i) != i) {
      return Internal("Batch dimensions must be leading: %s",
                      operands.dot->ToString());
    }
  }
  if (operands.addend.has_value() && num_batch_dims > 0) {
    return Internal("Dot output fusion requires non batch matrix vector dot");
  }

  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      config, *operands.dot, *target_machine_, /*allow_runtime_calls=*/false);

  if (!IsDotCodegenStrategy(strategy)) {
    return Internal("Unsupported dot implementation strategy");
  }
  if (strategy == DotImplementationStrategy::kTiledLlvmIrGemv &&
      (operands.dot->operand(0)->shape().dimensions().size() - num_batch_dims >
           2 ||
       operands.dot->operand(1)->shape().dimensions().size() - num_batch_dims >
           2)) {
    return Internal(
        "Matrix vector dot inner operands must have rank at most 2");
  }
  if (operands.addend.has_value() &&
      (strategy != DotImplementationStrategy::kTiledLlvmIrGemv ||
       ShapeUtil::IsScalar(operands.dot->operand(0)->shape()))) {
    return Internal("Dot output fusion requires non batch matrix vector dot");
  }

  auto ctx = std::make_unique<llvm::LLVMContext>();

  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx, KernelApiIrBuilder::Options::FromHloModuleConfig(config));

  std::unique_ptr<llvm::Module> llvm_module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(instr_->name(), "_elemental_kernel_module"), *ctx,
      target_machine_);

  ABSL_ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype kernel_prototype,
      kernel_api_ir_builder.EmitKernelPrototype(
          *llvm_module, instr_, buffer_assignment_, name(), "_kernel"));

  llvm::IRBuilder<> builder(*ctx);
  builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  const llvm_ir::IrArray& lhs_array = kernel_prototype.arguments[operands.lhs];
  const llvm_ir::IrArray& rhs_array = kernel_prototype.arguments[operands.rhs];
  const llvm_ir::IrArray* addend_array =
      operands.addend.has_value()
          ? &kernel_prototype.arguments[*operands.addend]
          : nullptr;
  const llvm_ir::IrArray& target_array = kernel_prototype.results[0];

  // The addend is accumulated by a single task.
  ABSL_ASSIGN_OR_RETURN(
      DotOpWorkGroupDim num_workgroups,
      EmitDotOperation(
          *operands.dot, target_array, lhs_array, rhs_array, addend_array,
          {kernel_prototype.workgroup_id.x, kernel_prototype.workgroup_id.y},
          &builder, config, *target_machine_,
          /*allow_parallelism=*/addend_array == nullptr));

  LlvmKernelSource source(std::move(ctx), std::move(llvm_module));

  KernelSpec::Buffers arguments, results;
  for (const auto& buffer : kernel_prototype.argument_buffers) {
    arguments.push_back({buffer.slice, buffer.shape});
  }
  for (const auto& buffer : kernel_prototype.result_buffers) {
    results.push_back({buffer.slice, buffer.shape});
  }

  KernelSpec spec(kernel_prototype.function->getName(),
                  NumWorkGroups{num_workgroups.x, num_workgroups.y},
                  std::move(arguments), std::move(results),
                  std::move(kernel_prototype.invariant_arguments));

  return KernelDefinition(std::move(spec), std::move(source));
}

}  // namespace xla::cpu
