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

#include "xla/backends/cpu/codegen/computation_kernel_emitter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/codegen/llvm_kernel_definition.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

namespace {

// Get the slices for the given instruction, expanding the elements if a tuple.
absl::Status GetInstructionSlices(
    const HloInstruction* instruction,
    const BufferAssignment* buffer_assignment,
    absl::flat_hash_set<KernelApiIrBuilder::KernelParameter>& parameters) {
  const Shape& shape = instruction->shape();
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      buffer_assignment->GetUniqueTopLevelSlice(instruction));
  if (slice.allocation()->is_thread_local()) {
    return absl::OkStatus();
  }
  parameters.insert({shape, std::move(slice)});

  if (shape.IsTuple()) {
    for (const auto& [leaf_index, leaf_shape] :
         ShapeUtil::GetLeafShapes(shape)) {
      TF_ASSIGN_OR_RETURN(
          BufferAllocation::Slice leaf_slice,
          buffer_assignment->GetUniqueSlice(instruction, {leaf_index}));
      parameters.insert({leaf_shape, std::move(leaf_slice)});
    }
  }
  return absl::OkStatus();
}

absl::Status GetAllSlices(
    const HloComputation* computation,
    const BufferAssignment* buffer_assignment,
    absl::flat_hash_set<KernelApiIrBuilder::KernelParameter>& arguments,
    absl::flat_hash_set<KernelApiIrBuilder::KernelParameter>& results) {
  for (const HloInstruction* instruction : computation->instructions()) {
    for (const HloInstruction* operand : instruction->operands()) {
      TF_RETURN_IF_ERROR(
          GetInstructionSlices(operand, buffer_assignment, arguments));
    }

    // Parameters just forward the results
    // TODO(willfroom): Is there a method somewhere to check if an instruction
    // just forwards the buffer? (e.g get-tuple-arg)
    if (instruction->opcode() != HloOpcode::kParameter) {
      TF_RETURN_IF_ERROR(
          GetInstructionSlices(instruction, buffer_assignment, results));
    }

    for (const HloComputation* nested_computation :
         instruction->called_computations()) {
      if (nested_computation->IsFusionComputation()) {
        continue;
      }

      TF_RETURN_IF_ERROR(GetAllSlices(nested_computation, buffer_assignment,
                                      arguments, results));
    }
  }

  return absl::OkStatus();
}

}  // namespace

ComputationKernelEmitter::ComputationKernelEmitter(
    const HloInstruction* instr, const BufferAssignment* buffer_assignment,
    const TargetMachineFeatures* target_machine)
    : instr_(instr),
      buffer_assignment_(buffer_assignment),
      target_machine_(target_machine) {}

absl::StatusOr<LlvmKernelDefinition>
ComputationKernelEmitter::EmitKernelDefinition() {
  VLOG(2) << "Emit Computation host kernel: " << instr_->name();

  auto ctx = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return Internal("HloModule is null");
  }

  absl::flat_hash_set<KernelApiIrBuilder::KernelParameter> arguments;
  absl::flat_hash_set<KernelApiIrBuilder::KernelParameter> results;
  TF_RETURN_IF_ERROR(
      GetAllSlices(instr_->to_apply(), buffer_assignment_, arguments, results));

  // As the computation is a series of operations, buffers are not disjoint.
  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()),
      KernelApiIrBuilder::BufferValidation::kNone);

  std::unique_ptr<llvm::Module> llvm_module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(instr_->name(), "_computation_kernel_module"), *ctx);

  TF_ASSIGN_OR_RETURN(std::string kernel_name,
                      kernel_api_ir_builder.GetKernelName(instr_, "_kernel"));

  TF_ASSIGN_OR_RETURN(KernelApiIrBuilder::KernelPrototype kernel_prototype,
                      kernel_api_ir_builder.EmitKernelPrototype(
                          *llvm_module, kernel_name,
                          std::vector<KernelApiIrBuilder::KernelParameter>(
                              arguments.begin(), arguments.end()),
                          std::vector<KernelApiIrBuilder::KernelParameter>(
                              results.begin(), results.end())));

  llvm::IRBuilder<> ir_builder(*ctx);
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm::Value* alloca_size = ir_builder.getInt64(
      kernel_prototype.arguments.size() + kernel_prototype.results.size());
  llvm::Value* buffer_table = ir_builder.CreateAlloca(
      ir_builder.getPtrTy(), alloca_size, "buffer_table");

  absl::flat_hash_map<BufferAllocation::Slice, int64_t>
      slice_to_buffer_table_index;

  int64_t buffer_table_index = 0;
  for (const auto& [array, slice] : llvm::zip(
           kernel_prototype.arguments, kernel_prototype.argument_buffers)) {
    int64_t index = buffer_table_index++;
    slice_to_buffer_table_index[slice] = index;
    llvm::Value* buffer_table_ptr = llvm_ir::EmitBufferIndexingGEP(
        buffer_table, ir_builder.getPtrTy(), index, &ir_builder);
    ir_builder.CreateStore(array.GetBasePointer(), buffer_table_ptr);
  }
  for (const auto& [array, slice] :
       llvm::zip(kernel_prototype.results, kernel_prototype.result_buffers)) {
    int64_t index = buffer_table_index++;
    slice_to_buffer_table_index[slice] = index;
    llvm::Value* buffer_table_ptr = llvm_ir::EmitBufferIndexingGEP(
        buffer_table, ir_builder.getPtrTy(), index, &ir_builder);
    ir_builder.CreateStore(array.GetBasePointer(), buffer_table_ptr);
  }

  TF_ASSIGN_OR_RETURN(
      llvm::Function * computation_function,
      EmitNestedComputation(
          kernel_prototype.function, kernel_prototype.return_block, ir_builder,
          *llvm_module, std::move(slice_to_buffer_table_index)));

  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm::Value* llvm_nullptr =
      llvm::Constant::getNullValue(ir_builder.getPtrTy());
  std::vector<llvm::Value*> args = {llvm_nullptr, llvm_nullptr, llvm_nullptr,
                                    buffer_table, llvm_nullptr, llvm_nullptr};
  ir_builder.CreateCall(computation_function, args);

  LlvmIrKernelSource source(std::move(ctx), std::move(llvm_module));

  KernelSpec spec(kernel_prototype.function->getName(), NumWorkGroups(),
                  std::move(kernel_prototype.argument_buffers),
                  std::move(kernel_prototype.result_buffers),
                  std::move(kernel_prototype.invariant_arguments));

  return LlvmKernelDefinition(std::move(spec), std::move(source));
}

absl::StatusOr<llvm::Function*> ComputationKernelEmitter::EmitNestedComputation(
    llvm::Function* function, llvm::BasicBlock* return_block,
    llvm::IRBuilderBase& builder, llvm::Module& llvm_module,
    absl::flat_hash_map<BufferAllocation::Slice, int64_t> buffer_table_index)
    const {
  const HloModule* hlo_module = instr_->GetModule();

  IrEmitter ir_emitter(
      nullptr, *hlo_module, *buffer_assignment_, &llvm_module,
      /*instruction_to_profile_idx=*/{},
      /*computation_to_profile_idx=*/{},
      ComputationsTransitivelyContainCustomCall(instr_), target_machine_,
      /*emit_code_for_msan=*/false, std::move(buffer_table_index),
      /*allow_runtime_calls=*/false);
  IrEmitter::IRBuilderGuard builder_guard = ir_emitter.WithBuilder(builder);

  TF_RETURN_IF_ERROR(ir_emitter.EmitSmallConstantGlobals());

  const HloComputation* computation = instr_->to_apply();
  return ir_emitter.EmitNestedComputation(*computation, computation->name(),
                                          /*is_reducer=*/false);
}

}  // namespace xla::cpu
