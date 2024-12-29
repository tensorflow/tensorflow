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

#include "xla/service/cpu/ir_emitter2.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CodeGen.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/cpu/elemental_math_emitter.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/parallel_loop_emitter.h"
#include "xla/service/cpu/shape_partition.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

namespace {

KernelApiIrBuilder::Options KernelApiIrBuilderOptionsFromHloModuleConfig(
    const HloModuleConfig& config) {
  return KernelApiIrBuilder::Options{
      config.debug_options().xla_llvm_enable_invariant_load_metadata(),
      config.debug_options().xla_cpu_prefer_vector_width()};
}

}  // namespace

//===----------------------------------------------------------------------===//
// ElementalIrEmitter
//===----------------------------------------------------------------------===//

class IrEmitter2::ElementalIrEmitter : public CpuElementalIrEmitter {
 public:
  ElementalIrEmitter(llvm::Module* module, llvm::IRBuilderBase* b,
                     IrEmitter* nested_ir_emitter, bool fast_min_max)
      : CpuElementalIrEmitter(module, b, true, fast_min_max),
        nested_ir_emitter_(nested_ir_emitter),
        fast_min_max_(fast_min_max) {}

 protected:
  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) override {
    // Add a thread local call to the nested computation.
    VLOG(2) << "Emit thread local call to: " << callee.name();
    auto values = nested_ir_emitter_->EmitThreadLocalCall(
        callee, parameters, name, is_reducer, /*in_compute_function=*/false);

    return values;
  }

  bool fast_min_max() override { return fast_min_max_; }

 private:
  IrEmitter* nested_ir_emitter_;
  bool fast_min_max_;
};

//===----------------------------------------------------------------------===//
// IrEmitter2
//===----------------------------------------------------------------------===//

IrEmitter2::IrEmitter2(const HloModule& hlo_module, llvm::Module* module,
                       IrEmitter* nested_ir_emitter)
    : hlo_module_(hlo_module),
      module_(module),
      nested_ir_emitter_(nested_ir_emitter),
      kernel_api_ir_builder_(
          module_->getContext(),
          KernelApiIrBuilderOptionsFromHloModuleConfig(hlo_module_.config())) {}

bool IrEmitter2::fast_min_max() const {
  return hlo_module_.config().debug_options().xla_cpu_enable_fast_min_max();
}
IrEmitter2::KernelInfo::KernelInfo(KernelPrototype prototype,
                                   const se::BlockDim& block_dims,
                                   const se::ThreadDim& thread_dims)
    : name(prototype.function->getName().str()),
      block_dims(block_dims),
      thread_dims(thread_dims),
      invariant_arguments(std::move(prototype.invariant_arguments)) {}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitElementalHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit elemental host kernel: " << instr->name();

  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(instr));

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  IrEmitter::IRBuilderGuard builder_guard = nested_ir_emitter_->WithBuilder(b);

  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* operand = instr->operand(i);
    operand_to_generator[operand] = [&, i](const llvm_ir::IrArray::Index& idx) {
      return kernel_prototype.arguments[i].EmitReadArrayElement(idx, &b);
    };
  }

  if (instr->has_to_apply()) {
    HloComputation* nested_computation = instr->to_apply();
    bool is_reducer = instr->opcode() == HloOpcode::kReduce ||
                      instr->opcode() == HloOpcode::kReduceWindow;
    TF_RETURN_IF_ERROR(EmitNestedComputation(
        *nested_computation, llvm_ir::IrName(instr), is_reducer));
  }

  ElementalIrEmitter elemental_emitter(module_, &b, nested_ir_emitter_,
                                       fast_min_max());
  llvm_ir::ElementGenerator element_generator =
      elemental_emitter.MakeElementGenerator(instr, operand_to_generator);

  TF_ASSIGN_OR_RETURN(
      se::ThreadDim thread_dims,
      EmitElementalLoops(b, instr, kernel_prototype, element_generator));

  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), thread_dims));
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitPadHostKernel(
    const HloInstruction* pad) {
  VLOG(2) << "Emit Pad host kernel.";

  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(pad));

  llvm_ir::IrArray operand_array = kernel_prototype.arguments[0];
  llvm_ir::IrArray padvalue_array = kernel_prototype.arguments[1];
  llvm_ir::IrArray output_array = kernel_prototype.results[0];

  llvm::LLVMContext& ctx = module_->getContext();
  llvm::IRBuilder<> b(ctx);
  auto builder_overwrite = nested_ir_emitter_->WithBuilder(b);

  nested_ir_emitter_->PushComputeFunction(
      &b, module_,
      /*num_dynamic_loop_bounds=*/0, kernel_prototype.function,
      /*dynamic_loop_bounds_arg=*/nullptr, kernel_prototype.return_block);

  TF_RETURN_IF_ERROR(nested_ir_emitter_->HandlePad(
      const_cast<HloInstruction*>(pad), operand_array, padvalue_array,
      output_array));

  nested_ir_emitter_->PopComputeFunction();

  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitFusionHostKernel(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emit fusion host kernel: " << fusion->name();

  // In XLA:CPU output fusion can only be a fusion into dot operation.
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kOutput) {
    return EmitDotFusionHostKernel(fusion);
  }

  if (fusion->fusion_kind() != HloInstruction::FusionKind::kLoop) {
    return Internal("Unsupported fusion kind for instruction: %s",
                    fusion->ToString());
  }

  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(fusion));

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  IrEmitter::IRBuilderGuard builder_guard = nested_ir_emitter_->WithBuilder(b);

  HloComputation* nested_computation = fusion->fused_instructions_computation();
  TF_RETURN_IF_ERROR(EmitNestedComputation(*nested_computation,
                                           llvm_ir::IrName(fusion), false));

  ElementalIrEmitter elemental_emitter(module_, &b, nested_ir_emitter_,
                                       fast_min_max());

  FusedIrEmitter fused_emitter(elemental_emitter);
  for (int i = 0; i < fusion->operand_count(); i++) {
    fused_emitter.BindGenerator(
        *fusion->fused_parameter(i), [&, i](llvm_ir::IrArray::Index idx) {
          return kernel_prototype.arguments[i].EmitReadArrayElement(idx, &b);
        });
  }

  // Check if the fusion can be emitted in-place and skip expensive loop for
  // all elements in the output array.
  if (llvm_ir::CanEmitFusedDynamicUpdateSliceInPlace(
          const_cast<HloFusionInstruction*>(fusion),
          nested_ir_emitter_->assignment())) {
    // Delegate to common implementation of fused in-place dynamic-update-slice.
    TF_RETURN_IF_ERROR(llvm_ir::EmitFusedDynamicUpdateSliceInPlace(
        const_cast<HloFusionInstruction*>(fusion), kernel_prototype.results[0],
        &fused_emitter, &b));

    return kernels_.emplace_back(KernelInfo(std::move(kernel_prototype),
                                            se::BlockDim(), se::ThreadDim()));
  }

  // Emit plain elemental loops for the fusion operation.
  TF_ASSIGN_OR_RETURN(
      auto element_generator,
      fused_emitter.GetGenerator(*fusion->fused_expression_root()));

  TF_ASSIGN_OR_RETURN(
      se::ThreadDim thread_dims,
      EmitElementalLoops(b, fusion, kernel_prototype, element_generator));

  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), thread_dims));
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitReductionHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit reduction host kernel: " << instr->name();

  // TODO(ezhulenev): Port vectorized reduction emitter from IrEmitter.
  return EmitElementalHostKernel(instr);
}

// Dot (fusion) host kernel only supports strategies that emit LLVM IR.
static bool IsDotCodegenStrategy(DotImplementationStrategy strategy) {
  static std::array<DotImplementationStrategy, 3> kDotCodegenStrategies = {
      DotImplementationStrategy::kNaiveLlvmIr,
      DotImplementationStrategy::kTiledLlvmIrGemm,
      DotImplementationStrategy::kTiledLlvmIrGemv,
  };

  return absl::c_find(kDotCodegenStrategies, strategy) !=
         kDotCodegenStrategies.end();
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitDotHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit dot host kernel: " << instr->name();

  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      hlo_module_.config(), *instr,
      nested_ir_emitter_->target_machine_features());

  if (!IsDotCodegenStrategy(strategy)) {
    return Internal("Unsupported dot implementation strategy");
  }

  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(instr));

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray lhs_array = kernel_prototype.arguments[0];
  llvm_ir::IrArray rhs_array = kernel_prototype.arguments[1];
  llvm_ir::IrArray target_array = kernel_prototype.results[0];

  TF_RETURN_IF_ERROR(EmitDotOperation(
      *instr, target_array, lhs_array, rhs_array,
      /*addend_array=*/nullptr, /*executable_run_options_value=*/nullptr, &b,
      hlo_module_.config(), nested_ir_emitter_->target_machine_features(),
      /*allow_runtime_calls=*/false));

  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitConcatenateHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit concatenate host kernel: " << instr->name();

  auto fast_impl_reason = CanDoFastConcatenate(instr);
  if (fast_impl_reason.ok()) {
    VLOG(1) << "Emitting fast concatenate for " << instr->ToString() << ": "
            << fast_impl_reason.message();
    TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                        EmitKernelPrototype(instr));
    llvm::IRBuilder<> ir_builder(module_->getContext());
    ir_builder.SetInsertPoint(
        kernel_prototype.function->getEntryBlock().getTerminator());

    llvm_ir::IrArray output_array = kernel_prototype.results[0];
    TF_RETURN_IF_ERROR(::xla::cpu::EmitFastConcatenate(
        instr, kernel_prototype.arguments, output_array, module_, ir_builder));
    return kernels_.emplace_back(KernelInfo(std::move(kernel_prototype),
                                            se::BlockDim(), se::ThreadDim()));
  }
  VLOG(1) << "Could not emit fast concatenate for " << instr->ToString() << ": "
          << fast_impl_reason.message();
  return EmitElementalHostKernel(instr);
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitDotFusionHostKernel(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emit dot fusion host kernel: " << fusion->name();

  // Dot fusion only supports adding a side input to the dot product.
  const HloInstruction* add = fusion->fused_expression_root();
  if (add->opcode() != HloOpcode::kAdd) {
    return Internal("Dot fusion supports only `add` root instruction");
  }

  // Check that fusion root has a single dot operand.
  bool is_dot_operand0 = add->operand(0)->opcode() == HloOpcode::kDot;
  bool is_dot_operand1 = add->operand(1)->opcode() == HloOpcode::kDot;
  if (is_dot_operand0 == is_dot_operand1) {
    return Internal("Dot fusion root instruction must have single dot operand");
  }

  int64_t dot_op_index = is_dot_operand0 ? 0 : 1;
  int64_t addend_op_index = 1 - dot_op_index;

  const HloInstruction* dot = add->operand(dot_op_index);

  // Check that we can emit LLVM IR for this dot operation.
  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      hlo_module_.config(), *dot,
      nested_ir_emitter_->target_machine_features());

  if (!IsDotCodegenStrategy(strategy)) {
    return Internal("Unsupported dot implementation strategy");
  }

  // Indices of fusion parameters that are used as dot operands and result.
  int64_t dot_lhs_pnum = dot->operand(0)->parameter_number();
  int64_t dot_rhs_pnum = dot->operand(1)->parameter_number();
  int64_t addend_pnum = add->operand(addend_op_index)->parameter_number();

  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(fusion));

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray lhs_array = kernel_prototype.arguments[dot_lhs_pnum];
  llvm_ir::IrArray rhs_array = kernel_prototype.arguments[dot_rhs_pnum];
  llvm_ir::IrArray addend_array = kernel_prototype.arguments[addend_pnum];
  llvm_ir::IrArray target_array = kernel_prototype.results[0];

  TF_RETURN_IF_ERROR(EmitDotOperation(
      *dot, target_array, lhs_array, rhs_array, &addend_array,
      /*executable_run_options_value=*/nullptr, &b, hlo_module_.config(),
      nested_ir_emitter_->target_machine_features(),
      /*allow_runtime_calls=*/false));

  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitSliceToDynamicHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit slice-to-dynamic host kernel: " << instr->name();

  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(instr));
  llvm::IRBuilder<> ir_builder(module_->getContext());
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  auto guard = nested_ir_emitter_->WithBuilder(ir_builder);
  TF_RETURN_IF_ERROR(nested_ir_emitter_->EmitSliceToDynamic(
      instr, kernel_prototype.arguments, output_array));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}

absl::StatusOr<IrEmitter2::KernelInfo>
IrEmitter2::EmitDynamicUpdateSliceHostKernel(const HloInstruction* instr) {
  if (llvm_ir::CanUpdateDynamicSliceInPlace(const_cast<HloInstruction*>(instr),
                                            nested_ir_emitter_->assignment())) {
    VLOG(2) << "Emit in-place dynamic-update-slice kernel: " << instr->name();

    TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                        EmitKernelPrototype(instr));

    llvm::IRBuilder<> b(module_->getContext());
    b.SetInsertPoint(
        kernel_prototype.function->getEntryBlock().getTerminator());

    TF_RETURN_IF_ERROR(llvm_ir::EmitDynamicUpdateSliceInPlace(
        kernel_prototype.arguments, kernel_prototype.results.front(),
        llvm_ir::IrName(instr, "in_place"), &b));

    return kernels_.emplace_back(KernelInfo(std::move(kernel_prototype),
                                            se::BlockDim(), se::ThreadDim()));
  }

  return EmitElementalHostKernel(instr);
}

absl::StatusOr<IrEmitter2::ComparatorInfo> IrEmitter2::EmitSortComparator(
    HloComputation* comparator) {
  // Find if we already emitted this comparator.
  auto info = absl::c_find_if(comparators_, [&](const ComparatorInfo& info) {
    return info.name == comparator->name();
  });
  if (info != comparators_.end()) return *info;

  // We use simple post-order schedule as we are not emitting a "real"
  // computation that requires buffer assignment.
  auto schedule = comparator->MakeInstructionPostOrder();

  // Emit LLVM IR for comparator function. We emit it as a top-level computation
  // to set external linkage and to get a pointer to compiled function later.
  TF_ASSIGN_OR_RETURN(llvm::Function * comparator_function,
                      nested_ir_emitter_->EmitComputation(
                          comparator, comparator->name(),
                          /*is_top_level_computation=*/true, schedule,
                          /*allow_reassociation=*/false));

  // Generate unwind information so that GDB can crawl through the stack frames
  // created by the JIT compiled code.
  comparator_function->setUWTableKind(llvm::UWTableKind::Default);

  return comparators_.emplace_back(
      ComparatorInfo{comparator_function->getName().str()});
}

//===----------------------------------------------------------------------===//
// Building HostKernel prototypes.
//===----------------------------------------------------------------------===//

absl::StatusOr<BufferAllocation::Slice> IrEmitter2::GetAllocationSlice(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return nested_ir_emitter_->assignment().GetUniqueSlice(instruction, index);
}

absl::StatusOr<std::vector<IrEmitter2::KernelParameter>>
IrEmitter2::GetKernelArgumentsParameters(const HloInstruction* instruction) {
  std::vector<KernelParameter> arguments;

  for (HloInstruction* operand : instruction->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                          GetAllocationSlice(operand, indexed.index));
      arguments.push_back(KernelParameter{indexed.shape, slice});
    }
  }
  return arguments;
}

absl::StatusOr<std::vector<IrEmitter2::KernelParameter>>
IrEmitter2::GetKernelResultsParameters(const HloInstruction* instruction) {
  std::vector<KernelParameter> results;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        GetAllocationSlice(instruction, indexed.index));
    results.push_back(KernelParameter{indexed.shape, slice});
  }
  return results;
}

absl::Status IrEmitter2::VerifyKernelParameters(
    absl::Span<const KernelParameter> arguments,
    absl::Span<const KernelParameter> results) {
  // IMPORTANT: Buffer slice non-overlapping property checked below does not
  // necessarily mean that the buffers do not alias. Parameter allocations
  // might have different index but at run time might be backed by the same
  // memory (or aliased memory). We conservatively do not emit noalias metadata
  // for buffers coming from parameter allocations.

  // Check that all kernel arguments are coming from non-overlapping slices. It
  // is fine to pass same slice as different arguments. This property is not
  // used anywhere during the codegen, it acts mostly as a sanity check for
  // the buffer assignment. In the future we might emit better aliasing metadata
  // based on this property.
  for (size_t i = 0; i < arguments.size(); ++i) {
    for (size_t j = i + 1; j < arguments.size(); ++j) {
      const KernelParameter& a = arguments[i];
      const KernelParameter& b = arguments[j];

      if (a.slice != b.slice && a.slice.OverlapsWith(b.slice)) {
        return Internal(
            "Kernel arguments must not overlap: result #%d (%s) overlaps "
            "with result #%d (%s)",
            i, a.slice.ToString(), j, b.slice.ToString());
      }
    }
  }

  // Check that all kernel results are unique and coming from non-overlapping
  // slices. We rely on this property to create LLVM `!alias.scope` for each
  // kernel result buffer and to construct `!noalias` metadata for arguments.
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = i + 1; j < results.size(); ++j) {
      const KernelParameter& a = results[i];
      const KernelParameter& b = results[j];

      if (a.slice.OverlapsWith(b.slice)) {
        return Internal(
            "Kernel results must not overlap: result #%d (%s) overlaps "
            "with result #%d (%s)",
            i, a.slice.ToString(), j, b.slice.ToString());
      }
    }
  }

  // Check that results do not overlap with arguments, or if they do, they must
  // be the same as one of the arguments, which can happen for inplace kernels.
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < arguments.size(); ++j) {
      const KernelParameter& result = results[i];
      const KernelParameter& argument = arguments[j];

      if (result.slice.OverlapsWith(argument.slice) &&
          result.slice != argument.slice) {
        return Internal(
            "Kernel results must not partially overlap with arguments: result "
            "#%d (%s) overlaps with argument #%d (%s)",
            i, result.slice.ToString(), j, argument.slice.ToString());
        break;
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<IrEmitter2::KernelPrototype> IrEmitter2::EmitKernelPrototype(
    absl::string_view name, absl::Span<const KernelParameter> arguments,
    absl::Span<const KernelParameter> results) {
  VLOG(3) << "Emit kernel prototype: " << name
          << ", #arguments=" << arguments.size()
          << ", #results=" << results.size();
  for (const KernelParameter& argument : arguments) {
    VLOG(3) << "  argument: " << argument.shape.ToString(true) << " in "
            << argument.slice.ToString();
  }
  for (const KernelParameter& result : results) {
    VLOG(3) << "  result: " << result.shape.ToString(true) << " in "
            << result.slice.ToString();
  }

  TF_RETURN_IF_ERROR(VerifyKernelParameters(arguments, results));

  llvm::LLVMContext& ctx = module_->getContext();
  llvm::MDBuilder mb(ctx);
  llvm::IRBuilder<> b(ctx);

  // Create an alias domain for the host kernel function.
  llvm::MDNode* domain = mb.createAliasScopeDomain(
      absl::StrFormat("XLA host kernel %s AA domain", name));

  // Emit alias scopes for all kernel result buffers. We do not emit alias
  // scopes for kernel arguments, because it's usually not profitable, and we
  // mostly care about avoiding reloading data from read-only buffers. We use
  // sorted container to make sure that emitted metadata is deterministic.
  absl::btree_map<BufferAllocation::Slice, llvm::MDNode*> alias_scopes;
  for (const KernelParameter& result : results) {
    // Skip result buffers that are aliased with entry parameters as we don't
    // know if they can alias with any other buffers.
    if (result.slice.allocation()->is_parameter_aliased_with_output()) {
      continue;
    }
    alias_scopes[result.slice] = mb.createAliasScope(
        absl::StrFormat("result slice: %s", result.slice.ToString()), domain);
  }

  // Returns alias scope for the given buffer slice.
  auto get_alias_scope = [&](BufferAllocation::Slice slice) -> llvm::MDNode* {
    auto it = alias_scopes.find(slice);
    return it == alias_scopes.end() ? nullptr
                                    : llvm::MDNode::get(ctx, it->second);
  };

  // Construct !noalias metadata for buffer slice.
  auto get_noalias = [&](BufferAllocation::Slice slice) -> llvm::MDNode* {
    llvm::SmallVector<llvm::Metadata*> scopes;
    for (const auto& [alias_slice, alias_scope] : alias_scopes) {
      if (!slice.OverlapsWith(alias_slice)) {
        scopes.push_back(alias_scope);
      }
    }
    return scopes.empty() ? nullptr : llvm::MDNode::get(ctx, scopes);
  };

  // Collect all buffer slices that the kernel writes to.
  absl::flat_hash_set<BufferAllocation::Slice> result_slices;
  result_slices.reserve(results.size());
  for (const KernelParameter& result : results) {
    result_slices.insert(result.slice);
  }

  // Create a kernel function with HostKernel API.
  llvm::Function* function =
      kernel_api_ir_builder_.EmitKernelFunction(*module_, name);

  // Create an entry basic block and set insert point to the end of it.
  b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "", function));

  llvm::Value* call_frame = function->getArg(0);
  // Build thread coordinates from the call frame.
  KernelApiIrBuilder::ThreadDims kernel_thread_dims =
      kernel_api_ir_builder_.EmitKernelThreadDims(b, call_frame);
  KernelApiIrBuilder::ThreadId kernel_thread =
      kernel_api_ir_builder_.EmitKernelThread(b, call_frame);

  int64_t idx = 0;

  // A set of invariant (read-only) buffer indices, feeded in the loop array in
  // the next section.
  absl::flat_hash_set<int64_t> invariant_arguments;

  // IrArrays for the parameters.
  std::vector<llvm_ir::IrArray> ir_arguments;
  for (int64_t i = 0; i < arguments.size(); ++i) {
    const KernelParameter& argument = arguments[i];
    auto ir_argument = kernel_api_ir_builder_.EmitKernelArgument(
        b, call_frame, idx++, argument.shape);
    if (auto* noalias = get_noalias(argument.slice)) {
      ir_argument.AddNoaliasMetadata(noalias);
    }

    // If a buffer slice is not a part of result set, then it must be invariant
    // (read-only).
    if (!result_slices.contains(argument.slice)) {
      ir_argument.MarkInvariantOverWholeProgram(&ctx);
      invariant_arguments.insert(i);
    }

    ir_arguments.push_back(std::move(ir_argument));
  }

  // IrArrays for the results.
  std::vector<llvm_ir::IrArray> ir_results;
  for (const KernelParameter& result : results) {
    auto ir_result = kernel_api_ir_builder_.EmitKernelArgument(
        b, call_frame, idx++, result.shape);
    if (auto* noalias = get_noalias(result.slice)) {
      ir_result.AddNoaliasMetadata(noalias);
    }
    if (auto* alias_scope = get_alias_scope(result.slice)) {
      ir_result.AddAliasScopeMetadata(alias_scope);
    }
    ir_results.push_back(std::move(ir_result));
  }

  // Return null pointer to signal success as we do not support error handling
  // in the compiled host kernel.
  llvm::BasicBlock* return_block =
      llvm::BasicBlock::Create(ctx, "return", function);

  b.CreateBr(return_block);

  b.SetInsertPoint(return_block);
  b.CreateRet(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx)));

  return KernelPrototype{function,
                         return_block,
                         kernel_thread_dims,
                         kernel_thread,
                         std::move(ir_arguments),
                         std::move(ir_results),
                         std::move(invariant_arguments)};
}

absl::StatusOr<IrEmitter2::KernelPrototype> IrEmitter2::EmitKernelPrototype(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(std::vector<KernelParameter> arguments,
                      GetKernelArgumentsParameters(instr));
  TF_ASSIGN_OR_RETURN(std::vector<KernelParameter> results,
                      GetKernelResultsParameters(instr));
  return EmitKernelPrototype(instr->name(), std::move(arguments),
                             std::move(results));
}

std::optional<IrEmitter2::ParallelConfig> IrEmitter2::GetParallelConfig(
    const HloInstruction* instr) {
  // Check if the instruction is marked for parallel execution.
  auto backend_config = instr->backend_config<BackendConfig>();
  if (!backend_config.ok() ||
      backend_config->outer_dimension_partitions().empty()) {
    return std::nullopt;
  }

  ParallelConfig config;
  config.outer_dimension_partitions.assign(
      backend_config->outer_dimension_partitions().begin(),
      backend_config->outer_dimension_partitions().end());

  return config;
}

absl::Status IrEmitter2::CanDoFastConcatenate(
    const HloInstruction* concatenate) const {
  if (!concatenate->parent()
           ->root_instruction()
           ->template backend_config<BackendConfig>()
           ->outer_dimension_partitions()
           .empty()) {
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        "Cannot generate memcpy-based concat for the parallel CPU backend");
  }
  const Shape& output_shape = concatenate->shape();
  for (auto* op : concatenate->operands()) {
    if (!LayoutUtil::Equal(op->shape().layout(), output_shape.layout())) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Operand has mismatching layouts");
    }
  }
  return absl::OkStatus();
};

IrEmitter2::ParallelPartitionBounds IrEmitter2::EmitParallelPartitionBounds(
    llvm::IRBuilderBase& b, const KernelPrototype& kernel_prototype,
    const ParallelConfig& parallel_config, const Shape& shape,
    absl::string_view name) {
  ShapePartitionIterator it(shape, parallel_config.outer_dimension_partitions);

  size_t num_parallel_dimensions =
      parallel_config.outer_dimension_partitions.size();

  // Create a constant array of all partition bounds. We will be indexing into
  // this array using block and thread dimension indices passed in a call frame.
  //
  // Type: [#partitions x [#outer_dimensions x [lower_bound, upper_bound]]]
  //
  llvm::ArrayType* dim_bounds_ty = llvm::ArrayType::get(b.getInt64Ty(), 2);
  llvm::ArrayType* partition_bounds_ty =
      llvm::ArrayType::get(dim_bounds_ty, num_parallel_dimensions);
  llvm::ArrayType* parallel_bounds_ty =
      llvm::ArrayType::get(partition_bounds_ty, it.GetTotalPartitionCount());

  // Build a nested array of partition bounds from shape partition iterator.
  std::vector<llvm::Constant*> partition_bounds;
  for (int64_t i = 0; i < it.GetTotalPartitionCount(); ++i) {
    std::vector<llvm::Constant*> dim_counts;
    for (auto [lower, size] : it.GetPartition(i)) {
      dim_counts.push_back(llvm::ConstantArray::get(
          dim_bounds_ty, {b.getInt64(lower), b.getInt64(lower + size)}));
    }
    partition_bounds.push_back(
        llvm::ConstantArray::get(partition_bounds_ty, dim_counts));
  }

  llvm::Constant* parallel_bounds =
      llvm::ConstantArray::get(parallel_bounds_ty, partition_bounds);

  llvm::Module* module = b.GetInsertBlock()->getParent()->getParent();
  llvm::GlobalVariable* parallel_bounds_global = new llvm::GlobalVariable(
      /*M=*/*module,
      /*Ty=*/parallel_bounds_ty,
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/parallel_bounds,
      /*Name=*/absl::StrCat(name, "_parallel_bounds"));

  // Construct IR to load bounds for all parallel dimensions.
  ParallelPartitionBounds bounds;
  for (size_t i = 0; i < num_parallel_dimensions; ++i) {
    llvm::Value* partition = kernel_prototype.thread.x;
    llvm::Value* parallel_dim = b.getInt32(i);

    llvm::Value* lower_gep = b.CreateInBoundsGEP(
        parallel_bounds_ty, parallel_bounds_global,
        {b.getInt32(0), partition, parallel_dim, b.getInt32(0)},
        absl::StrCat("lo_dim_", i, "_gep"));

    llvm::Value* upper_gep = b.CreateInBoundsGEP(
        parallel_bounds_ty, parallel_bounds_global,
        {b.getInt32(0), partition, parallel_dim, b.getInt32(1)},
        absl::StrCat("up_dim_", i, "_gep"));

    bounds.emplace_back(
        b.CreateLoad(b.getInt64Ty(), lower_gep, absl::StrCat("lo_dim_", i)),
        b.CreateLoad(b.getInt64Ty(), upper_gep, absl::StrCat("up_dim_", i)));
  }

  return bounds;
}

absl::StatusOr<se::ThreadDim> IrEmitter2::EmitElementalLoops(
    llvm::IRBuilderBase& b, const HloInstruction* instr,
    const KernelPrototype& kernel_prototype,
    const llvm_ir::ElementGenerator& element_generator) {
  // We can emit loops for instruction with multiple results only if it is a
  // fusion, reduce or reduce window.
  bool multiple_results = kernel_prototype.results.size() > 1;
  bool support_multiple_results = instr->opcode() == HloOpcode::kFusion ||
                                  instr->opcode() == HloOpcode::kReduce ||
                                  instr->opcode() == HloOpcode::kReduceWindow;

  auto parallel_config = GetParallelConfig(instr);
  bool has_parallel_config = parallel_config.has_value();

  if (multiple_results && !support_multiple_results) {
    return Internal(
        "Multi-output host kernels are not supported for %s instruction",
        HloOpcodeString(instr->opcode()));
  }

  // TODO(ezhulenev): Support multiple results for parallel loops.
  if (multiple_results) {
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, kernel_prototype.results, &b)
            .EmitLoop(llvm_ir::IrName(instr)));
    return se::ThreadDim();
  }

  const llvm_ir::IrArray& result = kernel_prototype.results.front();

  // Emit a loop for a single parallel partition with dynamic bounds computed
  // from thread index.
  if (has_parallel_config) {
    ParallelPartitionBounds parallel_bounds = EmitParallelPartitionBounds(
        b, kernel_prototype, *parallel_config, instr->shape(), instr->name());
    TF_RETURN_IF_ERROR(
        ParallelLoopEmitter(element_generator, result, &parallel_bounds, &b)
            .EmitLoop(llvm_ir::IrName(instr)));
    return se::ThreadDim(ShapePartitionAssigner::GetTotalPartitionCount(
        parallel_config->outer_dimension_partitions));
  }

  // Emit a whole loop for the instruction.
  TF_RETURN_IF_ERROR(llvm_ir::LoopEmitter(element_generator, result, &b)
                         .EmitLoop(llvm_ir::IrName(instr)));
  return se::ThreadDim();
}

absl::Status IrEmitter2::EmitNestedComputation(const HloComputation& callee,
                                               absl::string_view name,
                                               bool is_reducer) {
  // Module must be scheduled to emit thread local computation.
  if (!hlo_module_.has_schedule()) {
    return absl::InternalError(
        "HLO module must be scheduled to emit thread local computation.");
  }

  if (nested_ir_emitter_->is_computation_emitted(callee, is_reducer)) {
    return absl::OkStatus();
  }

  for (HloInstruction* instr : callee.instructions()) {
    bool nested_is_reducer = instr->opcode() == HloOpcode::kReduce ||
                             instr->opcode() == HloOpcode::kReduceWindow;
    for (HloComputation* called_computation : instr->called_computations()) {
      // reassociation is transitive so we "or" the caller and the callee.
      TF_RETURN_IF_ERROR(
          EmitNestedComputation(*called_computation, llvm_ir::IrName(instr),
                                is_reducer || nested_is_reducer));
    }
  }

  if (callee.IsFusionComputation()) {
    return absl::OkStatus();
  }

  VLOG(2) << "Emit nested computation: " << callee.name();
  return nested_ir_emitter_
      ->EmitComputation(const_cast<HloComputation*>(&callee), name, false,
                        hlo_module_.schedule().sequence(&callee).instructions(),
                        /*allow_reassociation=*/is_reducer,
                        /*function_attributes=*/{llvm::Attribute::AlwaysInline})
      .status();
}

// This is a convenience function taken from IrEmitter, it uses module_ class
// field. If there will be more functions that use module_, we should consider
// refactoring (like we did for compute_function_ and builder_).
int64_t IrEmitter2::ByteSizeOf(const Shape& shape) const {
  return llvm_ir::ByteSizeOf(shape, module_->getDataLayout());
}

void IrEmitter2::AttachInvariantLoadMetadataForLoad(
    llvm::LoadInst* instr) const {
  nested_ir_emitter_->AttachInvariantLoadMetadataForLoad(instr,
                                                         hlo_module_.config());
}

}  // namespace xla::cpu
