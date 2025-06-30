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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
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
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CodeGen.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"
#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter_config.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/symbol_name_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/parallel_loop_emitter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_partition.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

namespace {

// Explicitly in HLO we mostly see "loop" types for fusions.
// However, internally we pick a fusion kind to pick the appropriate
// fusion emitter.
enum class FusionEmitterKind {
  kLoop,
  kScatter,
};

// This is very crude at the moment. Eventually we will need to either have
// the fusion indicate what emitter it is meant for (e.g. producing this
// info via a cost model), or heuristics to estimate which emitter is best
// for the fusion.
FusionEmitterKind AnalyzeHloFusion(const HloFusionInstruction* fusion) {
  if (fusion->fused_expression_root()->opcode() == HloOpcode::kScatter) {
    return FusionEmitterKind::kScatter;
  }
  return FusionEmitterKind::kLoop;
}

std::string SortCsv(absl::string_view csv) {
  std::vector<absl::string_view> v =
      absl::StrSplit(csv, ',', absl::SkipEmpty());
  std::sort(v.begin(), v.end());
  return absl::StrJoin(v, ",");
}

}  // namespace

//===----------------------------------------------------------------------===//
// IrEmitter2
//===----------------------------------------------------------------------===//

IrEmitter2::IrEmitter2(const HloModule& hlo_module, llvm::Module* module,
                       IrEmitter* nested_ir_emitter)
    : hlo_module_(hlo_module),
      module_(module),
      nested_ir_emitter_(nested_ir_emitter),
      kernel_api_ir_builder_(module_->getContext(),
                             KernelApiIrBuilder::Options::FromHloModuleConfig(
                                 hlo_module_.config())) {}

bool IrEmitter2::fast_min_max() const {
  return hlo_module_.config().debug_options().xla_cpu_enable_fast_min_max();
}

bool IrEmitter2::IsSupportedByFusionEmitter(
    const HloFusionInstruction* fusion) const {
  if (!hlo_module_.config().debug_options().xla_cpu_use_fusion_emitters()) {
    return false;
  }
  FusionEmitterKind fusion_emitter_kind = AnalyzeHloFusion(fusion);
  switch (fusion_emitter_kind) {
    case FusionEmitterKind::kScatter:
      return kFusionEmitterScatterEnabled;
    default:
      return false;
  }
}

IrEmitter2::KernelInfo::KernelInfo(KernelPrototype prototype,
                                   const se::BlockDim& block_dims,
                                   const se::ThreadDim& thread_dims)
    : name(prototype.function->getName().str()),
      block_dims(block_dims),
      thread_dims(thread_dims),
      invariant_arguments(std::move(prototype.invariant_arguments)) {}

IrEmitter2::KernelInfo::KernelInfo(
    const std::string& name, const se::BlockDim& block_dims,
    const se::ThreadDim& thread_dims,
    const absl::flat_hash_set<int64_t>& invariant_arguments,
    absl::string_view backend_extra_options)
    : name(name),
      block_dims(block_dims),
      thread_dims(thread_dims),
      invariant_arguments(invariant_arguments),
      backend_extra_options(SortCsv(backend_extra_options)) {}

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
  TF_RETURN_IF_ERROR(nested_ir_emitter_
                         ->EmitNestedComputation(*nested_computation,
                                                 llvm_ir::IrName(fusion), false)
                         .status());

  CpuElementalIrEmitter elemental_emitter = ElementalIrEmmiterFactory(&b);

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
      hlo_module_.config(), *dot, nested_ir_emitter_->target_machine_features(),
      /*allow_runtime_calls=*/false);

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
  DCHECK(CanUpdateDynamicSliceInPlace(instr));

  VLOG(2) << "Emit in-place dynamic-update-slice kernel: " << instr->name();

  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(instr));

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  TF_RETURN_IF_ERROR(llvm_ir::EmitDynamicUpdateSliceInPlace(
      kernel_prototype.arguments, kernel_prototype.results.front(),
      llvm_ir::IrName(instr, "in_place"), &b));

  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}

absl::StatusOr<IrEmitter2::ComparatorInfo> IrEmitter2::EmitSortComparator(
    HloComputation* comparator) {
  std::string comparator_name(comparator->name().data(),
                              comparator->name().size());
  if (hlo_module_.config()
          .debug_options()
          .xla_cpu_generate_unique_c_style_kernel_entry_points()) {
    TF_ASSIGN_OR_RETURN(comparator_name, ConvertToCName(absl::StrCat(
                                             comparator->parent()->name(), "_",
                                             comparator->name())));
  }

  // Find if we already emitted this comparator.
  auto info = absl::c_find_if(comparators_, [&](const ComparatorInfo& info) {
    return info.name == comparator_name;
  });
  if (info != comparators_.end()) return *info;

  // We use simple post-order schedule as we are not emitting a "real"
  // computation that requires buffer assignment.
  auto schedule = comparator->MakeInstructionPostOrder();

  // Emit LLVM IR for comparator function. We emit it as a top-level computation
  // to set external linkage and to get a pointer to compiled function later.
  TF_ASSIGN_OR_RETURN(llvm::Function * comparator_function,
                      nested_ir_emitter_->EmitComputation(
                          comparator, comparator_name,
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

absl::StatusOr<IrEmitter2::KernelPrototype> IrEmitter2::EmitKernelPrototype(
    const HloInstruction* instr) {
  return kernel_api_ir_builder_.EmitKernelPrototype(
      *module_, instr, &nested_ir_emitter_->assignment(), "ir_emitter2");
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

bool IrEmitter2::CanUpdateDynamicSliceInPlace(
    const HloInstruction* update) const {
  return llvm_ir::CanUpdateDynamicSliceInPlace(
      const_cast<HloInstruction*>(update), nested_ir_emitter_->assignment());
}

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
    llvm::Value* partition = kernel_prototype.workgroup_id.x;
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

CpuElementalIrEmitter IrEmitter2::ElementalIrEmmiterFactory(
    llvm::IRBuilderBase* b) const {
  auto thread_local_call_fn = [this](const HloComputation& callee,
                                     absl::Span<llvm::Value* const> parameters,
                                     absl::string_view name, bool is_reducer) {
    return nested_ir_emitter_->EmitThreadLocalCall(
        callee, parameters, name, is_reducer,
        /*in_compute_function=*/false);
  };

  return CpuElementalIrEmitter(module_, b, thread_local_call_fn, true,
                               fast_min_max());
}

}  // namespace xla::cpu
