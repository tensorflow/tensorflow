/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/reduction.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/elemental_ir_emitter.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/reduction_base.h"
#include "xla/service/gpu/fusions/thunk_util.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_nested.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using TypedPointer = std::pair<llvm::Value* const, llvm::Type* const>;

// Fusion root -> array of indexes, one per reduction output.
using ReductionOutputMap =
    ConstHloInstructionMap<absl::Span<llvm_ir::IrArray const>>;

using ExtraOutputGensMap = ConstHloInstructionMap<llvm_ir::ElementGenerator>;

int GetNumOutputs(const Shape& shape) {
  if (shape.IsTuple()) {
    return shape.tuple_shapes_size();
  }
  return 1;
}

const Shape& OutputShape(const Shape& output_shape, int output_index) {
  CHECK(output_index == 0 || output_shape.IsTuple());
  return output_shape.IsTuple() ? output_shape.tuple_shapes(output_index)
                                : output_shape;
}

llvm::Type* GetIndexType(const HloFusionInstruction& fusion,
                         const Tiling& tiling, llvm::IRBuilder<>* builder) {
  return GetIndexTypeForKernel(
      &fusion, tiling.GetNumThreadsPerBlock() * tiling.GetNumBlocks(), builder);
}

llvm::Value* CastSharedToGlobal(llvm::IRBuilder<>* builder, llvm::Value* input,
                                llvm::Type* element_type, llvm::Twine name) {
  return builder->CreateAddrSpaceCast(
      input,
      llvm::PointerType::get(element_type,
                             /*AddressSpace=*/0),
      name);
}

class ReductionEmitter {
 public:
  ReductionEmitter(const HloFusionAnalysis& analysis,
                   const ReductionInfo& reduction_codegen_info,
                   IrEmitterContext& ir_emitter_context,
                   const HloFusionInstruction& fusion,
                   llvm::IRBuilder<>* builder)
      : builder_(builder),
        elemental_emitter_(ir_emitter_context, builder_),
        analysis_(analysis),
        reduction_codegen_info_(reduction_codegen_info),
        ir_emitter_context_(ir_emitter_context),
        fusion_(fusion),
        index_ty_(GetIndexType(fusion, reduction_codegen_info.GetTiling(),
                               elemental_emitter_.builder())) {
    for (auto hero : analysis.fusion_heroes()) {
      if (hero.opcode() == HloOpcode::kReduce) {
        for (int i = 0; i < hero.instruction().operand_count() / 2; ++i) {
          CHECK(LayoutUtil::IsMonotonicWithDim0Major(
              hero.instruction().operand(i)->shape().layout()))
              << "reduction-layout-normalizer must run before code generation";
        }
      }
    }
  }

  absl::StatusOr<FusionEmissionResult> EmitInitializers();
  absl::Status EmitKernel(const LaunchDimensions& launch_dims,
                          std::vector<llvm_ir::IrArray> inputs,
                          std::vector<llvm_ir::IrArray> outputs);

 private:
  friend class ReductionGroupEmitter;

  absl::StatusOr<std::unique_ptr<Thunk>> BuildKernelThunkForFusion(
      const LaunchDimensions& launch_dimensions,
      absl::string_view discriminator,
      std::function<absl::Status(std::vector<llvm_ir::IrArray>,
                                 std::vector<llvm_ir::IrArray>)>
          kernel_builder_fn);

  absl::StatusOr<std::unique_ptr<Thunk>> BuildFusedInitializerThunk(
      const HloInstruction* fusion_root, BufferAllocation::Slice dest_slice,
      int output_index);

  absl::Status EmitIRForReduction(
      absl::Span<const HloInstruction* const> instr_index_group,
      FusedIrEmitter& fused_emitter, const ReductionOutputMap& result_ir_arrays,
      const Shape& input_shape);

  void MaybeEmitFenceForAMDGPU();
  void EmitSyncThreads();

  int ReducedDimensionSize() const {
    return reduction_codegen_info_.GetTiling().GetShape()[2];
  }

  llvm::IRBuilder<>* builder_;
  GpuElementalIrEmitter elemental_emitter_;
  const HloFusionAnalysis& analysis_;
  const ReductionInfo& reduction_codegen_info_;
  IrEmitterContext& ir_emitter_context_;
  const HloFusionInstruction& fusion_;
  llvm::Type* index_ty_;
};

class ReductionEmitter;

class ReductionGroupEmitter {
 public:
  struct ReductionCalculationState {
    std::optional<llvm_ir::SharedMemoryTile> shared_cache;
    llvm::Value* initial_value;
    llvm::AllocaInst* partial_result_address;
    llvm::AllocaInst* input_address;
    llvm_ir::ElementGenerator input_gen;
  };

  ReductionGroupEmitter(
      ReductionEmitter& reduction_emitter,
      absl::Span<const HloReduceInstruction* const> reduce_instr_index_group,
      const ReductionOutputMap& result_ir_arrays,
      FusedIrEmitter& fused_emitter);

  const ReductionCalculationState& GetCalculationStateFor(
      const HloInstruction* instruction, int operand_idx) const {
    const ReductionOpState& op_state = state_.at(instruction);
    CHECK_LT(operand_idx, op_state.size());
    return op_state[operand_idx];
  }

  void SetCalculationStateFor(
      const ReductionCalculationState& calculation_state,
      const HloInstruction* instruction, int operand_idx) {
    ReductionOpState& op_state = state_[instruction];
    CHECK_EQ(operand_idx, op_state.size());
    op_state.push_back(calculation_state);
  }

  void EmitReductionOutputForRowReduction(
      const TilingKernelInfo& tiling_kernel_info,
      const HloReduceInstruction* reduction,
      const std::vector<const HloInstruction*>& roots) const;

  void EmitReductionOutputForColumnReduction(
      const TilingKernelInfo& tiling_kernel_info,
      const HloReduceInstruction* reduction,
      const std::vector<const HloInstruction*>& roots) const;

  void EmitFullWarpShuffleDownLoopForReduce(
      const HloComputation* reducer,
      absl::Span<TypedPointer const> partial_result_addresses,
      int threads_per_block, int num_results_per_warp) const;

  void WriteReductionOutput(const TilingKernelInfo& tiling_kernel_info,
                            const HloReduceInstruction* reduction,
                            const std::vector<const HloInstruction*>& roots,
                            absl::Span<TypedPointer const> values) const;

  llvm_ir::IrArray::Index GetOutputIndexForReduction(
      const TilingKernelInfo& tiling_kernel_info,
      const HloReduceInstruction* reduction, const HloInstruction* root,
      int output_idx) const;

  void GenerateElementForReducer(const HloReduceInstruction* reduction,
                                 const llvm_ir::IrArray::Index& index) const;

  absl::Status EmitExtraOutputsForReduce(
      const Shape& reduction_operand_shape,
      const llvm_ir::IrArray::Index& index,
      const ExtraOutputGensMap& extra_output_gens);

 private:
  ReductionEmitter& reduction_emitter_;
  const ReductionOutputMap& result_ir_arrays_;

  // One state per reduction operand.
  using ReductionOpState = absl::InlinedVector<ReductionCalculationState, 2>;

  // HloInstruction -> operand_idx -> cache
  absl::flat_hash_map<const HloInstruction*, ReductionOpState> state_;
};

// Creates accumulator alloca's, populates them with initial values, generates
// __shared__ caches and returns the populated object.
ReductionGroupEmitter::ReductionGroupEmitter(
    ReductionEmitter& reduction_emitter,
    absl::Span<const HloReduceInstruction* const> reduce_instr_index_group,
    const ReductionOutputMap& result_ir_arrays, FusedIrEmitter& fused_emitter)
    : reduction_emitter_(reduction_emitter),
      result_ir_arrays_(result_ir_arrays) {
  const ReductionInfo& reduction_info =
      reduction_emitter_.reduction_codegen_info_;
  VLOG(10) << "Emit prologue for reduction: "
           << reduction_emitter_.fusion_.ToString();

  auto* builder = reduction_emitter_.builder_;
  for (const HloReduceInstruction* reduce_hlo : reduce_instr_index_group) {
    for (int op_result_idx = 0;
         op_result_idx < GetNumOutputs(reduce_hlo->shape()); op_result_idx++) {
      Shape result_shape = OutputShape(reduce_hlo->shape(), op_result_idx);

      llvm::Type* element_type = llvm_ir::PrimitiveTypeToIrType(
          result_shape.element_type(), builder->GetInsertBlock()->getModule());
      llvm::AllocaInst* reduction_input_address =
          llvm_ir::EmitAllocaAtFunctionEntry(
              element_type, "reduction_input_address", builder);

      llvm::AllocaInst* result_address = llvm_ir::EmitAllocaAtFunctionEntry(
          element_type, "partial_reduction_result", builder);

      const HloInstruction* init_value =
          reduce_hlo->init_values()[op_result_idx];

      // Initialize the partial result with the initial value of the reduction.
      llvm::Value* init_ir_value = (*fused_emitter.GetGenerator(
          *init_value))(llvm_ir::IrArray::Index(builder->getInt32Ty()))
                                       .value();

      builder->CreateStore(init_ir_value, result_address);
      const Tiling& tiling = reduction_info.GetTiling();
      auto shared_cache = [&]() -> std::optional<llvm_ir::SharedMemoryTile> {
        auto* module = reduction_emitter.ir_emitter_context_.llvm_module();
        if (reduction_info.IsRowReduction()) {
          // Multi-row reductions do not use shared memory.
          if (RowReductionGetRowsPerWarp(
                  reduction_emitter_.ReducedDimensionSize()) > 1) {
            return std::nullopt;
          }
          // Allocate one shared memory element per warp.
          auto block_size = tiling.GetThreadsPerBlock();
          CHECK_EQ(block_size[ReductionDimensions::kRowMinorReducedDimension] %
                       WarpSize(),
                   0);
          return llvm_ir::AllocateSharedMemoryTile(
              module, element_type,
              {block_size[ReductionDimensions::kRowKeptDimension],
               block_size[ReductionDimensions::kRowMinorReducedDimension] /
                   WarpSize()},
              "shared_cache");
        }
        const auto& num_threads = tiling.GetThreadsPerBlock();
        int n = num_threads[ReductionDimensions::kColReducedDimension];
        CHECK_EQ(n, num_threads[ReductionDimensions::kColMinorKeptDimension]);
        // The "+1" is used to avoid bank conflicts.
        return llvm_ir::AllocateSharedMemoryTile(module, element_type,
                                                 {n, n + 1}, "shared_cache");
      }();

      llvm_ir::ElementGenerator input_gen =
          *fused_emitter.GetGenerator(*reduce_hlo->inputs()[op_result_idx]);
      SetCalculationStateFor({shared_cache, init_ir_value, result_address,
                              reduction_input_address, input_gen},
                             reduce_hlo, op_result_idx);
    }
  }
}

void ReductionEmitter::MaybeEmitFenceForAMDGPU() {
  auto* module = builder_->GetInsertBlock()->getModule();
  if (IsAMDGPU(module) &&
      ir_emitter_context_.rocm_compute_capability().fence_before_barrier()) {
    builder_->CreateFence(
        llvm::AtomicOrdering::SequentiallyConsistent,
        builder_->getContext().getOrInsertSyncScopeID("workgroup"));
  }
}

void ReductionEmitter::EmitSyncThreads() {
  MaybeEmitFenceForAMDGPU();
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kBarrierId, {}, {}, builder_);
}

// Builds a thunk that calls a new or reused kernel for a fusion operation.
//
// The caller must specify the same launch dimensions for fusions which have
// the same computation.
//
// If a given fusion is implemented using multiple kernels, then for each
// kernel we should provide a discriminator, such as "init" and "impl".
//
// The builder_fn is only invoked if the kernel couldn't be reused.
//
// This is the typical usage pattern of this method:
//
// ```
// auto builder_fn = [](std::vector<llvm_ir::IrArray> inputs,
//                      std::vector<llvm_ir::IrArray> outputs) { ... };
// TF_ASSIGN_OR_RETURN(
//   auto thunk,
//   BuildKernelThunkForFusion(..., launch_dimensions, builder_fn));
// AddThunkToThunkSequence(std::move(thunk))
// ```
absl::StatusOr<std::unique_ptr<Thunk>>
ReductionEmitter::BuildKernelThunkForFusion(
    const LaunchDimensions& launch_dimensions, absl::string_view discriminator,
    std::function<absl::Status(std::vector<llvm_ir::IrArray>,
                               std::vector<llvm_ir::IrArray>)>
        kernel_builder_fn) {
  const HloComputation* fused_computation =
      fusion_.fused_instructions_computation();
  std::string suggested_kernel_name = std::string(fusion_.name());

  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      KernelArguments::Create(
                          ir_emitter_context_.buffer_assignment(), &fusion_));

  auto [status_or_entry, cached] =
      ir_emitter_context_.kernel_cache().GetWithStatus(
          fused_computation, kernel_arguments.args(), discriminator,
          [&]() -> absl::StatusOr<KernelReuseCache::Entry> {
            llvm::Function* kernel;
            std::vector<llvm_ir::IrArray> input_arrays;
            std::vector<llvm_ir::IrArray> output_arrays;
            TF_ASSIGN_OR_RETURN(
                std::tie(kernel, input_arrays, output_arrays),
                BuildKernelPrototype(ir_emitter_context_, suggested_kernel_name,
                                     kernel_arguments.args(),
                                     fusion_.operand_count(), launch_dimensions,
                                     builder_));
            TF_RETURN_IF_ERROR(kernel_builder_fn(input_arrays, output_arrays));
            // Shared memory is allocated statically.
            return {{kernel->getName().str(), launch_dimensions,
                     /*cluster_dim=*/std::nullopt,
                     /*shmem_bytes=*/0}};
          });
  TF_ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry, status_or_entry);
  if (cached) {
    VLOG(3) << "Reuse: " << suggested_kernel_name << " -> "
            << entry->kernel_name;
  }

  return std::make_unique<KernelThunk>(
      &fusion_, entry->kernel_name, kernel_arguments.args(), launch_dimensions,
      entry->cluster_dim, entry->shmem_bytes);
}

absl::Status ReductionGroupEmitter::EmitExtraOutputsForReduce(
    const Shape& reduction_operand_shape, const llvm_ir::IrArray::Index& index,
    const ExtraOutputGensMap& extra_output_gens) {
  if (extra_output_gens.empty()) {
    return absl::OkStatus();
  }

  auto* builder = reduction_emitter_.builder_;
  // Compute all extra output values before writing them. This avoids
  // overwriting aliased input/output buffers before all reads occurred.
  std::vector<std::pair<const HloInstruction*, llvm::Value*>>
      extra_output_ir_values;
  extra_output_ir_values.reserve(extra_output_gens.size());

  auto get_index = [&](const HloInstruction* instr) {
    const Shape& s = instr->shape();
    return ShapeUtil::EqualIgnoringElementType(reduction_operand_shape, s)
               ? index
               : index.SourceIndexOfBitcast(reduction_operand_shape, s,
                                            builder);
  };

  for (const auto& [instr, generator] : extra_output_gens) {
    TF_ASSIGN_OR_RETURN(llvm::Value* const extra_output_ir_value,
                        generator(get_index(instr)));
    extra_output_ir_values.emplace_back(instr, extra_output_ir_value);
  }

  for (const auto& [instr, generator] : extra_output_ir_values) {
    absl::Span<llvm_ir::IrArray const> result_ir = result_ir_arrays_.at(instr);
    CHECK_EQ(result_ir.size(), 1);
    result_ir[0].EmitWriteArrayElement(get_index(instr), generator, builder);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Thunk>>
ReductionEmitter::BuildFusedInitializerThunk(const HloInstruction* fusion_root,
                                             BufferAllocation::Slice dest_slice,
                                             int output_index) {
  const HloReduceInstruction* reduce =
      DynCast<HloReduceInstruction>(fusion_root);
  TF_RET_CHECK(reduce);

  const HloInstruction* init_value = reduce->init_values()[0];
  TF_ASSIGN_OR_RETURN(
      std::optional<std::unique_ptr<Thunk>> constant_init_thunk,
      BuildConstantInitializerThunk(ir_emitter_context_, fusion_root,
                                    init_value, dest_slice));
  if (constant_init_thunk) {
    return *std::move(constant_init_thunk);
  }

  const Shape& dest_shape = fusion_root->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      dest_shape, ir_emitter_context_.gpu_device_info());
  const HloComputation* fused_computation =
      fusion_.fused_instructions_computation();

  auto builder_fn = [&](std::vector<llvm_ir::IrArray> inputs,
                        std::vector<llvm_ir::IrArray> outputs) -> absl::Status {
    FusedIrEmitter fused_emitter(elemental_emitter_);
    for (int i = 0; i < fused_computation->num_parameters(); i++) {
      fused_emitter.BindGenerator(
          *fused_computation->parameter_instruction(i),
          [builder = builder_,
           input = inputs[i]](llvm_ir::IrArray::Index index) {
            return input.EmitReadArrayElement(index, builder);
          });
    }
    HloInstruction* instr = fused_computation->root_instruction();
    if (instr->opcode() == HloOpcode::kTuple) {
      instr = instr->mutable_operand(output_index);
    } else {
      CHECK_EQ(0, output_index);
    }
    TF_RET_CHECK(instr->shape().IsArray());
    TF_ASSIGN_OR_RETURN(auto generator,
                        fused_emitter.GetGenerator(*instr->operand(1)));
    TF_RETURN_IF_ERROR(ParallelLoopEmitter(generator, {outputs[output_index]},
                                           launch_dimensions, builder_)
                           .EmitLoop(fusion_.name()));
    return absl::OkStatus();
  };

  return BuildKernelThunkForFusion(launch_dimensions,
                                   /*discriminator=*/
                                   absl::StrCat("init_", output_index),
                                   builder_fn);
}

// Emits shuffle-down reduction for the `partial_result_address` using the
// reduction computation `reducer`, writes output into
// `partial_result_address`.
//
// Multiple partial_result_address inputs happen when doing variadic
// reduction: each one should get the output value.
void ReductionGroupEmitter::EmitFullWarpShuffleDownLoopForReduce(
    const HloComputation* reducer,
    absl::Span<TypedPointer const> partial_result_addresses,
    int threads_per_block, int num_results_per_warp) const {
  // This only works when the block size is a multiple of 32 threads.
  // We check this here as a mistake in the number of threads per
  // block is very hard to detect.
  CHECK_EQ(threads_per_block % 32, 0);
  CHECK_EQ(WarpSize() % num_results_per_warp, 0);

  auto* builder = reduction_emitter_.builder_;
  for (int distance = 16 / num_results_per_warp; distance >= 1; distance /= 2) {
    absl::InlinedVector<llvm::Value*, 2> reduction_params;

    for (auto acc : partial_result_addresses) {
      reduction_params.push_back(acc.first);
    }

    for (auto [partial_result_address, element_type] :
         partial_result_addresses) {
      int bit_width = llvm_ir::GetSizeInBits(element_type);
      llvm::Value* result_from_other_lane = llvm_ir::EmitAllocaAtFunctionEntry(
          element_type, "result_from_other_lane", builder);

      reduction_params.push_back(result_from_other_lane);

      // Bitcast cannot be applied to aggregate types (even packed ones), so
      // we bitcast addresses of load/store to intN* of the same bit-width.
      llvm::Type* shuffled_value_type = element_type->isStructTy()
                                            ? builder->getIntNTy(bit_width)
                                            : element_type;

      llvm::Value* partial_result =
          builder->CreateLoad(shuffled_value_type, partial_result_address,
                              "partial_reduction_result");
      builder->CreateStore(
          EmitFullWarpShuffleDown(partial_result, builder->getInt32(distance),
                                  builder),
          result_from_other_lane);
    }

    absl::StatusOr<std::vector<llvm::Value*>> returned_scalars =
        CallNestedComputationWithScalarAddrs(
            builder, reduction_emitter_.ir_emitter_context_, *reducer,
            reduction_params);
    TF_CHECK_OK(returned_scalars.status());

    for (int i = 0; i < returned_scalars->size(); i++) {
      builder->CreateStore(/*Val=*/returned_scalars->at(i),
                           /*Ptr=*/partial_result_addresses[i].first);
    }
  }
}

llvm_ir::IrArray::Index ReductionGroupEmitter::GetOutputIndexForReduction(
    const TilingKernelInfo& tiling_kernel_info,
    const HloReduceInstruction* reduction, const HloInstruction* root,
    int output_idx) const {
  auto* builder = reduction_emitter_.builder_;
  auto* index_ty = reduction_emitter_.index_ty_;

  // 1d or 2d output index (for row/column reduction).
  auto projected_index = [&]() -> llvm_ir::IrArray::Index {
    const auto& reduction_info = reduction_emitter_.reduction_codegen_info_;
    const auto& offset = tiling_kernel_info.tile_origin;
    const auto& shape = reduction_info.GetTiling().GetXlaShape();
    const auto& thread_ids = tiling_kernel_info.thread_id_info.thread_ids;
    if (reduction_info.IsRowReduction()) {
      constexpr int kDim = ReductionDimensions::kRowKeptDimension;
      return {{builder->CreateAdd(offset[kDim], thread_ids[kDim])},
              {shape.dimensions(kDim)},
              index_ty};
    }
    auto* major_idx = offset[ReductionDimensions::kColMajorKeptDimension];
    auto* minor_idx = builder->CreateAdd(
        offset[ReductionDimensions::kColMinorKeptDimension],
        thread_ids[ReductionDimensions::kColReducedDimension]);
    return {{major_idx, minor_idx},
            ShapeUtil::DeleteDimension(
                ReductionDimensions::kColReducedDimension, shape),
            index_ty};
  }();

  auto physical_shape = ShapeUtil::DeleteDimensions(
      reduction->dimensions(), reduction->operand(output_idx)->shape());
  auto physical_index =
      projected_index.SourceIndexOfBitcast(physical_shape, builder);
  return llvm_ir::IrArray::Index(physical_index.multidim(),
                                 OutputShape(reduction->shape(), output_idx),
                                 index_ty)
      .SourceIndexOfBitcast(OutputShape(root->shape(), output_idx), builder);
}

void ReductionGroupEmitter::WriteReductionOutput(
    const TilingKernelInfo& tiling_kernel_info,
    const HloReduceInstruction* reduction,
    const std::vector<const HloInstruction*>& roots,
    const absl::Span<TypedPointer const> values) const {
  auto* builder = reduction_emitter_.builder_;
  const auto& reduction_info = reduction_emitter_.reduction_codegen_info_;
  const HloComputation* reducer = reduction->to_apply();
  for (const auto& [oidx, typed_ptr] : llvm::enumerate(values)) {
    auto [output_ptr, type] = typed_ptr;
    for (auto root : roots) {
      llvm_ir::IrArray::Index output_index =
          GetOutputIndexForReduction(tiling_kernel_info, reduction, root, oidx);

      llvm::Value* output_address =
          result_ir_arrays_.at(root)[oidx].EmitArrayElementAddress(
              output_index, builder, "output_element_address");
      if (reduction_info.IsRaceFree()) {
        FusedIrEmitter fused_emitter(reduction_emitter_.elemental_emitter_);
        llvm::Value* loaded = builder->CreateLoad(type, output_ptr, "output");
        fused_emitter.BindGenerator(
            *reduction,
            [&](const llvm_ir::IrArray::Index& index) { return loaded; });
        llvm_ir::ElementGenerator gen = *fused_emitter.GetGenerator(*root);
        llvm::Value* generated = *gen(output_index);
        builder->CreateStore(generated, output_address);
      } else {
        CHECK_EQ(values.size(), 1);
        CHECK_EQ(roots.size(), 1);
        CHECK_EQ(reduction, root)
            << "output fusion is not allowed for racing reductions";
        TF_CHECK_OK(EmitAtomicOperationForNestedComputation(
            builder, reduction_emitter_.ir_emitter_context_, *reducer,
            output_address, output_ptr, type));
      }
    }
  }
}

void ReductionGroupEmitter::EmitReductionOutputForRowReduction(
    const TilingKernelInfo& tiling_kernel_info,
    const HloReduceInstruction* reduction,
    const std::vector<const HloInstruction*>& roots) const {
  const HloComputation* reducer = reduction->to_apply();
  const auto& thread_id_info = tiling_kernel_info.thread_id_info;
  const auto& thread_ids = thread_id_info.thread_ids;
  auto* thread_id_x =
      thread_ids[ReductionDimensions::kRowMinorReducedDimension];
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(reduction_emitter_.index_ty_, c);
  };

  auto* builder = reduction_emitter_.builder_;
  auto is_zero = [&](llvm::Value* value) {
    return builder->CreateICmpEQ(value, constant(0));
  };

  int num_outputs = reducer->num_parameters() / 2;
  absl::InlinedVector<TypedPointer, 2> current_outputs;
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const auto& state = GetCalculationStateFor(reduction, output_idx);
    current_outputs.push_back(
        {state.partial_result_address,
         state.partial_result_address->getAllocatedType()});
  }

  const auto& reduction_info = reduction_emitter_.reduction_codegen_info_;
  const Tiling& tiling = reduction_info.GetTiling();
  int num_rows_per_warp =
      RowReductionGetRowsPerWarp(reduction_emitter_.ReducedDimensionSize());
  EmitFullWarpShuffleDownLoopForReduce(reducer, absl::MakeSpan(current_outputs),
                                       tiling.GetNumThreadsPerBlock(),
                                       num_rows_per_warp);

  KernelSupportLibrary ksl(builder);
  llvm::Value* warp_id = builder->CreateUDiv(thread_id_x, constant(WarpSize()));

  auto emit_write_output = [&](llvm::Value* write_condition,
                               const absl::Span<TypedPointer const> values) {
    ksl.If("reduction_write_output", write_condition, [&] {
      WriteReductionOutput(tiling_kernel_info, reduction, roots, values);
    });
  };

  // The major kept dimension and vector dimension are not tiled, so they're
  // always in bounds.
  llvm::Value* is_in_bounds_y = builder->CreateICmpULT(
      thread_ids[ReductionDimensions::kRowKeptDimension],
      tiling_kernel_info
          .output_tile_bounds[ReductionDimensions::kRowKeptDimension]);

  ksl.If("thread_in_bounds", is_in_bounds_y, [&] {
    if (num_rows_per_warp > 1) {
      llvm::Value* is_writing_thread = is_zero(builder->CreateAnd(
          thread_id_x,
          constant(reduction_emitter_.ReducedDimensionSize() - 1)));
      emit_write_output(is_writing_thread, current_outputs);
      return;
    }

    ksl.If("intra_warp_reduce_write", is_zero(thread_id_info.lane_id), [&] {
      for (int oidx = 0; oidx < num_outputs; oidx++) {
        auto& state = GetCalculationStateFor(reduction, oidx);
        state.shared_cache->Store(
            builder->CreateLoad(current_outputs[oidx].second,
                                current_outputs[oidx].first),
            {thread_id_info.thread_ids[ReductionDimensions::kRowKeptDimension],
             warp_id},
            builder);
      }
    });

    // TODO(cheshire): Don't we want to sync it once for everything in the
    // output? Not once per each?
    reduction_emitter_.EmitSyncThreads();
    ksl.If("inter_warp_reduce", is_zero(warp_id), [&] {
      absl::InlinedVector<TypedPointer, 2> selected_values;
      for (int oidx = 0; oidx < num_outputs; oidx++) {
        auto& state = GetCalculationStateFor(reduction, oidx);
        llvm::Value* block_accum_addr = state.shared_cache->Address(
            {thread_id_info.thread_ids[ReductionDimensions::kRowKeptDimension],
             thread_id_info.lane_id},
            builder);

        llvm::Type* element_type =
            state.partial_result_address->getAllocatedType();

        // Ensure initial value address is in generic, not scratch.
        llvm::Value* initial_value_addr =
            CastSharedToGlobal(builder,
                               llvm_ir::EmitAllocaAtFunctionEntry(
                                   element_type, "initial_value_addr", builder),
                               element_type, /*name=*/"");
        builder->CreateStore(state.initial_value, initial_value_addr);

        llvm::Value* warp_exists = builder->CreateICmpULT(
            thread_id_x,
            constant(tiling.GetThreadsPerBlock()
                         [ReductionDimensions::kRowMinorReducedDimension] /
                     WarpSize()));

        llvm::Value* selected_value = builder->CreateSelect(
            warp_exists, block_accum_addr, initial_value_addr);

        selected_values.push_back({selected_value, element_type});
      }

      // If only one warp produces the output element, we don't need to emit
      // an inter warp reduce. In our tiling, DimX is the minor reduced
      // dimension. The major reduced dimension is always emitted as a loop.
      // TODO(b/241414088) If only warp is present, then inter-warp
      // communication using shared memory and synchronization using barrier is
      // also unnecessary and should be removed.
      if (tiling.GetThreadsPerBlock()
              [ReductionDimensions::kRowMinorReducedDimension] > WarpSize()) {
        EmitFullWarpShuffleDownLoopForReduce(
            reducer, absl::MakeSpan(selected_values),
            tiling.GetNumThreadsPerBlock(), /*num_results_per_warp=*/1);
      }

      emit_write_output(is_zero(thread_id_x), selected_values);
    });
  });
}

// Same arguments as EmitReductionOutputForRowReduction.
void ReductionGroupEmitter::EmitReductionOutputForColumnReduction(
    const TilingKernelInfo& tiling_kernel_info,
    const HloReduceInstruction* reduction,
    const std::vector<const HloInstruction*>& roots) const {
  auto* builder = reduction_emitter_.builder_;
  KernelSupportLibrary ksl(builder);
  const HloComputation* reducer = reduction->to_apply();
  const auto& thread_id_info = tiling_kernel_info.thread_id_info;
  const auto& thread_ids = thread_id_info.thread_ids;

  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(reduction_emitter_.index_ty_, c);
  };
  auto is_zero = [&](llvm::Value* value) {
    return builder->CreateICmpEQ(value, constant(0));
  };
  const auto& reduction_info = reduction_emitter_.reduction_codegen_info_;
  const Tiling& tiling = reduction_info.GetTiling();
  int num_outputs = reducer->num_parameters() / 2;

  auto* kept_index = thread_ids[ReductionDimensions::kColMinorKeptDimension];
  auto* reduced_index = thread_ids[ReductionDimensions::kColReducedDimension];

  // Store the transpose in shared memory.
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const auto& state = GetCalculationStateFor(reduction, output_idx);
    auto* current_output_value =
        builder->CreateLoad(state.partial_result_address->getAllocatedType(),
                            state.partial_result_address);
    state.shared_cache->Store(current_output_value, {kept_index, reduced_index},
                              builder);
  }

  reduction_emitter_.EmitSyncThreads();

  // Get transposed element from shared memory.
  absl::InlinedVector<TypedPointer, 2> shmem_transposed_addrs;
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const auto& state = GetCalculationStateFor(reduction, output_idx);
    auto* shmem_transposed_addr =
        state.shared_cache->Address({reduced_index, kept_index}, builder);
    shmem_transposed_addrs.push_back(
        {shmem_transposed_addr, state.shared_cache->GetElementType()});
  }

  EmitFullWarpShuffleDownLoopForReduce(reducer,
                                       absl::MakeSpan(shmem_transposed_addrs),
                                       tiling.GetNumThreadsPerBlock(),
                                       /*num_results_per_warp=*/1);

  // Some warps in the block are completely outside of the bound of the
  // tensor, so they should not write any output at all.
  llvm::Value* has_output = builder->CreateAnd(
      builder->CreateICmpULT(
          reduced_index,
          tiling_kernel_info
              .output_tile_bounds[ReductionDimensions::kColMinorKeptDimension]),
      builder->CreateICmpULT(
          kept_index,
          tiling_kernel_info
              .output_tile_bounds[ReductionDimensions::kColReducedDimension]));

  ksl.If("reduction_write_output",
         builder->CreateAnd(has_output, is_zero(thread_id_info.lane_id)), [&] {
           WriteReductionOutput(tiling_kernel_info, reduction, roots,
                                shmem_transposed_addrs);
         });
}

// Generate a single element of the tile (update the accumulator state) for a
// given reducer.
void ReductionGroupEmitter::GenerateElementForReducer(
    const HloReduceInstruction* reduction,
    const llvm_ir::IrArray::Index& index) const {
  HloComputation* reducer = reduction->to_apply();
  auto* builder = reduction_emitter_.builder_;
  CHECK_EQ(reducer->num_parameters() % 2, 0);

  absl::InlinedVector<llvm::Value*, 2> reduction_accumulators;
  absl::InlinedVector<llvm::Value*, 2> reduction_input_value;
  for (int red_idx = 0; red_idx < reducer->num_parameters() / 2; red_idx++) {
    const auto& state = GetCalculationStateFor(reduction, red_idx);

    llvm::AllocaInst* input_address = state.input_address;
    auto input_index =
        index.SourceIndexOfBitcast(reduction->operand(0)->shape(), builder);
    llvm::Value* const input_ir_value = *state.input_gen(input_index);
    builder->CreateStore(input_ir_value, input_address);
    reduction_accumulators.push_back(state.partial_result_address);
    reduction_input_value.push_back(input_address);
  }

  absl::InlinedVector<llvm::Value*, 4> reduction_params;
  for (llvm::Value* acc : reduction_accumulators) {
    reduction_params.push_back(acc);
  }
  for (llvm::Value* value : reduction_input_value) {
    reduction_params.push_back(value);
  }

  // Emit a call to the variadic reducer. Since it may be returning a
  // tuple, we can't return it directly as a value. Instead, before
  // the call, we create N (N = # arguments in the tuple) allocas, one
  // for each returned argument, then when we make the call we pass N
  // pointers as last parameters, the called computation writes into
  // those pointers, and we have returned values on the stack (as well
  // as pointers to them).
  absl::StatusOr<std::vector<llvm::Value*>> returned_scalars =
      CallNestedComputationWithScalarAddrs(
          builder, reduction_emitter_.ir_emitter_context_, *reducer,
          reduction_params);
  TF_CHECK_OK(returned_scalars.status());

  for (int i = 0; i < returned_scalars->size(); i++) {
    builder->CreateStore(returned_scalars->at(i), reduction_accumulators[i]);
  }
}

// Emits code for reductions in the output_instructions.
absl::Status ReductionEmitter::EmitIRForReduction(
    absl::Span<const HloInstruction* const> instr_index_group,
    FusedIrEmitter& fused_emitter, const ReductionOutputMap& result_ir_arrays,
    const Shape& input_shape) {
  ExtraOutputGensMap extra_output_gens;
  absl::flat_hash_map<const HloReduceInstruction*,
                      std::vector<const HloInstruction*>>
      heroes_to_roots;
  // Keep a list of deduplicated heroes separate from heroes_to_roots to make
  // the CodeGen deterministic.
  std::vector<const HloReduceInstruction*> heroes;

  for (const HloInstruction* hlo : instr_index_group) {
    auto& hero = FindNonTrivialHero(*hlo);
    if (IsRealReductionHero(*hlo, hero)) {
      auto reduction = Cast<HloReduceInstruction>(&hero);
      if (heroes_to_roots.find(reduction) == heroes_to_roots.end()) {
        heroes.push_back(reduction);
      }
      heroes_to_roots[reduction].push_back(hlo);
    } else {
      extra_output_gens[hlo] = *fused_emitter.GetGenerator(*hlo);
    }
  }

  CHECK(!heroes.empty()) << " expect at least one reduce instructions.";
  const Tiling& tiling = reduction_codegen_info_.GetTiling();
  CHECK_EQ(tiling.GetNumThreadsPerBlock() % WarpSize(), 0);
  ReductionGroupEmitter group_emitter(*this, heroes, result_ir_arrays,
                                      fused_emitter);

  TF_ASSIGN_OR_RETURN(
      TilingKernelInfo tiling_kernel_info,
      EmitTilingKernel(
          builder_, tiling, index_ty_,
          [&](const TilingThreadIdInfo& thread_id_info,
              const llvm_ir::IrArray::Index& tile_index,
              absl::Span<llvm::Value* const> tile_dimensions) {
            auto emit_element =
                [&](absl::Span<llvm::Value* const> index_in_tile) {
                  auto index = tile_index.AddOffset(index_in_tile, builder_);

                  // Emit code to generate the input and perform the reduction
                  // computation for each reduction instruction.
                  for (const HloReduceInstruction* reduce : heroes) {
                    group_emitter.GenerateElementForReducer(reduce, index);
                  }

                  // Emit code to generate the output for the non-reduction
                  // instructions in the fusion, if any.
                  TF_CHECK_OK(group_emitter.EmitExtraOutputsForReduce(
                      ShapeUtil::MakeShape(
                          F32, reduction_codegen_info_.GetTiling().GetShape()),
                      index, extra_output_gens));
                };
            EmitTile(builder_, reduction_codegen_info_.GetTiling(),
                     thread_id_info, tile_dimensions, emit_element);
          }));

  KernelSupportLibrary ksl(builder_);
  for (auto reduce : heroes) {
    if (reduction_codegen_info_.IsRowReduction()) {
      group_emitter.EmitReductionOutputForRowReduction(
          tiling_kernel_info, reduce, heroes_to_roots[reduce]);
    } else {
      group_emitter.EmitReductionOutputForColumnReduction(
          tiling_kernel_info, reduce, heroes_to_roots[reduce]);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<FusionEmissionResult> ReductionEmitter::EmitInitializers() {
  FusionEmissionResult result;
  if (reduction_codegen_info_.IsRaceFree()) {
    return result;
  }
  // We need to get the dest slice by traversing the slice assigned to
  // fusion, because instructions inside fusion don't have buffer assignment.
  //
  // The order of fusion roots is determined by its position in the result
  // tuple. For example, in the following fused computation
  //
  // %fused_computation {
  //   %a = ...
  //   &b = ...
  //   ROOT %root = tuple(%a, %b)
  // }
  //
  // The fusion root with index = 0 is %a, and the fusion root %b has index 1.
  // Therefore we can get the ordered slices by calling ForEachSubshape on the
  // result shape.
  std::vector<BufferAllocation::Slice> slices;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      fusion_.shape(), [&](const Shape& subshape, ShapeIndex index) {
        if (!ShapeUtil::IsLeafIndex(fusion_.shape(), index)) {
          return absl::OkStatus();
        }

        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice slice,
            ir_emitter_context_.buffer_assignment().GetUniqueSlice(&fusion_,
                                                                   index));
        slices.push_back(slice);
        return absl::OkStatus();
      }));

  absl::Span<HloInstructionAdaptor const> fusion_roots =
      analysis_.fusion_roots();
  for (int i = 0; i < fusion_roots.size(); ++i) {
    const HloInstruction* fusion_root = &fusion_roots[i].instruction();

    if (IsReductionFromOrToContiguousDimensions(*fusion_root)) {
      TF_ASSIGN_OR_RETURN(
          result.thunks.emplace_back(),
          BuildFusedInitializerThunk(fusion_root, slices[i], i));
    }
  }
  return result;
}

absl::Status ReductionEmitter::EmitKernel(
    const LaunchDimensions& launch_dims, std::vector<llvm_ir::IrArray> inputs,
    std::vector<llvm_ir::IrArray> outputs) {
  const HloComputation* fused_computation =
      fusion_.fused_instructions_computation();
  FusedIrEmitter fused_emitter(elemental_emitter_);
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    HloInstruction* fused_operand = fused_computation->parameter_instruction(i);
    fused_emitter.BindGenerator(
        *fused_operand, [builder = builder_, input = inputs[i],
                         fused_operand](const llvm_ir::IrArray::Index& index) {
          return input.EmitReadArrayElement(index, builder,
                                            fused_operand->name());
        });
  }

  // Get outputs.
  ReductionOutputMap result_ir_arrays;

  int ir_arrays_idx = 0;
  for (const HloInstructionAdaptor& root : analysis_.fusion_roots()) {
    int get_num_results = GetNumOutputs(root.shape());
    result_ir_arrays[&root.instruction()] =
        absl::MakeSpan(outputs).subspan(ir_arrays_idx, get_num_results);
    ir_arrays_idx += get_num_results;
  }

  KernelSupportLibrary ksl(builder_, llvm_ir::UnrollMode::kDefaultUnroll);

  // Use raw block_id_y to select the i-th parallel reduction to run. Using
  // block_id_y instead of block_id_x simplifies the index calculation
  // for reduction code generation as the block_id_y is orthogonal to
  // the indices used within the reductions.
  const auto& instr_index_groups =
      reduction_codegen_info_.GetGroups().grouped_roots;
  Shape reduce_operand_shape = reduction_codegen_info_.GetReduceOperandShape();

  llvm::Value* block_id_y = gpu::EmitCallToTargetIntrinsic(
      gpu::TargetIntrinsicID::kBlockIdy, {}, {}, builder_);
  llvm_ir::AddRangeMetadata(0, instr_index_groups.size(),
                            llvm::cast<llvm::Instruction>(block_id_y),
                            builder_->GetInsertBlock()->getModule());
  block_id_y = builder_->CreateZExtOrTrunc(block_id_y, builder_->getInt32Ty());
  block_id_y->setName("block.id.y");
  for (int i = 0; i < instr_index_groups.size(); ++i) {
    TF_RETURN_IF_ERROR(ksl.IfWithStatus(
        absl::StrCat("reduce-group-", i),
        builder_->CreateICmpEQ(block_id_y, builder_->getInt32(i)), [&] {
          return EmitIRForReduction(instr_index_groups[i], fused_emitter,
                                    result_ir_arrays, reduce_operand_shape);
        }));
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<FusionEmissionResult> ReductionFusion::EmitInitializers(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  llvm::IRBuilder<> builder(ir_emitter_context.llvm_module()->getContext());
  return ReductionEmitter(analysis_, reduction_info_, ir_emitter_context,
                          fusion, &builder)
      .EmitInitializers();
}

absl::Status ReductionFusion::EmitKernel(IrEmitterContext& ir_emitter_context,
                                         const HloFusionInstruction& fusion,
                                         const LaunchDimensions& launch_dims,
                                         std::vector<llvm_ir::IrArray> inputs,
                                         std::vector<llvm_ir::IrArray> outputs,
                                         llvm::IRBuilder<>* builder) const {
  return ReductionEmitter(analysis_, reduction_info_, ir_emitter_context,
                          fusion, builder)
      .EmitKernel(launch_dims, inputs, outputs);
}

int ReductionInfo::GetRowsPerWarp() const {
  if (!is_row_reduction_) return 1;
  return RowReductionGetRowsPerWarp(
      tiling_.GetShape()[ReductionDimensions::kRowMinorReducedDimension]);
}

LaunchDimensions ReductionInfo::launch_dimensions() const {
  size_t blocks_y = groups_.grouped_roots.size();
  return {se::BlockDim(/*x=*/tiling_.GetNumBlocks(),
                       /*y=*/static_cast<int64_t>(blocks_y), /*z=*/1),
          se::ThreadDim(/*x=*/tiling_.GetNumThreadsPerBlock(),
                        /*y=*/1, /*z=*/1)};
}

ReductionInfo ReductionInfo::Create(const HloFusionAnalysis& analysis) {
  auto* hero_reduction = analysis.FindHeroReduction();
  CHECK_NE(hero_reduction, nullptr);
  Shape input_shape = hero_reduction->operand(0)->shape();
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  auto shape = reduction_dimensions.dimensions;
  VLOG(10) << "is_row_reduction " << reduction_dimensions.is_row_reduction
           << " " << shape[0] << " " << shape[1] << " " << shape[2];
  Vector3 reduction_tiling = GetReductionTiling(reduction_dimensions);

  int64_t num_threads_y =
      reduction_dimensions.is_row_reduction ? 1 : WarpSize();
  int64_t rows_per_warp =
      reduction_dimensions.is_row_reduction
          ? RowReductionGetRowsPerWarp(
                shape[ReductionDimensions::kRowMinorReducedDimension])
          : 1;
  int64_t num_threads_x = [&] {
    if (reduction_dimensions.is_row_reduction) {
      if (rows_per_warp > 1) {
        return shape[ReductionDimensions::kRowMinorReducedDimension];
      }
      int64_t max_block_size =
          MinThreadsXRowReduction(hero_reduction->GetModule()->config());
      return std::min(
          max_block_size,
          RoundUpTo(
              CeilOfRatio(shape[ReductionDimensions::kRowMinorReducedDimension],
                          reduction_tiling
                              [ReductionDimensions::kRowMinorReducedDimension]),
              WarpSize()));
    }
    return WarpSize();
  }();

  // If we're limited by the size of the x dimension, add additional parallelism
  // in the y dimension. The code generator doesn't currently support
  // parallelizing the z dimension (major reduced dimensions). The general
  // recommendation is to use between 128 and 512 threads, so we just go for
  // 256. See https://forums.developer.nvidia.com/t/55529
  constexpr int64_t kThreadsPerBlockTarget = 256;
  if (reduction_dimensions.is_row_reduction &&
      num_threads_x * 2 <= kThreadsPerBlockTarget) {
    int64_t kept_size =
        reduction_dimensions.dimensions[ReductionDimensions::kRowKeptDimension];
    // Increase the size of the y dimension as long as there's remaining
    // parallelism.
    if (kept_size * num_threads_x <= kThreadsPerBlockTarget) {
      num_threads_y = kept_size;
      // num_threads_x is a power of two, but it may be less than 32. If dim_y
      // is also small, we may have to increase the bound so the total number of
      // threads is a multiple of 32.
      while ((num_threads_x * num_threads_y) % 32) ++num_threads_y;
    } else {
      num_threads_y = kThreadsPerBlockTarget / num_threads_x;
    }
  }

  int vector_size = GetVectorSize(analysis, reduction_dimensions, num_threads_x,
                                  reduction_tiling);

  absl::InlinedVector<int64_t, 4> num_threads{1, num_threads_y, num_threads_x};
  absl::InlinedVector<int64_t, 4> tiled_shape{shape[0], shape[1],
                                              shape[2] / vector_size};
  absl::InlinedVector<int64_t, 4> tile_per_thread{
      reduction_tiling[0], reduction_tiling[1],
      std::max<int64_t>(reduction_tiling[2] / vector_size, 1)};
  if (rows_per_warp > 1) {
    // If we produce more than one element per thread, that means the reduced
    // dimension is small and it can't be tiled - we already have more threads
    // in a warp than the size of the reduced dimension. The code generator
    // doesn't currently support tiling the kept dimension, because it just
    // uses the thread ID as the coordinate.
    tile_per_thread[2] = 1;
  }
  if (vector_size != 1) {
    num_threads.push_back(1);  // The vector dimension is a loop.
    tiled_shape.push_back(vector_size);
    tile_per_thread.push_back(vector_size);
  }

  Tiling tiling(tiled_shape, tile_per_thread, num_threads,
                /*loops_to_unroll=*/{false, false, true, false});
  bool reduction_is_race_free = ReductionIsRaceFree(
      hero_reduction->GetModule()->config(), reduction_dimensions);
  return ReductionInfo(analysis, tiling, reduction_dimensions.is_row_reduction,
                       reduction_is_race_free,
                       GroupDisjointReductions(analysis, /*for_mlir=*/false),
                       hero_reduction);
}

std::optional<IndexingMap> ReductionInfo::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  if (!groups_.is_reduction_root[root_index]) {
    auto map = ComposeIndexingMaps(
        GetIndexingMapForTiling(tiling_, ctx),
        GetBitcastMap(tiling_.GetXlaShape(),
                      analysis_.fusion_root(root_index).shape(), ctx));
    AddGroupIdConstraint(map, root_index, groups_);
    return map;
  }
  const auto& hero = analysis_.fusion_hero(root_index).instruction();

  auto block_offsets = GetBlockOffsetsForTiling(tiling_, ctx);
  auto thread_ids = DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx),
                                             tiling_.GetThreadsPerBlock());

  auto physical_shape =
      ShapeUtil::DeleteDimensions(hero.dimensions(), hero.operand(0)->shape());
  std::vector<DimVar> dimension_ranges{
      {{0, tiling_.GetNumThreadsPerBlock() - 1}},
      {},
      {},
      {{0, tiling_.GetNumBlocks() - 1}},
      {{0, static_cast<int64_t>(groups_.grouped_roots.size() - 1)}},
      {},
  };

  constexpr int kRowKept = ReductionDimensions::kRowKeptDimension;
  constexpr int kRowMinorReduced =
      ReductionDimensions::kRowMinorReducedDimension;

  constexpr int kColMajorKept = ReductionDimensions::kColMajorKeptDimension;
  constexpr int kColMinorKept = ReductionDimensions::kColMinorKeptDimension;
  constexpr int kColReduced = ReductionDimensions::kColReducedDimension;

  auto map = [&]() {
    if (is_row_reduction_) {
      IndexingMap linear_index(
          mlir::AffineMap::get(
              6, 0, block_offsets.getResult(kRowKept) + thread_ids[kRowKept],
              ctx),
          dimension_ranges, /*range_vars=*/{}, /*rt_vars=*/{});
      int rows_per_warp = GetRowsPerWarp();
      if (rows_per_warp > 1) {
        linear_index.AddConstraint(
            thread_ids[kRowMinorReduced] % (WarpSize() / rows_per_warp),
            {0, 0});
      } else {
        linear_index.AddConstraint(thread_ids[kRowMinorReduced], {0, 0});
      }
      return ComposeIndexingMaps(
          linear_index, GetBitcastMap(ShapeUtil::MakeShape(
                                          PRED, {tiling_.GetShape()[kRowKept]}),
                                      physical_shape, ctx));
    }

    mlir::SmallVector<mlir::AffineExpr> projected_dims{
        block_offsets.getResult(kColMajorKept),
        block_offsets.getResult(kColMinorKept) + thread_ids[kColReduced]};
    std::vector<RangeVar> range_vars;
    if (thread_ids.size() == 4) {
      int vector_size = tiling_.GetThreadTileSize().back();
      range_vars.push_back({0, vector_size - 1});
      projected_dims.push_back(mlir::getAffineSymbolExpr(0, ctx));
    }
    IndexingMap projected_index(
        mlir::AffineMap::get(6, range_vars.size(), projected_dims, ctx),
        dimension_ranges, range_vars, /*rt_vars=*/{});

    projected_index.AddConstraint(
        mlir::getAffineDimExpr(
            KernelFusionInterface::kIndexingMapThreadIdxDims[0], ctx) %
            WarpSize(),
        {0, 0});
    if (!is_row_reduction_) {
      projected_index.AddConstraint(
          projected_index.GetAffineMap().getResult(1),
          {0, tiling_.GetShape()[ReductionDimensions::kColMinorKeptDimension] -
                  1});
    }

    return ComposeIndexingMaps(
        projected_index,
        GetBitcastMap(ShapeUtil::DeleteDimension(
                          ReductionDimensions::kColReducedDimension,
                          tiling_.GetXlaShape()),
                      physical_shape, ctx));
  }();

  AddGroupIdConstraint(map, root_index, groups_);
  return map;
}

std::optional<IndexingMap> ReductionInfo::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  if (groups_.is_reduction_root[root_index] &&
      hero_operand_index >= hero.operand_count() / 2) {
    // We don't have indexing for the init values.
    return std::nullopt;
  }
  if (!groups_.is_reduction_root[root_index]) {
    return ComposeIndexingMaps(
        *ComputeThreadIdToOutputIndexing(root_index, ctx),
        *ComputeOutputToInputIndexing(
             &analysis_.fusion_root(root_index).instruction(), 0, ctx)
             .indexing_maps[hero_operand_index]
             .begin());
  }

  auto map = ComposeIndexingMaps(
      GetIndexingMapForTiling(tiling_, ctx),
      GetBitcastMap(tiling_.GetXlaShape(),
                    hero.operand(hero_operand_index)->shape(), ctx));
  AddGroupIdConstraint(map, root_index, groups_);
  return map;
}

}  // namespace gpu
}  // namespace xla
