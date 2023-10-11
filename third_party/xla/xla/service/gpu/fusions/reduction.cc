/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
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
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/thunk_util.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_nested.h"
#include "xla/service/gpu/kernel_mapping_scheme.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/kernel_thunk.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/translate/mhlo_to_hlo/location_exporter.h"
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

// For a row reduction, returns the number of rows we can process in parallel
// per warp.
int RowReductionGetRowsPerWarp(int reduced_dimension_size) {
  if (WarpSize() % reduced_dimension_size != 0 ||
      reduced_dimension_size >= WarpSize()) {
    return 1;
  }
  return WarpSize() / reduced_dimension_size;
}

int GetNumOutputs(const Shape& shape) {
  if (shape.IsTuple()) {
    return shape.tuple_shapes_size();
  }
  return 1;
}

// Allocates a shared tile of given dimensions, applying scaling specified in
// tilng_scheme as a major-most dimension to avoid collisions.
llvm::GlobalVariable* AllocateShared(
    llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
    llvm::Type* element_type,
    absl::Span<int64_t const> dimensions_major_to_minor,
    absl::string_view buffer_name) {
  CHECK(!dimensions_major_to_minor.empty());
  llvm::Type* ty = element_type;
  for (auto dim : llvm::reverse(dimensions_major_to_minor)) {
    ty = llvm::ArrayType::get(ty, dim);
  }
  ty = llvm::ArrayType::get(ty, tiling_scheme.GetThreadIdScalingFactor());
  return llvm_ir::AllocateSharedMemoryTile(
      builder->GetInsertBlock()->getModule(), ty, buffer_name);
}

// Creates accumulator alloca's, populates them with initial values, generates
// __shared__ caches and returns the populated object.
ReductionCodegenState GenerateReductionCodegenState(
    llvm::IRBuilder<>* builder, mlir::lmhlo::FusionOp fusion,
    const ReductionCodegenInfo& reduction_info,
    absl::Span<const HloReduceInstruction* const> reduce_instr_index_group,
    FusedIrEmitter& fused_emitter) {
  ReductionCodegenState reduction_codegen_state(reduction_info);
  VLOG(10) << "Emit prologue for reduction: " << llvm_ir::DumpToString(fusion);

  for (const HloReduceInstruction* reduce_hlo : reduce_instr_index_group) {
    int num_partial_results = reduction_codegen_state.GetNumPartialResults();
    for (int op_result_idx = 0;
         op_result_idx < GetNumOutputs(reduce_hlo->shape()); op_result_idx++) {
      Shape result_shape = reduce_hlo->shape().IsTuple()
                               ? reduce_hlo->shape().tuple_shapes(op_result_idx)
                               : reduce_hlo->shape();

      llvm::Type* element_type = llvm_ir::PrimitiveTypeToIrType(
          result_shape.element_type(), builder->GetInsertBlock()->getModule());
      llvm::AllocaInst* reduction_input_address =
          llvm_ir::EmitAllocaAtFunctionEntry(
              element_type, "reduction_input_address", builder);

      llvm::AllocaInst* partial_result_address =
          llvm_ir::EmitAllocaAtFunctionEntryWithCount(
              element_type,
              /*element_count=*/builder->getInt32(num_partial_results),
              "partial_reduction_result", builder);

      const HloInstruction* init_value =
          reduce_hlo->init_values()[op_result_idx];

      // Initialize the partial result with the initial value of the reduction.
      llvm::Value* init_ir_value = (*fused_emitter.GetGenerator(
          *init_value))(llvm_ir::IrArray::Index(builder->getInt32Ty()))
                                       .value();

      for (int i = 0; i < num_partial_results; ++i) {
        builder->CreateStore(
            init_ir_value, builder->CreateInBoundsGEP(
                               partial_result_address->getAllocatedType(),
                               partial_result_address, {builder->getInt32(i)}));
      }

      const TilingScheme& tiling_scheme =
          reduction_codegen_state.GetTilingScheme();
      int64_t num_threads_x =
          tiling_scheme.GetNumThreadsFor(TilingScheme::DimX);
      llvm::GlobalVariable* shared_cache = [&]() -> llvm::GlobalVariable* {
        if (reduction_codegen_state.IsRowReduction()) {
          // Multi-row reductions do not use shared memory.
          if (RowReductionGetRowsPerWarp(tiling_scheme.GetDimsInElems()[2]) >
              1) {
            return nullptr;
          }
          // Allocate __shared__
          // cache[num_partial_results][num_warps][scaling_factor].
          CHECK_EQ(tiling_scheme.GetNumThreadsPerBlock() % WarpSize(), 0);
          int num_warps = tiling_scheme.GetNumThreadsPerBlock() / WarpSize();
          return AllocateShared(builder, tiling_scheme, element_type,
                                {num_partial_results, num_warps},
                                "shared_cache");
        } else {
          // Allocate __shared__
          // cache[num_threads][num_threads + 1], where
          // num_threads == num_threads_x == num_threads_y.  The "+1" is used to
          // avoid bank conflicts.
          //
          // (Although each thread produces num_partial_results results, we
          // don't need that much cache: Only one result is live at a time.)
          CHECK_EQ(num_threads_x,
                   tiling_scheme.GetNumThreadsFor(TilingScheme::DimY));
          return AllocateShared(builder, tiling_scheme, element_type,
                                {num_threads_x, num_threads_x + 1},
                                "shared_cache");
        }
      }();

      llvm_ir::ElementGenerator input_gen =
          *fused_emitter.GetGenerator(*reduce_hlo->inputs()[op_result_idx]);
      reduction_codegen_state.SetCalculationStateFor(
          {shared_cache, init_ir_value, partial_result_address,
           reduction_input_address, input_gen},
          reduce_hlo, op_result_idx);
    }
  }

  return reduction_codegen_state;
}

void MaybeEmitFenceForAMDGPU(llvm::IRBuilder<>* builder,
                             IrEmitterContext& ir_emitter_context) {
  auto* module = builder->GetInsertBlock()->getModule();
  if (IsAMDGPU(module) &&
      ir_emitter_context.rocm_compute_capability().gcn_arch_name().substr(
          0, 6) == "gfx90a") {
    builder->CreateFence(
        llvm::AtomicOrdering::SequentiallyConsistent,
        builder->getContext().getOrInsertSyncScopeID("workgroup"));
  }
}

void EmitSyncThreads(llvm::IRBuilder<>* builder,
                     IrEmitterContext& ir_emitter_context) {
  MaybeEmitFenceForAMDGPU(builder, ir_emitter_context);
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kBarrierId, {}, {}, builder);
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
//   BuildKernelThunkForFusion(..., fusion_op, launch_dimensions, builder_fn,
//                             ...));
// AddThunkToThunkSequence(std::move(thunk))
// ```
StatusOr<std::unique_ptr<Thunk>> BuildKernelThunkForFusion(
    IrEmitterContext& ir_emitter_context, KernelReuseCache& kernel_cache,
    mlir::lmhlo::FusionOp fusion_op, const HloComputation* fused_computation,
    const LaunchDimensions& launch_dimensions, absl::string_view discriminator,
    std::function<Status(std::vector<llvm_ir::IrArray>,
                         std::vector<llvm_ir::IrArray>)>
        kernel_builder_fn,
    llvm::IRBuilder<>* builder) {
  std::string suggested_kernel_name = GetIrNameFromLoc(fusion_op->getLoc());

  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context.allocations(), fusion_op));

  auto kernel_builder_status = OkStatus();
  auto [entry, cached] = kernel_cache.Get(
      fused_computation, kernel_arguments.args(), discriminator,
      [&]() -> KernelReuseCache::Entry {
        auto [kernel, input_arrays, output_arrays] = BuildKernelPrototype(
            ir_emitter_context, suggested_kernel_name, kernel_arguments.args(),
            fusion_op.getInputBuffers().size(), launch_dimensions, builder);
        kernel_builder_status = kernel_builder_fn(input_arrays, output_arrays);
        return {kernel->getName().str(), launch_dimensions};
      });
  TF_RETURN_IF_ERROR(kernel_builder_status);
  if (cached) {
    VLOG(3) << "Reuse: " << suggested_kernel_name << " -> "
            << entry.kernel_name;
  }

  return std::make_unique<KernelThunk>(
      fusion_op, entry.kernel_name, kernel_arguments.args(), launch_dimensions,
      // Shared memory is allocated statically.
      /*shmem_bytes=*/0);
}

Status EmitExtraOutputsForReduce(llvm::IRBuilder<>* builder,
                                 const Shape& reduction_operand_shape,
                                 const ReductionOutputMap& result_ir_arrays,
                                 const llvm_ir::IrArray::Index& index,
                                 const ReductionCodegenInfo& reduction_info,
                                 const ExtraOutputGensMap& extra_output_gens) {
  if (extra_output_gens.empty()) {
    return OkStatus();
  }

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
    absl::Span<llvm_ir::IrArray const> result_ir = result_ir_arrays.at(instr);
    CHECK_EQ(result_ir.size(), 1);
    result_ir[0].EmitWriteArrayElement(
        get_index(instr), generator, builder, /*use_linear_index=*/
        reduction_info.GetNumPartialResults() == 1);
  }
  return OkStatus();
}

StatusOr<std::unique_ptr<Thunk>> BuildFusedInitializerThunk(
    IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion,
    const HloComputation* fused_computation,
    ElementalIrEmitter& elemental_emitter, KernelReuseCache& kernel_cache,
    int output_index, llvm::IRBuilder<>* builder) {
  auto reduce = mlir::dyn_cast_or_null<mlir::mhlo::ReduceOp>(
      fusion.getFusionRoots()[output_index]);

  TF_RET_CHECK(reduce);
  TF_RET_CHECK(reduce.getNumResults() == 1);

  mlir::Value init_value = reduce.getInitValues()[0];
  mlir::Value dest = fusion.getOutputBuffers()[output_index];
  TF_ASSIGN_OR_RETURN(std::optional<std::unique_ptr<Thunk>> constant_init_thunk,
                      BuildConstantInitializerThunk(ir_emitter_context, fusion,
                                                    init_value, dest));
  if (constant_init_thunk) {
    return *std::move(constant_init_thunk);
  }

  const Shape dest_shape = GetShape(dest);

  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          dest_shape, ir_emitter_context.gpu_device_info()));

  auto builder_fn = [&](std::vector<llvm_ir::IrArray> inputs,
                        std::vector<llvm_ir::IrArray> outputs) -> Status {
    FusedIrEmitter fused_emitter(elemental_emitter);
    for (int i = 0; i < fused_computation->num_parameters(); i++) {
      fused_emitter.BindGenerator(
          *fused_computation->parameter_instruction(i),
          [builder, input = inputs[i]](llvm_ir::IrArray::Index index) {
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
                                           launch_dimensions, builder)
                           .EmitLoop(GetIrNameFromLoc(fusion.getLoc())));
    return OkStatus();
  };

  return BuildKernelThunkForFusion(ir_emitter_context, kernel_cache, fusion,
                                   fused_computation, launch_dimensions,
                                   /*discriminator=*/
                                   absl::StrCat("init_", output_index),
                                   builder_fn, builder);
}

// Gets the output offset as calculated from thread_id.x (to be applied to the
// offset calculated from block_id and thread_id.y).
static llvm::Value* GetStartOffsetX(const TilingScheme& tiling_scheme,
                                    llvm::Value* thread_id_x,
                                    llvm::Type* index_ty,
                                    llvm::IRBuilder<>* b) {
  int64_t multiplier =
      tiling_scheme.GetIndexingOrder() == TilingScheme::StridedIndexingX
          ? tiling_scheme.GetVectorSize()
          : tiling_scheme.GetTileSizeFor(TilingScheme::DimX);
  return b->CreateMul(thread_id_x,
                      llvm::ConstantInt::get(index_ty, multiplier));
}

// Emits shuffle-down reduction for the `partial_result_address` using the
// reduction computation `reducer`, writes output into
// `partial_result_address`.
//
// Multiple partial_result_address inputs happen when doing variadic
// reduction: each one should get the output value.
void EmitFullWarpShuffleDownLoopForReduce(
    llvm::IRBuilder<>* builder, IrEmitterContext& ir_emitter_context,
    const HloComputation* reducer,
    absl::Span<TypedPointer const> partial_result_addresses,
    int threads_per_block, int num_results_per_warp) {
  // This only works when the block size is a multiple of 32 threads.

  // We check this here as a mistake in the number of threads per
  // block is very hard to detect.
  CHECK_EQ(threads_per_block % 32, 0);
  CHECK_EQ(WarpSize() % num_results_per_warp, 0);

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
      auto convert_pointer_for_shuffle = [&](llvm::Value* ptr) {
        return builder->CreatePointerBitCastOrAddrSpaceCast(
            ptr, shuffled_value_type->getPointerTo());
      };

      llvm::Value* partial_result = builder->CreateLoad(
          shuffled_value_type,
          convert_pointer_for_shuffle(partial_result_address),
          "partial_reduction_result");
      builder->CreateStore(
          EmitFullWarpShuffleDown(partial_result, builder->getInt32(distance),
                                  builder),
          convert_pointer_for_shuffle(result_from_other_lane));
    }

    StatusOr<std::vector<llvm::Value*>> returned_scalars =
        CallNestedComputationWithScalarAddrs(builder, ir_emitter_context,
                                             *reducer, reduction_params);
    TF_CHECK_OK(returned_scalars.status());

    for (int i = 0; i < returned_scalars->size(); i++) {
      builder->CreateStore(/*Val=*/returned_scalars->at(i),
                           /*Ptr=*/partial_result_addresses[i].first);
    }
  }
}

llvm_ir::IrArray::Index GetOutputIndexForReduction(
    llvm::IRBuilder<>* builder, int partial_result_idx, llvm::Type* index_ty,
    const ReductionCodegenState& reduction_codegen_state,
    const TilingKernelInfo& tiling_kernel_info,
    const HloReduceInstruction* reduction, const HloInstruction* root,
    int output_idx) {
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  const TilingScheme& tiling_scheme = reduction_codegen_state.GetTilingScheme();
  const TilingThreadIdInfo& thread_id_info = tiling_kernel_info.thread_id_info;

  llvm_ir::IrArray::Index start_offset = [&] {
    llvm::Value* x_loc = thread_id_info.thread_id_x;
    llvm::Value* y_loc = thread_id_info.thread_id_y;
    if (!reduction_codegen_state.IsRowReduction()) {
      std::swap(x_loc, y_loc);
    }
    llvm::Value* start_offset_x =
        GetStartOffsetX(tiling_scheme, x_loc, index_ty, builder);
    return tiling_kernel_info.tile_origin
        .AddOffsetToDim(y_loc, TilingScheme::DimY, builder)
        .AddOffsetToDim(start_offset_x, TilingScheme::DimX, builder);
  }();

  const Shape& operand_shape = reduction->inputs()[output_idx]->shape();
  Shape reduction_kept_element_shape =
      ShapeUtil::DeleteDimensions(reduction->dimensions(), operand_shape);

  // Given the llvm_ir::IrArray index of a reduction input, returns the linear
  // address of the reduction output as if the reduction were going to keep
  // the input shape with the dimensions being reduced moved.
  llvm::Value* untransposed_output_linear_address = [&] {
    const llvm_ir::IrArray::Index index = start_offset.AddOffsetToDim(
        constant(partial_result_idx), TilingScheme::DimX, builder);
    if (reduction_codegen_state.IsRowReduction()) {
      // For row-reduction, y-coordinate determines which row we write into.
      return index[TilingScheme::DimY];
    }
    // For column reduction, we get the transposed address.
    absl::Span<const int64_t> dims_in_elem = tiling_scheme.GetDimsInElems();
    llvm::Value* x_dim_size =
        index.GetConstantWithIndexType(dims_in_elem[TilingScheme::DimX]);
    llvm::Value* x_block_offset =
        builder->CreateMul(index[TilingScheme::DimZ], x_dim_size);
    return builder->CreateAdd(x_block_offset, index[TilingScheme::DimX]);
  }();

  // A reduction is allowed to transpose its output.  For example, suppose
  // we are reducing the second dimension of f32[10,20,30]{3,2,1}.  We are
  // allowed to produce as output either f32[10,30]{1,0} (no transpose) or
  // f32[10,30]{0,1} (transposing the two output dims).
  //
  // At this point in the function we have a "partial sum" of input elements
  // (stored in partial_result_addresses), and we need to accumulate it into
  // the correct output element.
  llvm_ir::IrArray::Index element_index(
      /*linear=*/untransposed_output_linear_address,
      reduction_kept_element_shape, builder);
  const Shape& output_shape = !reduction->shape().IsTuple()
                                  ? reduction->shape()
                                  : reduction->shape().tuple_shapes(output_idx);
  llvm_ir::IrArray::Index output_index(element_index.multidim(), output_shape,
                                       element_index.GetType());
  // We need to check for root == reduction separately, because for variadic
  // reduce the root shape would be a tuple, while 'output_shape' is the
  // subshape.
  return (root == reduction ||
          ShapeUtil::EqualIgnoringElementType(output_shape, root->shape()))
             ? output_index
             : output_index.SourceIndexOfBitcast(output_shape, root->shape(),
                                                 builder);
}

llvm::Value* CastSharedToGlobal(llvm::IRBuilder<>* builder, llvm::Value* input,
                                llvm::Type* element_type, llvm::Twine name) {
  return builder->CreateAddrSpaceCast(
      input,
      llvm::PointerType::get(element_type,
                             /*AddressSpace=*/0),
      name);
}

void WriteReductionOutput(llvm::IRBuilder<>* builder,
                          IrEmitterContext& ir_emitter_context,
                          llvm::Type* index_ty,
                          const ReductionCodegenState& reduction_codegen_state,
                          const TilingKernelInfo& tiling_kernel_info,
                          const ReductionOutputMap& output_arrays,
                          const HloReduceInstruction* reduction,
                          const HloInstruction* root, int partial_result_idx,
                          const absl::Span<TypedPointer const> values,
                          ElementalIrEmitter& elemental_emitter) {
  const HloComputation* reducer = reduction->to_apply();
  for (const auto& [oidx, typed_ptr] : llvm::enumerate(values)) {
    auto [output_ptr, type] = typed_ptr;
    llvm_ir::IrArray::Index output_index = GetOutputIndexForReduction(
        builder, partial_result_idx, index_ty, reduction_codegen_state,
        tiling_kernel_info, reduction, root, oidx);

    llvm::Value* output_address =
        output_arrays.at(root)[oidx].EmitArrayElementAddress(
            output_index, builder, "output_element_address");
    if (reduction_codegen_state.IsRaceFree()) {
      FusedIrEmitter fused_emitter(elemental_emitter);
      llvm::Value* loaded = builder->CreateLoad(type, output_ptr, "output");
      fused_emitter.BindGenerator(
          *reduction,
          [&](const llvm_ir::IrArray::Index& index) { return loaded; });
      llvm_ir::ElementGenerator gen = *fused_emitter.GetGenerator(*root);
      llvm::Value* generated = *gen(output_index);
      builder->CreateStore(generated, output_address);
    } else {
      CHECK_EQ(values.size(), 1);
      CHECK_EQ(reduction, root)
          << "output fusion is not allowed for racing reductions";
      TF_CHECK_OK(EmitAtomicOperationForNestedComputation(
          builder, ir_emitter_context, *reducer, output_address, output_ptr,
          type));
    }
  }
}

// `current_output`: the value the tile has calculated.
// `output_address`: address where the output value has to be written.
void EmitReductionOutputForRowReduction(
    llvm::IRBuilder<>* builder, IrEmitterContext& ir_emitter_context,
    const TilingKernelInfo& tiling_kernel_info,
    const ReductionCodegenState& reduction_codegen_state, llvm::Type* index_ty,
    const ReductionOutputMap& output_arrays,
    const HloReduceInstruction* reduction, const HloInstruction* root,
    int partial_result_idx, ElementalIrEmitter& elemental_emitter) {
  const HloComputation* reducer = reduction->to_apply();
  const auto& thread_id_info = tiling_kernel_info.thread_id_info;
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  auto is_zero = [&](llvm::Value* value) {
    return builder->CreateICmpEQ(value, constant(0));
  };

  int num_outputs = reducer->num_parameters() / 2;
  const TilingScheme& tiling_scheme = reduction_codegen_state.GetTilingScheme();
  absl::InlinedVector<TypedPointer, 2> current_outputs;
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const ReductionCodegenState::ReductionCalculationState& state =
        reduction_codegen_state.GetCalculationStateFor(reduction, output_idx);
    current_outputs.push_back(
        {builder->CreateInBoundsGEP(
             state.partial_result_address->getAllocatedType(),
             state.partial_result_address, {constant(partial_result_idx)},
             "current_output"),
         state.partial_result_address->getAllocatedType()});
  }

  int reduced_dimension_size = tiling_scheme.GetDimsInElems()[2];
  int num_rows_per_warp = RowReductionGetRowsPerWarp(reduced_dimension_size);
  EmitFullWarpShuffleDownLoopForReduce(
      builder, ir_emitter_context, reducer, absl::MakeSpan(current_outputs),
      tiling_scheme.GetNumThreadsPerBlockPhysical(), num_rows_per_warp);

  KernelSupportLibrary ksl(builder);
  llvm::Value* warp_id =
      builder->CreateUDiv(thread_id_info.thread_id_x, constant(WarpSize()));

  auto emit_write_output = [&](llvm::Value* write_condition,
                               const absl::Span<TypedPointer const> values) {
    ksl.If("reduction_write_output", write_condition, [&] {
      WriteReductionOutput(builder, ir_emitter_context, index_ty,
                           reduction_codegen_state, tiling_kernel_info,
                           output_arrays, reduction, root, partial_result_idx,
                           values, elemental_emitter);
    });
  };

  if (num_rows_per_warp > 1) {
    llvm::Value* is_writing_thread = is_zero(builder->CreateAnd(
        thread_id_info.thread_id_x, constant(reduced_dimension_size - 1)));
    emit_write_output(is_writing_thread, current_outputs);
    return;
  }

  ksl.If("intra_warp_reduce_write", is_zero(thread_id_info.lane_id), [&] {
    for (int oidx = 0; oidx < num_outputs; oidx++) {
      const ReductionCodegenState::ReductionCalculationState& state =
          reduction_codegen_state.GetCalculationStateFor(reduction, oidx);
      llvm::Value* shmem_output_addr = thread_id_info.GEPIntoSharedMemory(
          builder, state.shared_cache, {constant(partial_result_idx), warp_id});
      builder->CreateStore(builder->CreateLoad(current_outputs[oidx].second,
                                               current_outputs[oidx].first),
                           shmem_output_addr);
    }
  });

  // TODO(cheshire): Don't we want to sync it once for everything in the
  // output? Not once per each?
  EmitSyncThreads(builder, ir_emitter_context);
  ksl.If("inter_warp_reduce", is_zero(warp_id), [&] {
    absl::InlinedVector<TypedPointer, 2> selected_values;
    for (int oidx = 0; oidx < num_outputs; oidx++) {
      const ReductionCodegenState::ReductionCalculationState& state =
          reduction_codegen_state.GetCalculationStateFor(reduction, oidx);
      llvm::Value* block_accum_addr = thread_id_info.GEPIntoSharedMemory(
          builder, state.shared_cache,
          {constant(partial_result_idx), thread_id_info.lane_id});

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
          thread_id_info.thread_id_x,
          constant(tiling_scheme.GetNumThreadsFor(TilingScheme::DimX) /
                   WarpSize()));

      llvm::Value* selected_value = builder->CreateSelect(
          warp_exists, block_accum_addr, initial_value_addr);

      selected_values.push_back({selected_value, element_type});
    }

    // If only one warp is present in the block, then we don't need inter-warp
    // reduction.
    // TODO(b/241414088) If only warp is present, then inter-warp
    // communication using shared memory and synchronization using barrier is
    // also unnecessary and should be removed.
    if (tiling_scheme.GetNumThreadsPerBlock() > WarpSize()) {
      EmitFullWarpShuffleDownLoopForReduce(
          builder, ir_emitter_context, reducer, absl::MakeSpan(selected_values),
          tiling_scheme.GetNumThreadsPerBlock(), /*num_results_per_warp=*/1);
    }

    emit_write_output(is_zero(thread_id_info.thread_id_x), selected_values);
  });
}

// Same arguments as EmitReductionOutputForRowReduction.
void EmitReductionOutputForColumnReduction(
    llvm::IRBuilder<>* builder, IrEmitterContext& ir_emitter_context,
    const TilingKernelInfo& tiling_kernel_info,
    const ReductionCodegenState& reduction_codegen_state, llvm::Type* index_ty,
    const ReductionOutputMap& output_arrays,
    const HloReduceInstruction* reduction, const HloInstruction* root,
    int partial_result_idx, ElementalIrEmitter& elemental_emitter) {
  KernelSupportLibrary ksl(builder);
  const HloComputation* reducer = reduction->to_apply();
  const auto& thread_id_info = tiling_kernel_info.thread_id_info;

  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  auto is_zero = [&](llvm::Value* value) {
    return builder->CreateICmpEQ(value, constant(0));
  };
  const TilingScheme& tiling_scheme = reduction_codegen_state.GetTilingScheme();
  int num_outputs = reducer->num_parameters() / 2;

  // Wait for reads from shmem in the last iteration to complete.  (If this is
  // slow, we could "double-buffer" by having two shmem buffers and switching
  // between them.)
  if (partial_result_idx > 0) {
    EmitSyncThreads(builder, ir_emitter_context);
  }

  // Store the transpose in shared memory.
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const ReductionCodegenState::ReductionCalculationState& state =
        reduction_codegen_state.GetCalculationStateFor(reduction, output_idx);
    llvm::GlobalVariable* shared_cache = state.shared_cache;
    llvm::AddrSpaceCastInst* shmem_output_addr =
        llvm::cast<llvm::AddrSpaceCastInst>(thread_id_info.GEPIntoSharedMemory(
            builder, shared_cache,
            {thread_id_info.thread_id_x, thread_id_info.thread_id_y},
            "shmem_output_address"));
    llvm::Value* current_output = builder->CreateInBoundsGEP(
        state.partial_result_address->getAllocatedType(),
        state.partial_result_address, {constant(partial_result_idx)},
        "current_output");

    llvm::Value* current_output_value = builder->CreateLoad(
        state.partial_result_address->getAllocatedType(), current_output);
    builder->CreateStore(current_output_value, shmem_output_addr);
  }

  EmitSyncThreads(builder, ir_emitter_context);

  // Get transposed element from shared memory.
  absl::InlinedVector<TypedPointer, 2> shmem_transposed_addrs;
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const ReductionCodegenState::ReductionCalculationState& state =
        reduction_codegen_state.GetCalculationStateFor(reduction, output_idx);
    llvm::AddrSpaceCastInst* shmem_transposed_addr =
        llvm::cast<llvm::AddrSpaceCastInst>(thread_id_info.GEPIntoSharedMemory(
            builder, state.shared_cache,
            {thread_id_info.thread_id_y, thread_id_info.thread_id_x},
            "shmem_transposed_addr"));
    shmem_transposed_addrs.push_back(
        {shmem_transposed_addr, llvm::cast<llvm::GetElementPtrInst>(
                                    shmem_transposed_addr->getPointerOperand())
                                    ->getResultElementType()});
  }

  EmitFullWarpShuffleDownLoopForReduce(builder, ir_emitter_context, reducer,
                                       absl::MakeSpan(shmem_transposed_addrs),
                                       tiling_scheme.GetNumThreadsPerBlock(),
                                       /*num_results_per_warp=*/1);

  // Some warps in the block are completely outside of the bound of the
  // tensor, so they should not write any output at all.
  llvm::Value* has_output = builder->CreateAnd(
      builder->CreateICmpULT(
          GetStartOffsetX(tiling_scheme, thread_id_info.thread_id_y, index_ty,
                          builder),
          tiling_kernel_info.output_tile_bounds[1]),
      builder->CreateICmpULT(thread_id_info.thread_id_x,
                             tiling_kernel_info.output_tile_bounds[0]));

  ksl.If("reduction_write_output",
         builder->CreateAnd(has_output, is_zero(thread_id_info.lane_id)), [&] {
           WriteReductionOutput(
               builder, ir_emitter_context, index_ty, reduction_codegen_state,
               tiling_kernel_info, output_arrays, reduction, root,
               partial_result_idx, shmem_transposed_addrs, elemental_emitter);
         });
}

// Generate a single element of the tile (update the accumulator state) for a
// given reducer of index `i`.
void GenerateElementForReducer(
    llvm::IRBuilder<>* builder, IrEmitterContext& ir_emitter_context,
    const HloReduceInstruction* reduction, llvm::Value* partial_result_index,
    const ReductionCodegenState& codegen_state,
    const llvm_ir::IrArray::Index& index_without_linear,
    const llvm_ir::IrArray::Index& input_index, int num_partial_results,
    const ReductionOutputMap& result_ir_arrays) {
  HloComputation* reducer = reduction->to_apply();
  CHECK_EQ(reducer->num_parameters() % 2, 0);

  absl::InlinedVector<llvm::Value*, 2> reduction_accumulators;
  absl::InlinedVector<llvm::Value*, 2> reduction_input_value;
  for (int red_idx = 0; red_idx < reducer->num_parameters() / 2; red_idx++) {
    const ReductionCodegenState::ReductionCalculationState& state =
        codegen_state.GetCalculationStateFor(reduction, red_idx);

    llvm::AllocaInst* input_address = state.input_address;
    llvm::AllocaInst* partial_reduction_result_address =
        state.partial_result_address;
    llvm::Value* const input_ir_value = *state.input_gen(
        num_partial_results > 1 ? index_without_linear : input_index);
    builder->CreateStore(input_ir_value, input_address);
    llvm::Value* partial_result_address = builder->CreateInBoundsGEP(
        partial_reduction_result_address->getAllocatedType(),
        partial_reduction_result_address, {partial_result_index});
    reduction_accumulators.push_back(partial_result_address);
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
  StatusOr<std::vector<llvm::Value*>> returned_scalars =
      CallNestedComputationWithScalarAddrs(builder, ir_emitter_context,
                                           *reducer, reduction_params);
  TF_CHECK_OK(returned_scalars.status());

  for (int i = 0; i < returned_scalars->size(); i++) {
    builder->CreateStore(returned_scalars->at(i), reduction_accumulators[i]);
  }
}

// Emits code for reductions in the output_instructions.
Status EmitIRForReduction(
    llvm::IRBuilder<>* builder, IrEmitterContext& ir_emitter_context,
    mlir::lmhlo::FusionOp fusion,
    absl::Span<const HloInstruction* const> instr_index_group,
    FusedIrEmitter& fused_emitter, const ReductionOutputMap& result_ir_arrays,
    const ReductionCodegenInfo& reduction_info, const Shape& input_shape,
    ElementalIrEmitter& elemental_emitter) {
  std::vector<const HloInstruction*> roots;
  std::vector<const HloReduceInstruction*> heroes;
  ExtraOutputGensMap extra_output_gens;

  for (const HloInstruction* hlo : instr_index_group) {
    auto& hero = FindNonTrivialHero(*hlo);
    if (IsRealReductionHero(*hlo, hero)) {
      auto reduction = Cast<HloReduceInstruction>(&hero);
      roots.push_back(hlo);
      heroes.push_back(reduction);
    } else {
      extra_output_gens[hlo] = *fused_emitter.GetGenerator(*hlo);
    }
  }

  CHECK(!heroes.empty()) << " expect at least one reduce instructions.";
  const TilingScheme& tiling_scheme = reduction_info.GetTilingScheme();
  CHECK_EQ(tiling_scheme.GetNumThreadsPerBlockPhysical() % WarpSize(), 0);
  llvm::Type* index_ty =
      GetIndexTypeForKernel(fusion,
                            tiling_scheme.GetNumThreadsPerBlockPhysical() *
                                tiling_scheme.GetNumberOfBlocksPhysical(),
                            builder);
  ReductionCodegenState codegen_state = GenerateReductionCodegenState(
      builder, fusion, reduction_info, heroes, fused_emitter);

  EmitTileElementFunction emit_reduction_element =
      [&](const TilingThreadIdInfo& thread_id_info,
          const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
          llvm::Value* x_loc) {
        llvm_ir::IrArray::Index input_index = GetUnnormalizedIndex(
            index, input_shape, builder,
            codegen_state.GetTilingScheme().GetDimsInElems());
        llvm::Value* partial_result_index =
            codegen_state.IsRowReduction()
                ? builder->getInt32(0)
                : builder->CreateSub(
                      x_loc,
                      GetStartOffsetX(tiling_scheme, thread_id_info.thread_id_x,
                                      index_ty, builder));

        // Clear the linear index field of the llvm_ir::IrArray::Index to enable
        // the use of GetElementPointer with array types. This enables the
        // vectorization of the computation for different partial results. Use
        // this index if 'num_partial_results > 1'.
        int num_partial_results = codegen_state.GetNumPartialResults();
        llvm_ir::IrArray::Index index_without_linear{
            input_index.multidim(), input_shape, input_index.GetType()};

        // Emit code to generate the input and perform the reduction computation
        // for each reduction instruction.
        for (const HloReduceInstruction* reduce : heroes) {
          GenerateElementForReducer(builder, ir_emitter_context, reduce,
                                    partial_result_index, codegen_state,
                                    index_without_linear, input_index,
                                    num_partial_results, result_ir_arrays);
        }

        // Emit code to generate the output for the non-reduction instructions
        // in the fusion, if any.
        TF_CHECK_OK(EmitExtraOutputsForReduce(
            builder, input_shape, result_ir_arrays, input_index, reduction_info,
            extra_output_gens));
      };

  TF_ASSIGN_OR_RETURN(
      TilingKernelInfo tiling_kernel_info,
      EmitTilingKernel(builder, tiling_scheme, index_ty,
                       [&](const TilingThreadIdInfo& thread_id_info,
                           const llvm_ir::IrArray::Index& index,
                           std::array<llvm::Value*, 2> tile_dimensions) {
                         EmitTile(builder, codegen_state.GetTilingScheme(),
                                  index, thread_id_info, tile_dimensions,
                                  emit_reduction_element);
                       }));

  KernelSupportLibrary ksl(builder);
  for (auto [reduce, root] : llvm::zip(heroes, roots)) {
    for (int partial_result_idx = 0;
         partial_result_idx < reduction_info.GetNumPartialResults();
         ++partial_result_idx) {
      if (codegen_state.IsRowReduction()) {
        EmitReductionOutputForRowReduction(
            builder, ir_emitter_context, tiling_kernel_info, codegen_state,
            index_ty, result_ir_arrays, reduce, root, partial_result_idx,
            elemental_emitter);
      } else {
        EmitReductionOutputForColumnReduction(
            builder, ir_emitter_context, tiling_kernel_info, codegen_state,
            index_ty, result_ir_arrays, reduce, root, partial_result_idx,
            elemental_emitter);
      }
    }
  }

  return OkStatus();
}

}  // namespace

StatusOr<FusionEmissionResult> ReductionFusion::Emit(
    IrEmitterContext& ir_emitter_context, ElementalIrEmitter& elemental_emitter,
    mlir::lmhlo::FusionOp fusion_op, const HloFusionInstruction& fusion,
    KernelReuseCache& kernel_cache, llvm::IRBuilder<>* builder) const {
  auto* reduction_codegen_info = analysis_.GetReductionCodegenInfo();
  TF_ASSIGN_OR_RETURN(auto launch_dimensions, analysis_.GetLaunchDimensions());

  FusionEmissionResult result;
  VLOG(3) << "Launch dimensions of "
          << mlir::mhlo::GetDebugNameFromLocation(fusion_op.getLoc()) << ": "
          << launch_dimensions.ToString();
  const HloComputation* fused_computation =
      fusion.fused_instructions_computation();
  if (!reduction_codegen_info->IsRaceFree()) {
    absl::Span<const HloInstruction* const> fusion_roots =
        analysis_.fusion_roots();
    for (int i = 0; i < fusion_roots.size(); ++i) {
      if (IsReductionFromOrToContiguousDimensions(*fusion_roots[i])) {
        TF_ASSIGN_OR_RETURN(
            result.thunks.emplace_back(),
            BuildFusedInitializerThunk(ir_emitter_context, fusion_op,
                                       fused_computation, elemental_emitter,
                                       kernel_cache, i, builder));
      }
    }
  }

  auto builder_fn = [&, this](std::vector<llvm_ir::IrArray> inputs,
                              std::vector<llvm_ir::IrArray> outputs) -> Status {
    FusedIrEmitter fused_emitter(elemental_emitter);
    for (int i = 0; i < fused_computation->num_parameters(); i++) {
      HloInstruction* fused_operand =
          fused_computation->parameter_instruction(i);
      fused_emitter.BindGenerator(*fused_operand,
                                  [builder, input = inputs[i], fused_operand](
                                      const llvm_ir::IrArray::Index& index) {
                                    return input.EmitReadArrayElement(
                                        index, builder, fused_operand->name());
                                  });
    }

    // Get outputs.
    ReductionOutputMap result_ir_arrays;

    int ir_arrays_idx = 0;
    for (const HloInstruction* root : analysis_.fusion_roots()) {
      int get_num_results = GetNumOutputs(root->shape());
      result_ir_arrays[root] =
          absl::MakeSpan(outputs).subspan(ir_arrays_idx, get_num_results);
      ir_arrays_idx += get_num_results;
    }

    KernelSupportLibrary ksl(builder, llvm_ir::UnrollMode::kDefaultUnroll);

    // Use raw block_id_y to select the i-th parallel reduction to run. Using
    // block_id_y instead of block_id_x simplifies the index calculation
    // for reduction code generation as the block_id_y is orthogonal to
    // the indices used within the reductions.
    const std::vector<std::vector<const HloInstruction*>>& instr_index_groups =
        reduction_codegen_info->GetIndexGroups();
    Shape reduce_operand_shape =
        reduction_codegen_info->GetReduceOperandShape();

    llvm::CallInst* raw_block_id_y = gpu::EmitCallToTargetIntrinsic(
        gpu::TargetIntrinsicID::kBlockIdy, {}, {}, builder);
    llvm_ir::AddRangeMetadata(0, instr_index_groups.size(),
                              llvm::cast<llvm::Instruction>(raw_block_id_y));
    for (int i = 0; i < instr_index_groups.size(); ++i) {
      TF_RETURN_IF_ERROR(ksl.IfWithStatus(
          absl::StrCat("reduce-group-", i),
          builder->CreateICmpEQ(raw_block_id_y, builder->getInt32(i)), [&] {
            return EmitIRForReduction(builder, ir_emitter_context, fusion_op,
                                      instr_index_groups[i], fused_emitter,
                                      result_ir_arrays, *reduction_codegen_info,
                                      reduce_operand_shape, elemental_emitter);
          }));
    }

    return OkStatus();
  };

  TF_ASSIGN_OR_RETURN(
      result.thunks.emplace_back(),
      BuildKernelThunkForFusion(ir_emitter_context, kernel_cache, fusion_op,
                                fused_computation, launch_dimensions, "",
                                builder_fn, builder));
  return result;
}

}  // namespace gpu
}  // namespace xla
